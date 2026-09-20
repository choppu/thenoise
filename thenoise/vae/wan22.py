# Wan 2.2 VAE (AutoencoderKL) — 2D still-image port.
#
# Ported from ComfyUI's ``comfy/ldm/wan/vae2_2.py`` (itself from the official
# ``Wan-Video/Wan2.2`` release, ``wan/modules/vae2_2.py``). The reference is a
# video VAE; this port keeps only its single-frame (4D) code path, which is
# what Qwen-Image-2.1 uses:
#
#   * every causal 3D conv becomes a plain ``Conv2d``. Causal time padding
#     places a single frame at the *end* of the time axis, so only the last
#     time slice of each ``(3, 3, 3)`` kernel contributes; the loader collapses
#     the 5D weights by that slice;
#   * the per-chunk ``time_conv`` layers and the whole ``feat_cache`` streaming
#     machinery are dropped — they only take effect on multi-frame input;
#   * the parameter-free ``AvgDown3D`` / ``DupUp3D`` channel-to-block shortcuts
#     are folded into their exact single-frame behaviour (see the classes).
#
# Net effect: a ``[B, 3, H, W]`` image in ``[-1, 1]`` maps to a
# ``[B, 48, H // 16, W // 16]`` latent and back (2x2 patchify + 8x spatial
# compression), identical to the reference's ``WanVAE.encode`` / ``decode`` on
# 4D inputs (decoder run with ``first_chunk=True``).
#
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from thenoise.utils.safetensors import load_safetensors

from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``[B, C, H*p, W*p]`` -> ``[B, C*p*p, H, W]`` (row-major 2x2 blocks)."""
    if patch_size == 1:
        return x
    return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    if patch_size == 1:
        return x
    return rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)


class Wan22RMSNorm(nn.Module):
    """RMS normalization over the channel dim, gamma shape ``(C, 1, 1)``.

    The 3D reference keeps a trailing time dim on the resnet/head gammas
    (``(C, 1, 1, 1)``); it is squeezed away at load time. The math is
    ``x * rsqrt(mean(x^2)) * sqrt(C) * gamma`` (``F.normalize(x, dim=1)``).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class Wan22ResidualBlock(nn.Module):
    """v2_2 residual block with the checkpoint's key layout.

    ``residual`` is a fixed 7-slot sequential (RMS, SiLU, conv, RMS, SiLU,
    dropout, conv) and ``shortcut`` is a 1x1 conv (or identity) — matching
    ``*.residual.{0,2,3,6}.*`` / ``*.shortcut.*`` in the weights.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            Wan22RMSNorm(in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            Wan22RMSNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class Wan22AttentionBlock(nn.Module):
    """Single-head spatial attention with 1x1 QKV convs (``norm``/``to_qkv``/``proj``)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = Wan22RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.size()
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.view(b, c, -1).transpose(1, 2)  # (b, h*w, c)
        k = k.view(b, c, -1).transpose(1, 2)
        v = v.view(b, c, -1).transpose(1, 2)
        # Manual single-head attention instead of ``F.scaled_dot_product_attention``:
        # on ROCm the fused SDPA backends produce localized broken-pixel artifacts in
        # VAE decoders (see thenoise/vae/qwen_image.py).
        scale = c**0.5  # SDPA default scale = 1/sqrt(head_dim); single head
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.proj(x) + identity


class Wan22Resample(nn.Module):
    """Spatial 2x up/down resampling — the 2D part of the v2_2 ``Resample``.

    ``upsample3d``/``downsample3d`` differ from their 2D counterparts only in the
    ``time_conv`` layer, which the single-frame path never runs; both therefore
    collapse to the same 2D stack here.
    """

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        else:
            raise ValueError(f"unsupported resample mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class Wan22AvgDown(nn.Module):
    """Parameter-free single-frame fold of the v2_2 ``AvgDown3D`` encoder shortcut.

    Averages ``factor_s x factor_s`` spatial blocks over channel groups, exactly
    as the reference does for one input frame. The reference's channel groups are
    ordered ``(c, t, r, s)`` — for each input channel, all spatial sub-blocks of
    time slot ``t=0``, then all sub-blocks of slot ``t=1``, and so on — and the
    causal padding places a single input frame at the *end* of the time axis,
    so with ``factor_t=2`` the ``t=0`` slot is a zero-padded frame. With
    ``in_dim < out_dim`` (the temporal-downsample stages) a group then spans
    *less* than one full ``(t, r, s)`` cell per channel: for the 160->320 and
    320->640 stages every *even* output channel is fed by the zero-padded frame
    only (and is zero) while every *odd* one averages the 2x2 block of its
    paired input channel. Building the zero-padded block explicitly below
    reproduces all of that bit for bit.
    """

    def __init__(self, in_channels: int, out_channels: int, factor_t: int = 1, factor_s: int = 2) -> None:
        super().__init__()
        assert in_channels * factor_t * factor_s * factor_s % out_channels == 0
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.group_size = in_channels * factor_t * factor_s * factor_s // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        if fs == 1:
            block = x.view(b, c, 1, h, w)  # (b, c, fs*fs, h, w) with a single cell
        else:
            # (b, c, r, s, h/fs, w/fs) -> (b, c, fs*fs, h/fs, w/fs), groups in (c, r, s) order
            block = x.view(b, c, h // fs, fs, w // fs, fs)
            block = block.permute(0, 1, 3, 5, 2, 4).reshape(b, c, fs * fs, h // fs, w // fs)
        if ft > 1:
            # the reference zero-pads the time axis at the front, so in (c, t, r, s)
            # group order each real (c, r, s) block is preceded by its zero copies
            block = torch.cat(
                [block.new_zeros(b, c, ft - 1, fs * fs, h // fs, w // fs), block.unsqueeze(2)], dim=2
            )
        block = block.reshape(b, c * ft * fs * fs, h // fs, w // fs)
        return block.view(b, self.out_channels, self.group_size, h // fs, w // fs).mean(dim=2)


class Wan22DupUp(nn.Module):
    """Parameter-free single-frame fold of the v2_2 ``DupUp3D`` decoder shortcut.

    Duplicates ``factor_s x factor_s`` spatial samples per output channel group,
    via the reference's exact ``repeat_interleave`` + channel grouping. The
    reference interleaves each group over a ``(t, r, s)`` cell and the image
    path (``first_chunk=True``) keeps only the *last* temporal frame, so the
    ``t = factor_t - 1`` cell is selected below. When ``in_dim == out_dim`` the
    cell is the same input channel repeated, i.e. a plain 2x2 duplicate; when
    ``in_dim > out_dim`` (1024->512) each 4-way cell holds two input channels,
    each twice.
    """

    def __init__(self, in_channels: int, out_channels: int, factor_t: int = 1, factor_s: int = 2) -> None:
        super().__init__()
        assert out_channels * factor_t * factor_s * factor_s % in_channels == 0
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.repeats = out_channels * factor_t * factor_s * factor_s // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.repeat_interleave(self.repeats, dim=1)  # (b, out*ft*fs*fs, h, w)
        x = x.view(b, self.out_channels, ft, fs, fs, h, w)
        x = x[:, :, ft - 1, :, :, :, :]  # first_chunk: keep the last temporal frame
        x = x.permute(0, 1, 4, 2, 5, 3)  # (b, out, h, r, w, s)
        return x.reshape(b, self.out_channels, h * fs, w * fs)


class Wan22DownBlock(nn.Module):
    """v2_2 ``Down_ResidualBlock``: residual stack + resample, plus the
    parameter-free ``avg_shortcut`` (no checkpoint keys of its own)."""

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temperal_downsample: bool, downsample: bool) -> None:
        super().__init__()
        self.avg_shortcut = Wan22AvgDown(
            in_dim, out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if downsample else 1,
        )
        layers: list[nn.Module] = []
        in_c = in_dim
        for _ in range(num_res_blocks):
            layers.append(Wan22ResidualBlock(in_c, out_dim))
            in_c = out_dim
        if downsample:
            layers.append(Wan22Resample(out_dim, "downsample3d" if temperal_downsample else "downsample2d"))
        self.downsamples = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.avg_shortcut(x)
        for layer in self.downsamples:
            x = layer(x)
        return x + h


class Wan22UpBlock(nn.Module):
    """v2_2 ``Up_ResidualBlock``: residual stack + resample, plus the
    parameter-free ``avg_shortcut`` (no checkpoint keys of its own)."""

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temperal_upsample: bool, upsample: bool) -> None:
        super().__init__()
        self.avg_shortcut = (
            Wan22DupUp(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2)
            if upsample
            else None
        )
        layers: list[nn.Module] = []
        in_c = in_dim
        for _ in range(num_res_blocks + 1):
            layers.append(Wan22ResidualBlock(in_c, out_dim))
            in_c = out_dim
        if upsample:
            layers.append(Wan22Resample(out_dim, "upsample3d" if temperal_upsample else "upsample2d"))
        self.upsamples = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.upsamples:
            h = layer(h)
        if self.avg_shortcut is not None:
            h = h + self.avg_shortcut(x)
        return h


class Wan22Encoder(nn.Module):
    """2D encoder (v2_2 ``Encoder3d`` without the video cache path)."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: List[int],
        num_res_blocks: int,
        in_channels: int,
        temperal_downsample: List[bool],
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]

        self.conv1 = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        self.downsamples = nn.Sequential(
            *[
                Wan22DownBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_downsample[i] if i < len(temperal_downsample) else False,
                    downsample=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

        out_dim = dims[-1]
        self.middle = nn.Sequential(
            Wan22ResidualBlock(out_dim, out_dim),
            Wan22AttentionBlock(out_dim),
            Wan22ResidualBlock(out_dim, out_dim),
        )
        self.head = nn.Sequential(
            Wan22RMSNorm(out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Wan22Decoder(nn.Module):
    """2D decoder (v2_2 ``Decoder3d`` without the video cache path).

    ``temperal_upsample`` is the reverse of the encoder's ``temperal_downsample``
    (see the reference); the decoder base width is its own ``dec_dim``.
    """

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: List[int],
        num_res_blocks: int,
        out_channels: int,
        temperal_upsample: List[bool],
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult)[::-1]]

        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            Wan22ResidualBlock(dims[0], dims[0]),
            Wan22AttentionBlock(dims[0]),
            Wan22ResidualBlock(dims[0], dims[0]),
        )

        self.upsamples = nn.Sequential(
            *[
                Wan22UpBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_upsample[i] if i < len(temperal_upsample) else False,
                    upsample=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

        out_dim = dims[-1]
        self.head = nn.Sequential(
            Wan22RMSNorm(out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


#: Official Wan 2.2 latent statistics (48 channels), from the ``Wan2_2_VAE``
#: wrapper of the Wan-Video/Wan2.2 release. The VAE normalises latents as
#: ``(mu - mean) / std`` before handing them to the model and undoes it before
#: decoding.
WAN22_LATENTS_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]

WAN22_LATENTS_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


class AutoencoderKLWan22(nn.Module):
    r"""Wan 2.2 VAE, still-image (2D) inference.

    Args:
        dim (int): Encoder base width (160 for the released weights).
        dec_dim (int): Decoder base width (256 — the v2_2 decoder is wider than
            the encoder).
        z_dim (int): Latent channel count (48).
        dim_mult (list[int]): Channel multipliers of the 4 down/up stages.
        num_res_blocks (int): Residual blocks per encoder stage (the decoder
            uses ``num_res_blocks + 1``).
        temperal_downsample (list[bool]): Per-stage temporal-downsample flags of
            the *video* model; for a single frame they only decide which
            channel grouping the parameter-free shortcuts use.
        image_channels (int): Pixel channels (3).
        patch_size (int): 2x2 pixel patchify factor (2).
        latents_mean / latents_std (list[float]): Latent affine statistics, see
            :meth:`encode_pixels_to_latents` / :meth:`decode_to_pixels`.
    """

    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 48,
        dim_mult: List[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: List[bool] = (False, True, True),
        image_channels: int = 3,
        patch_size: int = 2,
        latents_mean: Optional[List[float]] = None,
        latents_std: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        dim_mult = list(dim_mult)
        if latents_mean is None:
            latents_mean = WAN22_LATENTS_MEAN
        if latents_std is None:
            latents_std = WAN22_LATENTS_STD
        assert len(latents_mean) == z_dim and len(latents_std) == z_dim

        self.dim_mult = dim_mult
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        # Hoisted buffers (built once; moved with the module via `.to(device)`).
        self.register_buffer(
            "_latents_mean", torch.tensor(latents_mean).view(1, z_dim, 1, 1), persistent=False
        )
        self.register_buffer(
            "_latents_inv_std", (1.0 / torch.tensor(latents_std)).view(1, z_dim, 1, 1), persistent=False
        )

        self.encoder = Wan22Encoder(
            dim, z_dim * 2, dim_mult, num_res_blocks,
            image_channels * patch_size * patch_size, temperal_downsample,
        )
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Wan22Decoder(
            dec_dim, z_dim, dim_mult, num_res_blocks,
            image_channels * patch_size * patch_size, temperal_downsample[::-1],
        )

    @property
    def dtype(self):
        return self.encoder.parameters().__next__().dtype

    @property
    def device(self):
        return self.encoder.parameters().__next__().device

    @property
    def compression(self) -> int:
        """Spatial compression factor: patchify x 2^(num_stages-1), e.g. 2 * 8 = 16."""
        return self.patch_size * 2 ** (len(self.dim_mult) - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images to (deterministic) latents.

        Args:
            x (`torch.Tensor`): Input images in ``[-1, 1]``, shape ``[B, C, H, W]``.

        Returns:
            The latent mean ``[B, z, H // 16, W // 16]`` (the reference's 4D
            path returns only the mean; no sampling).
        """
        x = patchify(x, self.patch_size)
        return self.conv1(self.encoder(x)).chunk(2, dim=1)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""Decode latents to pixels (unclamped).

        Args:
            z (`torch.Tensor`): Latents, shape ``[B, z, H, W]``.

        Returns:
            Pixels in ~``[-1, 1]``, shape ``[B, C, 16H, 16W]``.
        """
        x = self.decoder(self.conv2(z))
        return unpatchify(x, self.patch_size)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels in ``[-1, 1]`` (``[B, C, H, W]``) -> normalised latents."""
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        latents = self.encode(pixels)
        latents_mean = self._latents_mean.to(latents.device, latents.dtype)
        latents_inv_std = self._latents_inv_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) * latents_inv_std

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalised latents (``[B, z, H, W]``) -> pixels clamped to ``[-1, 1]``."""
        latents = latents.to(self.dtype)
        latents_mean = self._latents_mean.to(latents.device, latents.dtype)
        latents_inv_std = self._latents_inv_std.to(latents.device, latents.dtype)
        latents = latents / latents_inv_std + latents_mean
        return self.decode(latents).clamp(-1.0, 1.0)


def load_wan22_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    latents_mean: Optional[List[float]] = None,
    latents_std: Optional[List[float]] = None,
) -> AutoencoderKLWan22:
    """Load a Wan 2.2 VAE (ComfyUI ``vae2_2`` key layout) for single-frame use.

    The variable architecture parameters (``dim``/``dec_dim``/``z_dim``, image
    channels, per-stage temporal flags) are read from the checkpoint itself, the
    same way ComfyUI detects the layout, then the video weight layout is
    collapsed to 2D:

      * the never-used-for-images ``time_conv`` layers are dropped;
      * every 5D conv weight becomes its **last** time slice — causal padding
        pads the time axis by ``(2, 0)``, so the single frame lands at the end
        and only the last kernel slice contributes;
      * resnet/head RMS gammas ``(C, 1, 1, 1)`` are squeezed to ``(C, 1, 1)``
        (the attention norm gammas are already ``(C, 1, 1)``).
    """
    logger.info("Initializing VAE")
    state_dict = load_safetensors(vae_path, device=device)

    required = (
        "encoder.conv1.weight",
        "decoder.conv1.weight",
        "conv1.weight",
        "conv2.weight",
        "encoder.head.0.gamma",
        "encoder.head.2.weight",
        "decoder.middle.0.residual.0.gamma",
        "decoder.head.2.weight",
    )
    for key in required:
        if key not in state_dict:
            raise ValueError(f"'{key}' not found in {vae_path} (not a Wan 2.2 VAE?)")

    dim_mult = [1, 2, 4, 4]
    num_res_blocks = 2

    # Architecture parameters that vary across Wan-2.2-family checkpoints.
    dim = state_dict["encoder.conv1.weight"].shape[0]
    dec_dim = state_dict["decoder.head.0.gamma"].shape[0]
    z_dim = state_dict["conv2.weight"].shape[0]
    in_channels = state_dict["encoder.conv1.weight"].shape[1]
    out_channels = state_dict["decoder.head.2.weight"].shape[0]
    patch_size = 2
    if in_channels != out_channels or in_channels % (patch_size * patch_size):
        raise ValueError(
            f"unexpected in/out channels {in_channels}/{out_channels} in {vae_path}"
        )
    image_channels = in_channels // (patch_size * patch_size)

    # Consistency of the fixed layout against the actual weights.
    if state_dict["encoder.head.0.gamma"].shape[0] != dim * dim_mult[-1]:
        raise ValueError(f"encoder width {state_dict['encoder.head.0.gamma'].shape[0]} != dim*mult[-1] {dim * dim_mult[-1]}")
    if state_dict["decoder.conv1.weight"].shape[0] != dec_dim * dim_mult[-1]:
        raise ValueError(f"decoder width {state_dict['decoder.conv1.weight'].shape[0]} != dec_dim*mult[-1] {dec_dim * dim_mult[-1]}")
    if state_dict["conv1.weight"].shape != (z_dim * 2, z_dim * 2, 1, 1, 1):
        raise ValueError(f"unexpected conv1 shape {tuple(state_dict['conv1.weight'].shape)}")

    # The per-stage temporal flags are recorded in the checkpoint by the
    # presence of each resample's ``time_conv`` (video-only; dropped below).
    enc_flags = [
        f"encoder.downsamples.{i}.downsamples.{num_res_blocks}.time_conv.weight" in state_dict
        for i in range(len(dim_mult) - 1)
    ]
    dec_flags = [
        f"decoder.upsamples.{i}.upsamples.{num_res_blocks + 1}.time_conv.weight" in state_dict
        for i in range(len(dim_mult) - 1)
    ]
    if dec_flags != enc_flags[::-1]:
        raise ValueError(f"inconsistent temporal flags in {vae_path}: encoder {enc_flags}, decoder {dec_flags}")

    vae = AutoencoderKLWan22(
        dim=dim,
        dec_dim=dec_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        temperal_downsample=enc_flags,
        image_channels=image_channels,
        patch_size=patch_size,
        latents_mean=latents_mean,
        latents_std=latents_std,
    )

    # Collapse the video weight layout to 2D (see docstring).
    state_dict = {k: v for k, v in state_dict.items() if ".time_conv." not in k}
    for key in state_dict:
        val = state_dict[key]
        if val.dim() == 5:
            state_dict[key] = val[:, :, -1]
        elif key.endswith(".gamma") and val.dim() == 4:
            state_dict[key] = val.reshape(val.shape[0], 1, 1)

    logger.info(
        "Loading VAE from %s (dim=%d dec_dim=%d z_dim=%d channels=%d flags=%s)",
        vae_path, dim, dec_dim, z_dim, image_channels, enc_flags,
    )
    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded VAE: {info}")

    vae.to(device)
    if dtype is not None:
        vae.to(dtype)
    return vae.eval().requires_grad_(False)
