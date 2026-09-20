"""The Wan 2.2 VAE (2D still-image port, used by Qwen-Image-2.1).

The tests cover the 16x encode/decode geometry, the documented latent
normalisation, the exact single-frame folds of the reference's
``AvgDown3D``/``DupUp3D`` shortcuts (checked against a verbatim copy of the
official 5D code run on a one-frame input), and the checkpoint loader's
weight collapse + architecture inference. A reduced-channel instance keeps
everything sub-second; the shipped constants are asserted on the class.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import write_safetensors
from thenoise.vae import AutoencoderKLWan22, load_wan22_vae
from thenoise.vae.wan22 import (
    WAN22_LATENTS_MEAN,
    WAN22_LATENTS_STD,
    Wan22AvgDown,
    Wan22DupUp,
)

TINY_MEAN = [0.1, 0.2, 0.3, 0.4]
TINY_STD = [2.0, 4.0, 8.0, 16.0]


def _tiny_vae() -> AutoencoderKLWan22:
    """The shipped architecture shape with 8-channel blocks (random weights)."""
    return AutoencoderKLWan22(
        dim=8, dec_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        latents_mean=TINY_MEAN, latents_std=TINY_STD,
    )


@pytest.fixture(scope="module")
def tiny_vae():
    return _tiny_vae().eval().requires_grad_(False)


# --------------------------------------------------------------- documented facts


def test_documented_shape_constants():
    assert len(WAN22_LATENTS_MEAN) == 48
    assert len(WAN22_LATENTS_STD) == 48


def test_compression_is_16x(tiny_vae):
    assert tiny_vae.compression == 16  # 2x2 patchify + 8x spatial (three downsample stages)


def test_dtype_and_device_follow_the_weights(tiny_vae):
    assert tiny_vae.dtype == tiny_vae.encoder.parameters().__next__().dtype
    assert tiny_vae.device == tiny_vae.encoder.parameters().__next__().device


# ------------------------------------------------------------------- geometry


def test_encode_compresses_16x_and_returns_the_mean(tiny_vae):
    with torch.no_grad():
        latents = tiny_vae.encode(torch.randn(1, 3, 64, 64))
    assert latents.shape == (1, 4, 4, 4)  # 64/16 = 4
    assert torch.isfinite(latents).all()


def test_decode_upsamples_16x(tiny_vae):
    with torch.no_grad():
        pixels = tiny_vae.decode(torch.randn(1, 4, 2, 2))
    assert pixels.shape == (1, 3, 32, 32)
    assert torch.isfinite(pixels).all()


def test_roundtrip_is_finite_and_clamped(tiny_vae):
    with torch.no_grad():
        pixels = torch.randn(1, 3, 32, 48)
        out = tiny_vae.decode_to_pixels(tiny_vae.encode_pixels_to_latents(pixels))
    assert out.shape == pixels.shape
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0


# ---------------------------------------------------------- latent normalisation


def test_decode_to_pixels_denormalises_before_the_decoder(tiny_vae, monkeypatch):
    """``decode_to_pixels`` maps canonical latents back through ``z * std + mean``."""
    seen = {}

    def spy_decode(z):
        seen["z"] = z.clone()
        return torch.zeros(1, 3, 16, 16)

    monkeypatch.setattr(tiny_vae, "decode", spy_decode)
    latents = torch.full((1, 4, 1, 1), 0.5)
    tiny_vae.decode_to_pixels(latents)

    expected = torch.tensor(TINY_STD).view(1, 4, 1, 1) * 0.5 + torch.tensor(TINY_MEAN).view(1, 4, 1, 1)
    assert torch.allclose(seen["z"].float(), expected, atol=1e-6)


def test_encode_pixels_to_latents_normalises_after_encode(tiny_vae, monkeypatch):
    """``encode_pixels_to_latents`` applies ``(mu - mean) / std`` to the encoder mean."""
    mu = torch.full((1, 4, 1, 1), 2.0)

    def spy_encode(x):
        assert x.shape == (1, 3, 32, 32)
        return mu

    monkeypatch.setattr(tiny_vae, "encode", spy_encode)
    latents = tiny_vae.encode_pixels_to_latents(torch.zeros(1, 3, 32, 32))

    expected = (torch.tensor(2.0) - torch.tensor(TINY_MEAN).view(1, 4, 1, 1)) / torch.tensor(TINY_STD).view(1, 4, 1, 1)
    assert torch.allclose(latents.float(), expected, atol=1e-6)


# ------------------------------------------------ single-frame shortcut folds
# Verbatim copies of the official Wan 2.2 ``AvgDown3D`` / ``DupUp3D`` forwards
# (parameter-free), used as the reference on a one-frame 5D input.


def _ref_avgdown3d(x, in_channels, out_channels, factor_t, factor_s=1):
    pad_t = (factor_t - x.shape[2] % factor_t) % factor_t
    pad = (0, 0, 0, 0, pad_t, 0)
    x = F.pad(x, pad)
    B, C, T, H, W = x.shape
    x = x.view(
        B, C, T // factor_t, factor_t, H // factor_s, factor_s, W // factor_s, factor_s,
    )
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(B, C * factor_t * factor_s * factor_s, T // factor_t, H // factor_s, W // factor_s)
    x = x.view(B, out_channels, in_channels * factor_t * factor_s * factor_s // out_channels,
               T // factor_t, H // factor_s, W // factor_s)
    return x.mean(dim=2)


def _ref_dupup3d(x, in_channels, out_channels, factor_t, factor_s=1, first_chunk=False):
    repeats = out_channels * factor_t * factor_s * factor_s // in_channels
    x = x.repeat_interleave(repeats, dim=1)
    x = x.view(x.size(0), out_channels, factor_t, factor_s, factor_s, x.size(2), x.size(3), x.size(4))
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(x.size(0), out_channels, x.size(2) * factor_t, x.size(4) * factor_s, x.size(6) * factor_s)
    if first_chunk:
        x = x[:, :, factor_t - 1:, :, :]
    return x


def test_avgdown_matches_the_reference_single_frame():
    """The 2D fold must equal the official 5D shortcut on a one-frame input."""
    gen = torch.Generator().manual_seed(0)
    # (in, out, factor_t, factor_s) — covers identity, spatial-only, temporal
    # stages with in < out (groups split across channels) and in == out.
    for c, o, ft, fs in [
        (16, 16, 1, 1),
        (16, 16, 1, 2),
        (16, 16, 2, 2),
        (16, 32, 2, 2),
        (32, 64, 2, 2),
        (32, 32, 1, 2),
        (32, 16, 1, 2),
        (16, 32, 2, 1),
    ]:
        x = torch.randn(2, c, 8, 8, generator=gen)
        ref = _ref_avgdown3d(x.unsqueeze(2), c, o, ft, fs).squeeze(2)
        got = Wan22AvgDown(c, o, factor_t=ft, factor_s=fs)(x)
        assert torch.allclose(got, ref, atol=1e-6), (c, o, ft, fs)


def test_avgdown_temporal_stage_zeroes_and_keeps_groups():
    """With ``factor_t=2`` and ``in < out`` the front zero-padded time slot lands
    *inside* the channel groups: even output channels are zero, odd ones keep
    the 2x2 mean of their paired input channel."""
    x = torch.rand(1, 16, 8, 8) + 1.0  # strictly positive, so the zero frame is visible
    out = Wan22AvgDown(16, 32, factor_t=2, factor_s=2)(x)
    spatial_mean = F.avg_pool2d(x, 2)
    assert torch.allclose(out[:, 1::2], spatial_mean, atol=1e-6)
    assert torch.all(out[:, 0::2] == 0)


def test_dupup_matches_the_reference_single_frame():
    """The 2D fold must equal the official 5D shortcut with ``first_chunk=True``."""
    gen = torch.Generator().manual_seed(1)
    for c, o, ft in [(16, 16, 2), (16, 16, 1), (32, 16, 1), (32, 16, 2)]:
        x = torch.randn(2, c, 6, 7, generator=gen)
        ref = _ref_dupup3d(x.unsqueeze(2), c, o, ft, 2, first_chunk=True).squeeze(2)
        got = Wan22DupUp(c, o, factor_t=ft, factor_s=2)(x)
        assert torch.allclose(got, ref, atol=1e-6), (c, o, ft)
        assert got.shape == (2, o, 12, 14)


def test_dupup_equals_nearest_duplicate_when_in_equals_out():
    x = torch.randn(1, 16, 5, 6)
    out = Wan22DupUp(16, 16, factor_t=2, factor_s=2)(x)
    assert torch.equal(out, x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))


# ------------------------------------------------------------------- the loader


def _as_video_state_dict(sd: dict, enc_flags: list, dec_flags: list, dim, dec_dim) -> dict:
    """Re-expand a 2D state dict into the checkpoint's 5D video layout.

    Non-last time slices are filled with garbage on purpose: the loader must
    pick the last slice, not a specific one.
    """
    num_res_blocks = 2
    out = {}
    for key, val in sd.items():
        if val.dim() == 3 and key.endswith(".gamma"):
            # checkpoint gammas: 4D, except the 3D attention norm
            out[key] = val.unsqueeze(-1) if ".norm.gamma" not in key else val
        elif val.dim() == 4:  # conv weight (out, in, kh, kw)
            time = 3 if val.shape[-2:] == (3, 3) else 1  # 3x3 vs 1x1 convs
            w = torch.full((val.shape[0], val.shape[1], time, val.shape[2], val.shape[3]), -12345.0)
            w[:, :, -1] = val
            out[key] = w
        else:
            out[key] = val
    # video-only time_conv layers, present exactly on the temporal stages
    for i, flag in enumerate(enc_flags):
        if not flag:
            continue
        out_c = dim * [1, 2, 4, 4][i]
        out[f"encoder.downsamples.{i}.downsamples.{num_res_blocks}.time_conv.weight"] = torch.zeros(out_c, out_c, 3, 1, 1)
        out[f"encoder.downsamples.{i}.downsamples.{num_res_blocks}.time_conv.bias"] = torch.zeros(out_c)
    for i, flag in enumerate(dec_flags):
        if not flag:
            continue
        out_c = dec_dim * [4, 4, 2, 1][i]
        out[f"decoder.upsamples.{i}.upsamples.{num_res_blocks + 1}.time_conv.weight"] = torch.zeros(2 * out_c, out_c, 3, 1, 1)
        out[f"decoder.upsamples.{i}.upsamples.{num_res_blocks + 1}.time_conv.bias"] = torch.zeros(2 * out_c)
    return out


def test_load_wan22_vae_collapses_and_infers(tmp_path):
    source = _tiny_vae()
    video_sd = _as_video_state_dict(
        source.state_dict(), enc_flags=[False, True, True], dec_flags=[True, True, False],
        dim=8, dec_dim=8,
    )
    assert any(".time_conv." in k for k in video_sd)
    path = write_safetensors(tmp_path / "wan22.safetensors", video_sd)

    vae = load_wan22_vae(path, device="cpu", latents_mean=TINY_MEAN, latents_std=TINY_STD)

    assert isinstance(vae, AutoencoderKLWan22)
    assert vae.training is False
    # architecture inferred from the weights
    assert vae.z_dim == 4
    assert vae.encoder.conv1.weight.shape[1] == 12  # 3 channels x 2x2 patchify
    assert vae.decoder.head[-1].weight.shape[0] == 12
    # temporal flags inferred from the time_conv keys
    assert vae.encoder.downsamples[0].avg_shortcut.factor_t == 1
    assert vae.encoder.downsamples[1].avg_shortcut.factor_t == 2
    assert vae.encoder.downsamples[2].avg_shortcut.factor_t == 2
    assert vae.decoder.upsamples[0].avg_shortcut.factor_t == 2
    assert vae.decoder.upsamples[1].avg_shortcut.factor_t == 2
    assert vae.decoder.upsamples[2].avg_shortcut.factor_t == 1
    assert vae.decoder.upsamples[3].avg_shortcut is None
    # every 2D weight survived the collapse bit-exactly (last slice, not garbage)
    loaded, source_sd = vae.state_dict(), source.state_dict()
    assert set(loaded) == set(source_sd)
    for key in loaded:
        assert torch.equal(loaded[key], source_sd[key]), key


def test_load_wan22_vae_without_temporal_flags(tmp_path):
    """A non-temporal (all-2D-resample) checkpoint loads with factor_t=1 everywhere."""
    source = _tiny_vae()
    video_sd = _as_video_state_dict(source.state_dict(), [False, False, False], [False, False, False], 8, 8)
    path = write_safetensors(tmp_path / "wan22_2d.safetensors", video_sd)

    vae = load_wan22_vae(path, device="cpu", latents_mean=TINY_MEAN, latents_std=TINY_STD)
    for i in range(3):
        assert vae.encoder.downsamples[i].avg_shortcut.factor_t == 1
        assert vae.decoder.upsamples[i].avg_shortcut.factor_t == 1


def test_load_wan22_vae_rejects_a_wrong_file(tmp_path):
    path = write_safetensors(tmp_path / "wrong.safetensors", {"unet.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="not a Wan 2\\.2 VAE"):
        load_wan22_vae(path, device="cpu")


def test_load_wan22_vae_rejects_default_stats_for_other_z_dim(tmp_path):
    """Shipped 48ch defaults vs a z_dim-4 checkpoint -> clear error, not an assert."""
    source = _tiny_vae()  # z_dim=4
    video_sd = _as_video_state_dict(source.state_dict(), [False, True, True], [True, True, False], 8, 8)
    path = write_safetensors(tmp_path / "wan22_z4.safetensors", video_sd)

    with pytest.raises(ValueError, match="z_dim"):
        load_wan22_vae(path, device="cpu")


def test_load_wan22_vae_honours_dtype(tmp_path):
    source = _tiny_vae()
    video_sd = _as_video_state_dict(source.state_dict(), [False, True, True], [True, True, False], 8, 8)
    path = write_safetensors(tmp_path / "wan22.safetensors", video_sd)

    vae = load_wan22_vae(path, device="cpu", dtype=torch.bfloat16, latents_mean=TINY_MEAN, latents_std=TINY_STD)
    assert vae.dtype == torch.bfloat16
