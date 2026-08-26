"""Generic reference-latent helpers for instruction-based editing.

Shared by Flux2 Klein and Qwen Image Edit (and any future reference-latent
model). Both use the Flux-family mechanism: the input image is encoded to a
latent, packed into DiT tokens + position ids, concatenated to the generated
image tokens, and sliced back off the output. The only per-model differences
are the number of position axes, the reference index offset, and whether the
spatial axes are centered — all knobs on ``build_reference_ids``.

  * ``build_reference_ids``    — position ids for a reference latent (index method).
  * ``concat_reference``       — prepend reference tokens+ids to image tokens+ids.
  * ``slice_reference_output`` — drop the trailing reference tokens from the DiT output.
"""
from __future__ import annotations

import torch

__all__ = ["build_reference_ids", "concat_reference", "slice_reference_output"]


def build_reference_ids(
    h: int,
    w: int,
    *,
    index: int,
    axes: int,
    center: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Position ids for a reference latent of spatial size ``(h, w)``.

    Mirrors ComfyUI's ``process_img`` for the "index" reference method.

    Returns ``[1, h*w, axes]`` integer ids on ``device``.
    """
    ids = torch.zeros((h, w, axes), device=device, dtype=torch.long)
    ids[:, :, 0] = index
    ids[:, :, 1] = torch.linspace(0, h - 1, h, device=device, dtype=torch.long).unsqueeze(1)
    ids[:, :, 2] = torch.linspace(0, w - 1, w, device=device, dtype=torch.long).unsqueeze(0)
    if center:
        ids[:, :, 1] -= h // 2
        ids[:, :, 2] -= w // 2
    return ids.reshape(1, -1, axes)


def concat_reference(
    img: torch.Tensor,
    img_ids: torch.Tensor,
    ref_tokens: torch.Tensor | None,
    ref_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend reference tokens+ids to the image tokens+ids.

    ``ref_tokens``/``ref_ids`` of ``None`` (plain generation) return
    ``img``/``img_ids`` unchanged.
    """
    if ref_tokens is None or ref_ids is None:
        return img, img_ids
    return torch.cat([img, ref_tokens], dim=1), torch.cat([img_ids, ref_ids], dim=1)


def slice_reference_output(out: torch.Tensor, num_img_tokens: int) -> torch.Tensor:
    """Drop the trailing reference tokens from a DiT output.

    Reference tokens are concatenated *after* the image tokens.
    """
    return out[:, :num_img_tokens]
