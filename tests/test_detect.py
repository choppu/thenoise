"""Model-type detection tests.

Detection reads safetensors header keys only (no tensors, no weights). We test the
``detect(f)`` logic against a fake handle with synthetic keys, plus the catalog
``resolve()`` against a tiny real safetensors file.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from diffuse.models import AnimaModel, Krea2Model, resolve


class _FakeHandle:
    """Mimics ``safetensors.safe_open``'s ``keys()`` for detection."""

    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys


_ANIMA_KEYS = [
    "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight",
    "model.diffusion_model.x_embedder.linear.weight",
]

_KREA2_KEYS = [
    "x_embedder.linear.weight",
    "txtfusion.layerwise_blocks.0.attn.q_proj.weight",
    "blocks.0.mod.lin.weight",
    "txtmlp.1.weight",
]


def test_anima_detect_true():
    assert AnimaModel.detect(_FakeHandle(_ANIMA_KEYS)) is True


def test_krea2_detect_true():
    assert Krea2Model.detect(_FakeHandle(_KREA2_KEYS)) is True


def test_krea2_rejects_anima_keys():
    assert Krea2Model.detect(_FakeHandle(_ANIMA_KEYS)) is False


def test_anima_rejects_krea2_keys():
    assert AnimaModel.detect(_FakeHandle(_KREA2_KEYS)) is False


def _write_safetensors(path: Path, keys: dict) -> None:
    import torch
    from safetensors.torch import save_file
    save_file({k: torch.zeros(1) for k in keys}, str(path))


def test_resolve_anima(tmp_path):
    p = tmp_path / "anima.safetensors"
    _write_safetensors(p, _ANIMA_KEYS)
    assert resolve(str(p)) is AnimaModel


def test_resolve_krea2(tmp_path):
    p = tmp_path / "krea2.safetensors"
    _write_safetensors(p, _KREA2_KEYS)
    assert resolve(str(p)) is Krea2Model


def test_resolve_unknown_raises(tmp_path):
    p = tmp_path / "unknown.safetensors"
    _write_safetensors(p, {"some.random.key": 0})
    with pytest.raises(ValueError):
        resolve(str(p))


@pytest.mark.skipif(
    not os.path.exists("models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors"),
    reason="real Anima checkpoint not present",
)
def test_resolve_real_anima_checkpoint():
    """Sanity-check the detector against the real Anima DiT on disk."""
    p = "models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors"
    assert resolve(p) is AnimaModel
