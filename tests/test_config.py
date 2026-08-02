"""Config tests (no torch / ROCm required)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffuse.config import load_settings


def test_defaults():
    s = load_settings(None)
    assert s.device == "cuda"
    assert s.dtype == "bfloat16"
    assert not s.krea2.enabled


def test_load_from_json(tmp_path):
    cfg = {
        "device": "cuda",
        "dtype": "bfloat16",
        "host": "0.0.0.0",
        "port": 9000,
        "krea2": {
            "dit_path": "dit.safetensors",
            "vae_path": "vae.safetensors",
            "text_encoder_path": "te.safetensors",
        },
        "anima": {
            "dit_path": "anima.safetensors",
            "vae_path": "avae.safetensors",
            "text_encoder_path": "ate.safetensors",
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    s = load_settings(str(p))
    assert s.host == "0.0.0.0"
    assert s.port == 9000
    assert s.krea2.enabled
    assert s.krea2.dit_path == "dit.safetensors"
    assert s.anima.enabled
    assert s.anima.dit_path == "anima.safetensors"


def test_env_override(tmp_path, monkeypatch):
    cfg = {"krea2": {"dit_path": "a", "vae_path": "b", "text_encoder_path": "c"}}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv("KREA2_DIT", "env_dit")
    monkeypatch.setenv("ANIMA_DIT", "env_anima")
    s = load_settings(str(p))
    assert s.krea2.dit_path == "env_dit"
    assert s.anima.dit_path == "env_anima"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(str(tmp_path / "nope.json"))
