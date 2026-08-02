"""Config + CLI parsing tests (no torch / ROCm required, no real weights)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffuse.config import load_settings
from diffuse.cli import build_parser


def test_defaults():
    s = load_settings(None)
    assert s.device == "cuda"
    assert s.dtype == "bfloat16"
    assert s.host == "127.0.0.1"
    assert s.port == 8000


def test_load_from_json(tmp_path):
    cfg = {
        "device": "cuda",
        "dtype": "bfloat16",
        "host": "0.0.0.0",
        "port": 9000,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    s = load_settings(str(p))
    assert s.host == "0.0.0.0"
    assert s.port == 9000
    assert s.device == "cuda"


def test_env_override(tmp_path, monkeypatch):
    cfg = {"host": "127.0.0.1", "port": 8000}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv("DEVICE", "hip")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9999")
    s = load_settings(str(p))
    assert s.device == "hip"
    assert s.host == "0.0.0.0"
    assert s.port == 9999


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(str(tmp_path / "nope.json"))


def test_cli_serve_parses_model_paths():
    args = build_parser().parse_args([
        "serve", "--model", "krea2",
        "--dit", "dit.safetensors",
        "--vae", "vae.safetensors",
        "--text-encoder", "te.safetensors",
        "--lora", "lora1.safetensors", "--lora-multiplier", "0.5",
        "--port", "9000",
    ])
    assert args.command == "serve"
    assert args.model == "krea2"
    assert args.dit == "dit.safetensors"
    assert args.lora == ["lora1.safetensors"]
    assert args.lora_multiplier == ["0.5"]
    assert args.port == 9000


def test_cli_serve_requires_paths():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--model", "anima"])


def test_cli_generate_parses():
    args = build_parser().parse_args([
        "generate", "--model", "anima",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "a fox", "--steps", "30", "--seed", "7", "--out", "x.png",
    ])
    assert args.command == "generate"
    assert args.prompt == "a fox"
    assert args.steps == 30
    assert args.seed == 7
    assert args.out == "x.png"
