"""CLI parsing tests (no torch / ROCm required, no real weights)."""
from __future__ import annotations

import pytest

from diffuse.cli import build_parser


def test_cli_serve_parses_model_paths():
    args = build_parser().parse_args([
        "serve",
        "--dit", "dit.safetensors",
        "--vae", "vae.safetensors",
        "--text-encoder", "te.safetensors",
        "--lora", "lora1.safetensors", "--lora-multiplier", "0.5",
        "--host", "0.0.0.0", "--port", "9000", "--device", "hip",
    ])
    assert args.command == "serve"
    assert args.dit == "dit.safetensors"
    assert args.lora == ["lora1.safetensors"]
    assert args.lora_multiplier == ["0.5"]
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.device == "hip"


def test_cli_serve_defaults():
    args = build_parser().parse_args([
        "serve",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
    ])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.device == "cuda"


def test_cli_serve_requires_paths():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve"])


def test_cli_generate_parses():
    args = build_parser().parse_args([
        "generate",
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
    assert args.device == "cuda"


def test_cli_rejects_unknown_flags():
    # --model and --dtype are gone: everything is auto-detected / fixed bf16.
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "serve", "--model", "krea2",
            "--dit", "d.safetensors",
            "--vae", "v.safetensors",
            "--text-encoder", "te.safetensors",
        ])
