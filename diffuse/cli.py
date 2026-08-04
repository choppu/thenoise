"""Command-line interface for diffuse-rocm.

Two subcommands share the same checkpoint flags:
  * ``serve``    run the FastAPI HTTP server with a single loaded model
  * ``generate`` run one generation and save the PNG

Model checkpoints are supplied here (``--dit`` / ``--vae`` / ``--text-encoder``),
and the model type is detected automatically from the ``--dit`` checkpoint.
All options are passed on the command line (there is no config file).
"""
from __future__ import annotations

import argparse


def _add_model_paths(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dit", required=True, metavar="PATH",
                   help="DiT checkpoint (.safetensors)")
    p.add_argument("--vae", required=True, metavar="PATH",
                   help="VAE checkpoint (.safetensors)")
    p.add_argument("--text-encoder", required=True, metavar="PATH",
                   help="text encoder checkpoint (.safetensors)")
    p.add_argument("--lora", action="append", default=[], metavar="PATH",
                   help="LoRA safetensor to merge at load time (repeatable)")
    p.add_argument("--lora-multiplier", action="append", default=[], metavar="FLOAT",
                   help="LoRA multiplier (repeatable, pairs with --lora)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffuse")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    serve = sub.add_parser("serve", help="run the FastAPI HTTP server")
    _add_model_paths(serve)
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000,
                       help="bind port (default: 8000)")
    serve.add_argument("--device", default="cuda",
                       help="inference device; ROCm aliases cuda -> hip (default: cuda)")

    # generate
    gen = sub.add_parser("generate", help="run one generation and save a PNG")
    _add_model_paths(gen)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--negative-prompt", default="")
    gen.add_argument("--width", type=int)
    gen.add_argument("--height", type=int)
    gen.add_argument("--steps", type=int)
    gen.add_argument("--guidance-scale", type=float)
    gen.add_argument("--seed", type=int)
    gen.add_argument("--out", default="out.png")
    gen.add_argument("--upscale", action="store_true",
                     help="upscale the latent 2x in latent space (SesquiLSR) and "
                          "run a low-strength refine denoise before decoding")
    gen.add_argument("--sampler", choices=["euler", "er_sde"], default=None,
                     help="denoising solver (default: er_sde)")
    gen.add_argument("--qwen-vae-enhance", action="store_true",
                     help="apply the Nyquist Notch post filter to decoded pixels "
                          "(removes 2px grid artifacts)")
    gen.add_argument("--film-grain", type=float, default=0.0,
                     help="film-grain strength 0.0–10.0. Reasonable values are <= 1.0 (default: 0.0)")
    gen.add_argument("--sharpening", type=float, default=0.0,
                     help="RCAS sharpening strength 0.0–1.0 (contrast-adaptive, default: 0.0 / off)")
    gen.add_argument("--device", default="cuda",
                     help="inference device; ROCm aliases cuda -> hip (default: cuda)")

    return parser
