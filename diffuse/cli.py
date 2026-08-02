"""Command-line interface for diffuse-rocm.

Two subcommands share the same checkpoint flags:
  * ``serve``    run the FastAPI HTTP server with a single loaded model
  * ``generate`` (Phase 4) run one generation and save the PNG

Model checkpoints are supplied here (``--dit`` / ``--vae`` / ``--text-encoder``),
never from config.
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
    serve.add_argument("-c", "--config", default=None,
                       help="path to config.json (server knobs only)")
    serve.add_argument("--model", required=True,
                       choices=["krea2", "anima"],
                       help="which model to load (Phase 3 makes this optional via detection)")
    _add_model_paths(serve)
    serve.add_argument("--host", help="override config host")
    serve.add_argument("--port", type=int, help="override config port")
    serve.add_argument("--device", help="override config device (e.g. cuda)")
    serve.add_argument("--dtype", help="override config dtype (bf16 only)")

    # generate (implemented in Phase 4)
    gen = sub.add_parser("generate", help="run one generation and save a PNG")
    gen.add_argument("--model", required=True,
                     choices=["krea2", "anima"],
                     help="which model to load")
    _add_model_paths(gen)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--negative-prompt", default="")
    gen.add_argument("--width", type=int)
    gen.add_argument("--height", type=int)
    gen.add_argument("--steps", type=int)
    gen.add_argument("--guidance-scale", type=float)
    gen.add_argument("--seed", type=int)
    gen.add_argument("--out", default="out.png")
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--dtype", default="bfloat16")

    return parser
