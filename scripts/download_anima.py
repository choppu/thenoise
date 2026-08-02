"""Download the Anima (Cosmos-Predict2 2B text2image) model artifacts.

Source: https://huggingface.co/circlestone-labs/Anima

  DiT      split_files/diffusion_models/anima-<variant>.safetensors
           (variants: base-v1.0, aesthetic-v1.1, preview3-base, turbo-v1.0, ...)
  Text enc split_files/text_encoders/qwen_3_06b_base.safetensors
  VAE      split_files/vae/qwen_image_vae.safetensors

The Qwen3 / T5 tokenizer configs are packaged under ``diffuse/dit/anima/configs/``, so
they are not downloaded.

Usage:
    python scripts/download_anima.py --out ./models/anima
    python scripts/download_anima.py --out ./models/anima --variant aesthetic-v1.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "circlestone-labs/Anima"
DEFAULT_VARIANT = "aesthetic-v1.1"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Anima model artifacts")
    ap.add_argument("--out", default="./models/anima", help="output directory")
    ap.add_argument(
        "--variant", default=DEFAULT_VARIANT,
        help=f"DiT variant to download (default: {DEFAULT_VARIANT})",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = [
        ("dit", f"split_files/diffusion_models/anima-{args.variant}.safetensors"),
        ("text_encoder", "split_files/text_encoders/qwen_3_06b_base.safetensors"),
        ("vae", "split_files/vae/qwen_image_vae.safetensors"),
    ]
    for name, path in artifacts:
        dest = hf_hub_download(REPO, path, local_dir=str(out))
        print(f"{name:14s} -> {dest}")


if __name__ == "__main__":
    main()
