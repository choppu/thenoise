"""Download the Krea 2 model artifacts into a local directory.

Uses huggingface_hub. All files come from `Comfy-Org/Krea-2`, which needs no
authentication. We only fetch the bf16 variants:

  DiT (Turbo)  diffusion_models/krea2_turbo_bf16.safetensors
  DiT (RAW)    diffusion_models/krea2_raw_bf16.safetensors   (--include-raw)
  VAE          vae/qwen_image_vae.safetensors
  Text encoder text_encoders/qwen3vl_4b_bf16.safetensors

The Qwen3-VL tokenizer is fetched automatically (by repo id) at first
text-encoder load.

Usage:
    python scripts/download_krea2.py --out ./models/krea2
    python scripts/download_krea2.py --out ./models/krea2 --include-raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Comfy-Org/Krea-2"

ARTIFACTS = {
    "dit_turbo": "diffusion_models/krea2_turbo_bf16.safetensors",
    "vae": "vae/qwen_image_vae.safetensors",
    "text_encoder": "text_encoders/qwen3vl_4b_bf16.safetensors",
    "dit_raw": "diffusion_models/krea2_raw_bf16.safetensors",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Krea 2 model artifacts (bf16)")
    ap.add_argument("--out", default="./models/krea2", help="output directory")
    ap.add_argument("--include-raw", action="store_true", help="also download the RAW DiT")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = list(ARTIFACTS.items())
    if not args.include_raw:
        items = [i for i in items if i[0] != "dit_raw"]

    for name, path in items:
        dest = hf_hub_download(REPO, path, local_dir=str(out))
        print(f"{name:14s} -> {dest}")


if __name__ == "__main__":
    main()
