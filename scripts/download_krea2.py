"""Download the Krea 2 model artifacts into a local directory.

Uses huggingface_hub. Files (per the musubi-tuner krea2 doc):

  DiT (Turbo)     krea/Krea-2-Turbo/turbo.safetensors
  DiT (RAW, opt.) krea/Krea-2-Raw/raw.safetensors
  VAE             Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors
  Text encoder    Comfy-Org/Qwen3-VL/text_encoders/qwen3vl_4b_bf16.safetensors

The Qwen3-VL tokenizer is fetched automatically (by repo id) at first text-encoder load.

Usage:
    python scripts/download_krea2.py --out ./models/krea2
    python scripts/download_krea2.py --out ./models/krea2 --include-raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

ARTIFACTS = {
    "dit_turbo": ("krea/Krea-2-Turbo", "turbo.safetensors"),
    "vae": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
    "text_encoder": ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_bf16.safetensors"),
    "dit_raw": ("krea/Krea-2-Raw", "raw.safetensors"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Krea 2 model artifacts")
    ap.add_argument("--out", default="./models/krea2", help="output directory")
    ap.add_argument("--include-raw", action="store_true", help="also download the RAW DiT")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = list(ARTIFACTS.items())
    if not args.include_raw:
        items = [i for i in items if i[0] != "dit_raw"]

    for name, (repo, path) in items:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"{name:14s} -> {dest}")


if __name__ == "__main__":
    main()
