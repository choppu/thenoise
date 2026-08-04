"""Focused HTTP API. A single generic /text2image endpoint serves whichever model
the runtime currently holds (the runtime loads exactly one model at a time).

Synchronous request/response: each generate() call blocks until the image is ready.
A per-model inference lock serializes concurrent requests.

The request carries only the shared, model-agnostic parameters. Per-model defaults
(including the "advanced" sampler params) are owned by the model class and are NOT
exposed here.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from PIL import Image

from .runtime import NotLoadedError

logger = logging.getLogger(__name__)


class Text2ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    upscale: bool = False
    sampler: Optional[str] = None
    qwen_vae_enhance: bool = False
    film_grain: float = 0.0
    sharpening: float = 0.0
    lora_specs: Optional[List[str]] = None  # ["filename.safetensors:0.8", ...]


def _to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(runtime) -> FastAPI:
    app = FastAPI(title="diffuse-rocm", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "models": runtime.available()}

    @app.post("/text2image")
    def text2image(req: Text2ImageRequest):
        try:
            model = runtime.model
        except NotLoadedError:
            raise HTTPException(503, "no model is loaded")
        try:
            image = model.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                upscale=req.upscale,
                sampler=req.sampler,
                qwen_vae_enhance=req.qwen_vae_enhance,
                film_grain=req.film_grain,
                sharpening=req.sharpening,
                lora_specs=req.lora_specs,
            )
        except Exception as e:  # surface generation errors cleanly
            logger.exception("generation failed")
            raise HTTPException(500, f"generation failed: {e}") from e
        return {
            "model": runtime.model_name,
            "seed": req.seed,
            "image": _to_base64_png(image),
        }

    return app
