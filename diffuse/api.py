"""Focused HTTP API. Deliberately small surface: one endpoint per model + /health.

Synchronous request/response: each generate() call blocks until the image is ready.
A per-model inference lock serializes concurrent requests. (A queue / job model can be
added later if concurrency or long-running requests become a concern.)
"""
from __future__ import annotations

import base64
import io
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from PIL import Image


class Krea2Request(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 8
    guidance_scale: float = 0.0
    seed: Optional[int] = None
    y1: float = 0.5
    y2: float = 1.15
    mu: Optional[float] = None
    num_images: int = 1


class AnimaRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 50
    guidance_scale: float = 3.5
    flow_shift: float = 5.0
    seed: Optional[int] = None


def _to_base64_pngs(images: list[Image.Image]) -> list[str]:
    out: list[str] = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def create_app(registry) -> FastAPI:
    app = FastAPI(title="diffuse-rocm", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "models": registry.available()}

    @app.post("/krea2/text2image")
    def krea2_text2image(req: Krea2Request):
        try:
            model = registry.get("krea2")
        except KeyError:
            raise HTTPException(503, "krea2 model is not loaded")
        try:
            images = model.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                y1=req.y1,
                y2=req.y2,
                mu=req.mu,
                num_images=req.num_images,
            )
        except Exception as e:  # surface generation errors cleanly
            logger.exception("krea2 generation failed")
            raise HTTPException(500, f"generation failed: {e}") from e
        return {
            "model": "krea2",
            "seed": req.seed,
            "images": _to_base64_pngs(images),
        }

    @app.post("/anima/text2image")
    def anima_text2image(req: AnimaRequest):
        try:
            model = registry.get("anima")
        except KeyError:
            raise HTTPException(503, "anima model is not loaded")
        try:
            images = model.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                flow_shift=req.flow_shift,
                seed=req.seed,
            )
        except Exception as e:  # surface generation errors cleanly
            logger.exception("anima generation failed")
            raise HTTPException(500, f"generation failed: {e}") from e
        return {
            "model": "anima",
            "seed": req.seed,
            "images": _to_base64_pngs(images),
        }

    return app
