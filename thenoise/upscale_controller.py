"""Standalone pixel upscaling controller.

Model-free analog of ``PipelineController`` for the ``/upscale`` endpoint and
the ``upscale`` CLI subcommand. Owns the inference lock (serializes on-device
upscaler loads), so callers -- like ``text2image`` -- hold no lock themselves.
The manager stays unchanged; this controller is the caller that serializes.
"""
from __future__ import annotations

from PIL import Image

from thenoise.locks import inference_lock
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.image_tensor import pil_to_pixels, pixels_to_pil, resize_to_target


class PixelUpscaleController:
    """Owns standalone pixel upscaling: lock + apply + resize + PIL conversion."""

    def __init__(self, pixel_upscalers: PixelUpscalerManager):
        self._pixel_upscalers = pixel_upscalers
        self._lock = inference_lock

    def upscale(
        self, image: Image.Image, upscale_factor: float, pixel_upscaler: str
    ) -> Image.Image:
        """Upscale a PIL image by ``upscale_factor``x with the named upscaler.

        Validates the upscaler name and factor, applies the model at its detected
        native scale under the lock, then resizes to the requested factor (rounded
        to integer output dimensions).
        """
        name = self._pixel_upscalers.validate(pixel_upscaler)
        scale = self._pixel_upscalers.scale(name)
        if scale == 0:
            raise ValueError(
                "no pixel upscaler configured; pass --upscaler-dir "
                "(or run scripts/download_esrgan.py)"
            )
        if not 1 <= upscale_factor <= scale:
            raise ValueError(
                f"upscale_factor must be in [1, {scale}] for a {scale}x upscaler"
            )

        pixels = pil_to_pixels(image).to(self._pixel_upscalers.device)
        with self._lock:
            pixels = self._pixel_upscalers.apply(name, pixels, scale)
            pixels = resize_to_target(
                pixels,
                round(image.width * upscale_factor),
                round(image.height * upscale_factor),
            )
        return pixels_to_pil(pixels)


__all__ = ["PixelUpscaleController"]
