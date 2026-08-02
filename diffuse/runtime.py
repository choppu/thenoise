"""Single-model runtime: loads exactly one DiffusionModel at a time.

Replaces the multi-model registry. The runtime is deliberately thin: it resolves the
model class (either from the CLI's ``--model`` or by detecting the DiT type), holds
the single resident instance, and swaps (unloads + GCs) on reload so only one set of
weights is ever resident.

The text encoder and VAE are assumed to match the detected model; a wrong type throws
during load and we fail anyway.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelPaths:
    """Checkpoint paths supplied by the CLI (never read from config)."""
    dit_path: str
    vae_path: str
    text_encoder_path: str
    lora_weights: list = field(default_factory=list)
    lora_multipliers: list = field(default_factory=list)


class NotLoadedError(RuntimeError):
    """Raised when no model is loaded."""


class Runtime:
    def __init__(self, settings):
        self._settings = settings
        self._model: Any = None
        self._model_name: Optional[str] = None

    def load(self, model: Optional[str], paths: ModelPaths) -> None:
        from .models import MODEL_CATALOG, resolve

        if model:
            name = model.strip().lower()
            cls = next((c for c in MODEL_CATALOG if c.name == name), None)
            if cls is None:
                raise ValueError(
                    f"unknown model '{model}' (choose from "
                    f"{sorted(c.name for c in MODEL_CATALOG)})"
                )
            # Validate the explicit choice against the actual DiT type.
            detected = resolve(paths.dit_path)
            if detected.name != name:
                raise ValueError(
                    f"--model {name} does not match detected DiT type '{detected.name}'"
                )
        else:
            cls = resolve(paths.dit_path)
            name = cls.name

        kwargs = dict(
            dit_path=paths.dit_path,
            vae_path=paths.vae_path,
            text_encoder_path=paths.text_encoder_path,
            device=self._settings.device,
            dtype=self._settings.torch_dtype,
        )
        if name == "krea2":
            kwargs["lora_weights"] = paths.lora_weights or None
            kwargs["lora_multipliers"] = paths.lora_multipliers or None

        self._unload()  # swap: only one model resident at a time
        logger.info("Loading model '%s'", name)
        self._model = cls(**kwargs)
        self._model_name = name

    def _unload(self) -> None:
        if self._model is None:
            return
        logger.info("Unloading model '%s'", self._model_name)
        del self._model
        self._model = None
        self._model_name = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # torch may be unavailable (config-only tests)
            pass

    @property
    def model(self) -> Any:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model

    @property
    def model_name(self) -> str:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model_name

    def available(self) -> list[str]:
        return [self._model_name] if self._model else []
