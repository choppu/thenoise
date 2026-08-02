"""Entrypoint: build settings + model registry, then serve via uvicorn."""
from __future__ import annotations

import argparse

import uvicorn

from .api import create_app
from .config import load_settings
from .models import ModelRegistry


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="diffuse")
    parser.add_argument(
        "-c", "--config", default=None,
        help="path to config.json (see config.example.json)",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    registry = ModelRegistry(settings)
    registry.load_all()

    app = create_app(registry)
    print(f"diffuse-rocm serving models {registry.available()} on {settings.device}")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
