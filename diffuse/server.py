"""Entrypoint: parse CLI, load a single model, then serve via uvicorn."""
from __future__ import annotations

from .cli import build_parser


def _apply_overrides(settings, args) -> None:
    for field in ("host", "port", "device", "dtype"):
        v = getattr(args, field, None)
        if v is not None:
            setattr(settings, field, v)


def _serve(args) -> None:
    from .config import load_settings
    settings = load_settings(args.config)
    _apply_overrides(settings, args)

    from .runtime import ModelPaths, Runtime
    runtime = Runtime(settings)
    runtime.load(
        args.model,
        ModelPaths(
            dit_path=args.dit,
            vae_path=args.vae,
            text_encoder_path=args.text_encoder,
            lora_weights=args.lora,
            lora_multipliers=[float(m) for m in (args.lora_multiplier or [])],
        ),
    )

    from .api import create_app
    import uvicorn

    app = create_app(runtime)
    print(f"diffuse-rocm serving model '{runtime.model_name}' on {settings.device}")
    uvicorn.run(app, host=settings.host, port=settings.port)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        _serve(args)
    elif args.command == "generate":
        from .generate import run_generate  # Phase 4
        run_generate(args)
    else:  # pragma: no cover - argparse requires a subcommand
        raise SystemExit("choose a subcommand: serve | generate")


if __name__ == "__main__":
    main()
