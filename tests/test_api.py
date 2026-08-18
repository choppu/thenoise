"""API tests using a fake runtime (no torch, no weights, no TestClient)."""
from __future__ import annotations

import base64
import io

from thenoise.api import create_app, Text2ImageRequest, UpscaleRequest
from thenoise.runtime import Settings, Runtime


def _fake_runtime(tmp_path=None):
    """Runtime with a fake pipeline + fake model (and optionally upscalers dir)."""
    class FakePipeline:
        def generate(self, request):
            self.last_request = request
            from PIL import Image
            return Image.new("RGB", (8, 8))

        def list_loras(self):
            return ["style", "pose"]

    runtime = Runtime(Settings())
    runtime._pipeline = FakePipeline()
    runtime._model = object()
    runtime._model_name = "fake"

    if tmp_path is not None:
        (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x2.safetensors").write_text("x")
        runtime._pixel_upscalers.upscaler_dir = str(tmp_path)
    return runtime


def _empty_runtime():
    return Runtime(Settings())


def _endpoint(app, path):
    for r in app.routes:
        if getattr(r, "path", None) == path:
            return r.endpoint
    raise AssertionError(f"no route {path}")


def test_upscalers_lists_names(tmp_path):
    app = create_app(_fake_runtime(tmp_path))
    res = _endpoint(app, "/upscalers")()
    assert res["upscalers"] == ["RealESRGAN_x4", "sub/x2"]


def test_upscalers_available_without_model():
    # Pixel upscalers are a pixel-space/server concern and need no diffusion model.
    app = create_app(_empty_runtime())
    res = _endpoint(app, "/upscalers")()
    assert res["upscalers"] == []


def test_text2image_passes_pixel_upscaler(tmp_path):
    runtime = _fake_runtime(tmp_path)
    app = create_app(runtime)
    req = Text2ImageRequest(prompt="a fox", pixel_upscaler="RealESRGAN_x4")
    res = _endpoint(app, "/text2image")(req)
    assert res.status_code == 200
    assert runtime._pipeline.last_request.pixel_upscaler == "RealESRGAN_x4"


def test_request_field_defaults_none():
    req = Text2ImageRequest(prompt="x")
    assert req.pixel_upscaler is None


def _png_b64():
    """A tiny valid base64-encoded PNG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_upscale_returns_400_without_upscaler_dir():
    """No upscaler_dir configured -> controller raises ValueError -> 400."""
    app = create_app(_empty_runtime())
    req = UpscaleRequest(image_b64=_png_b64(), pixel_upscaler="x4")
    res = _endpoint(app, "/upscale")(req)
    assert res.status_code == 400


def test_upscale_returns_400_for_unknown_upscaler(tmp_path):
    """Unknown upscaler name -> validate raises -> 400."""
    app = create_app(_fake_runtime(tmp_path))
    req = UpscaleRequest(image_b64=_png_b64(), pixel_upscaler="nonexistent")
    res = _endpoint(app, "/upscale")(req)
    assert res.status_code == 400


def test_upscale_returns_b64_json(monkeypatch):
    """A successful upscale returns a b64_json payload (no torch needed)."""
    from PIL import Image

    runtime = _fake_runtime()

    def fake_upscale(image, upscale_factor, pixel_upscaler):
        return Image.new("RGB", (4, 4))

    monkeypatch.setattr(runtime.upscaler, "upscale", fake_upscale)
    app = create_app(runtime)
    req = UpscaleRequest(image_b64=_png_b64(), pixel_upscaler="x4", out="json")
    res = _endpoint(app, "/upscale")(req)
    assert isinstance(res, dict)
    assert "b64_json" in res


def test_upscale_defaults_to_png(monkeypatch):
    """Default out='png' returns a raw image/png response."""
    from PIL import Image

    runtime = _fake_runtime()
    monkeypatch.setattr(
        runtime.upscaler, "upscale", lambda image, f, name: Image.new("RGB", (4, 4))
    )
    app = create_app(runtime)
    req = UpscaleRequest(image_b64=_png_b64(), pixel_upscaler="x4")
    res = _endpoint(app, "/upscale")(req)
    assert res.status_code == 200
    assert res.media_type == "image/png"
    assert res.body
