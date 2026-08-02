"""Runtime tests using a fake model class (no torch adapters, no real weights)."""
from __future__ import annotations

import types

from diffuse.runtime import Settings, ModelPaths, NotLoadedError, Runtime


def _runtime_with_fake_model(monkeypatch):
    import diffuse.models as dm

    constructed = []

    class FakeModel:
        name = "fake"

        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(dm, "MODEL_CATALOG", [FakeModel])
    monkeypatch.setattr(dm, "resolve", lambda path: FakeModel)
    return Runtime(Settings()), constructed


def test_empty_runtime(monkeypatch):
    runtime, _ = _runtime_with_fake_model(monkeypatch)
    assert runtime.available() == []
    try:
        runtime.model
        assert False, "expected NotLoadedError"
    except NotLoadedError:
        pass


def test_load_resolves_model_from_dit(monkeypatch):
    runtime, constructed = _runtime_with_fake_model(monkeypatch)
    runtime.load(ModelPaths("dit", "vae", "te"))
    assert runtime.available() == ["fake"]
    assert runtime.model_name == "fake"
    assert constructed[0]["dit_path"] == "dit"


def test_load_swaps_single_model(monkeypatch):
    runtime, constructed = _runtime_with_fake_model(monkeypatch)
    runtime.load(ModelPaths("dit", "vae", "te"))
    # Loading a second model swaps (unloads) the first -- still one resident.
    runtime.load(ModelPaths("dit2", "vae2", "te2"))
    assert runtime.available() == ["fake"]
    assert len(constructed) == 2
    assert constructed[1]["dit_path"] == "dit2"
