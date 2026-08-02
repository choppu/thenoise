"""Runtime tests using a fake model class (no torch adapters, no real weights)."""
from __future__ import annotations

import types

from diffuse.config import Settings
from diffuse.runtime import ModelPaths, NotLoadedError, Runtime


def _runtime_with_fake_model(monkeypatch):
    import diffuse.runtime as rt

    constructed = []

    class FakeModel:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    fake_module = types.SimpleNamespace(FakeModel=FakeModel)
    monkeypatch.setattr(rt, "_MODEL_CLASSES", {"fake": ("FakeModel", ".fakemod")})
    monkeypatch.setattr(rt, "importlib", types.SimpleNamespace(import_module=lambda *a, **k: fake_module))
    return Runtime(Settings()), constructed


def test_empty_runtime(monkeypatch):
    runtime, _ = _runtime_with_fake_model(monkeypatch)
    assert runtime.available() == []
    try:
        runtime.model
        assert False, "expected NotLoadedError"
    except NotLoadedError:
        pass


def test_unknown_model_raises(monkeypatch):
    runtime, _ = _runtime_with_fake_model(monkeypatch)
    try:
        runtime.load("bogus", ModelPaths("d", "v", "t"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_sets_single_model(monkeypatch):
    runtime, constructed = _runtime_with_fake_model(monkeypatch)
    runtime.load("fake", ModelPaths("dit", "vae", "te"))
    assert runtime.available() == ["fake"]
    assert runtime.model_name == "fake"
    assert constructed[0]["dit_path"] == "dit"

    # Loading a second model swaps (unloads) the first -- still one resident.
    runtime.load("fake", ModelPaths("dit2", "vae2", "te2"))
    assert runtime.available() == ["fake"]
    assert len(constructed) == 2
    assert constructed[1]["dit_path"] == "dit2"
