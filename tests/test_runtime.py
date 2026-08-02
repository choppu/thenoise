"""Runtime tests using a fake model class (no torch adapters, no real weights)."""
from __future__ import annotations

import types

from diffuse.config import Settings
from diffuse.runtime import ModelPaths, NotLoadedError, Runtime


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


def test_unknown_model_raises(monkeypatch):
    runtime, _ = _runtime_with_fake_model(monkeypatch)
    try:
        runtime.load("bogus", ModelPaths("d", "v", "t"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_resolves_when_model_omitted(monkeypatch):
    runtime, constructed = _runtime_with_fake_model(monkeypatch)
    runtime.load(None, ModelPaths("dit", "vae", "te"))
    assert runtime.available() == ["fake"]
    assert runtime.model_name == "fake"
    assert constructed[0]["dit_path"] == "dit"


def test_load_swaps_single_model(monkeypatch):
    runtime, constructed = _runtime_with_fake_model(monkeypatch)
    runtime.load(None, ModelPaths("dit", "vae", "te"))
    # Loading a second model swaps (unloads) the first -- still one resident.
    runtime.load(None, ModelPaths("dit2", "vae2", "te2"))
    assert runtime.available() == ["fake"]
    assert len(constructed) == 2
    assert constructed[1]["dit_path"] == "dit2"


def test_explicit_model_mismatch_raises(monkeypatch):
    import pytest
    import diffuse.models as dm

    class ClassA:
        name = "a"
        def __init__(self, **kw): pass

    class ClassB:
        name = "b"
        def __init__(self, **kw): pass

    monkeypatch.setattr(dm, "MODEL_CATALOG", [ClassA, ClassB])
    monkeypatch.setattr(dm, "resolve", lambda path: ClassB)  # DiT is actually 'b'

    runtime = Runtime(Settings())
    # User asked for 'a' but the DiT resolves to 'b' -> clear error before loading.
    with pytest.raises(ValueError):
        runtime.load("a", ModelPaths("d", "v", "t"))
