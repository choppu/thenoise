# Development Plan — diffuse-rocm

## Current state

The project is a focused ROCm diffusion engine that today:

- reads **two** models (Krea2 + Anima) from `config.json` and loads **both** at once
  (`diffuse/models.py: ModelRegistry.load_all()`)
- exposes only a FastAPI HTTP interface (`diffuse/api.py`, `diffuse/server.py`)
- depends on **two vendored libraries** for all model code:
  - `vendor/musubi_tuner/` → Krea2 (DiT, encoder, VAE)
  - `vendor/sd_scripts/` → Anima (DiT, encoder, VAE)
- has no model-type detection — the model is chosen implicitly by which config block
  is populated

The vendored trees are heavily **pruned** already and carry substantial **duplicated
code** (the Qwen-Image VAE `qwen_image_autoencoder_kl.py`, `attention.py`,
`fp8_optimization_utils.py`, `safetensors_utils.py`, `device_utils.py`,
`custom_offloading_utils.py` exist in **both** trees). The inference path only touches a
small subset of each tree.

## Goal architecture

```
diffuse/
  __init__.py         # no longer mutates sys.path (vendor/ removed)
  cli.py              # NEW: argparse for `serve` + `generate`
  server.py           # HTTP entrypoint (thin)
  api.py              # FastAPI app (unchanged surface)
  runtime.py          # single-model holder + model/VAE catalogs
  models/             # plugin registry of DiffusionModel classes
    base.py           #   abstract DiffusionModel (detect/load/generate)
    krea2.py          #   Krea2 adapter (own impl, was vendor/musubi_tuner)
    anima.py          #   Anima adapter (own impl, was vendor/sd_scripts)
  dit/                # own DiT code
    krea2_mmdit.py    #   SingleStreamDiT (from musubi_tuner)
    krea2_encoder.py  #   Qwen3-VL conditioner
    anima_models.py   #   Anima DiT + LLM adapter
  vae/                # VAE implementation (no separate detection)
    qwen_image.py     #   QwenImageVAE (single copy, was duplicated)
  sampling/           # shared flow-matching sampler + rotary helpers
  strategies/         # tokenize/text-encode strategies (from sd_scripts)
  utils/              # safetensors loading, device helpers, lora
scripts/
  download_krea2.py
  download_anima.py
vendor/               # REMOVED after migration
```

## Extensibility model (Req: add new models cleanly)

New models are added by registering a class that implements a small interface and
**owns its own detection routine**. The runtime never hard-codes a type; it asks each
registered class whether it recognizes the DiT checkpoint.

```python
# diffuse/models/base.py
class DiffusionModel(ABC):
    name: str

    @staticmethod
    def detect(f) -> bool:
        """Return True if this open safetensors handle is this model's DiT.

        ``f`` is a ``safetensors.safe_open(..., framework="pt")`` handle, opened
        ONCE by the runtime and passed to each class in turn — the file is never
        re-opened per class.
        """
        ...

    def generate(self, *, prompt, negative_prompt, width, height, steps,
                 guidance_scale, seed) -> Image.Image: ...
```

```python
# diffuse/models/__init__.py
MODEL_CATALOG = [Krea2Model, AnimaModel]
```

The `Runtime` walks the catalog, opening the **DiT file once** and passing the same
handle to every class's `detect()`; it loads the first class whose `detect()` returns
True. Adding `NewModel` = write the class, append it to `MODEL_CATALOG`, done.

> We detect **only the main model (DiT)**. Each model is paired with a specific text
> encoder and VAE, so once the DiT is identified we assume the `--text-encoder` and
> `--vae` checkpoints are of the correct type — if they aren't, loading throws and we
> fail anyway. There is no separate VAE/encoder detection.

> The **concrete body** of each `detect()` is a trivial implementation detail to be
> tackled later (the key-signature intuition below is good enough). The priority is
> getting the **scaffolding** right: the handle-passing protocol, the catalog, and the
> runtime resolution flow.

## Model-type auto-detection (Req 3)

Each model class owns its `detect()` routine, which takes a **loaded safetensors
handle** (opened once by the runtime); the runtime walks `MODEL_CATALOG` and loads the
first class that recognizes the DiT handle. Confirmed against the real Anima checkpoint
on disk, the intended signatures are:

| Class | `detect()` signature (DiT keys) |
|---|---|
| `Krea2Model` | has `x_embedder.` **and** `txtfusion.`, no `model.diffusion_model` |
| `AnimaModel` | any key starts `model.diffusion_model.` |

Notes:
- The DiT is the authoritative type signal. The text encoder is matched to the DiT
  (Krea2→Qwen3-VL, Anima→Qwen3-0.6B).
- **The concrete `detect()` bodies are a later detail** — the signatures above are
  the scaffolding target, to be validated against the Krea2 checkpoint when supplied.
- `--model` stays an **optional override**; when omitted the type is derived by
  calling each class's `detect()` on the opened `--dit` handle. When supplied it is
  validated against the detected class.

## VAE / text-encoder handling

The VAE and text encoder are **paired with the main model** and loaded directly by the
adapter. We do **not** detect them: once the DiT type is known, the `--text-encoder`
and `--vae` checkpoints are assumed correct, and a wrong type simply throws during
load. New VAE types are added by changing the adapter that loads them — no detection
scaffolding needed.

## Advanced model parameters (Req 4)

Per-model "advanced" parameters are **hard-coded defaults owned by the model class**
and are **not exposed** to the API or CLI:

| Model | Advanced params | Hard-coded default |
|---|---|---|
| Krea2 | `y1`, `y2`, `mu` | `y1=0.5`, `y2=1.15`, `mu=None` |
| Anima | `flow_shift` | `flow_shift=5.0` |

These live as class-level constants (e.g. `Krea2Model.DEFAULT_Y1=0.5`) and are read
inside `generate()`. They are removed from the Pydantic request models and from CLI
flags. The public surface stays `prompt / negative_prompt / width / height / steps /
guidance_scale / seed / num_images` only.

---

## Phased implementation

Ordering minimizes risk: the **runtime-surface** changes (1–4) are small and isolated
and keep the vendored imports intact; **de-vendoring** (5) is a pure relocation/refactor
done last with no behavior change.

### Phase 0 — Baseline
- Run `pytest` (config tests) and record green.
- Capture the exact set of symbols the two adapters import from the vendors (already
  known — ~10 call sites). This is the contract the own-implementation must satisfy.
- **Testing constraint:** the dev environment is a VM with **no GPU and limited RAM**.
  We do **not** attempt to load real models or run inference here — the maintainer runs
  those on the Strix Halo box. Unit tests here cover config, CLI parsing, detection
  scaffolding (synthetic key sets), and adapter wiring — never real weights.

> Whenever a step calls for loading a real checkpoint or running a generation, **stop
> and hand the exact commands to the user** to run on the GPU machine.

### Phase 1 — Single model at a time (Req 1)
- Replace `ModelRegistry` (multi-model map) with a `Runtime` that holds **one** model
  instance of a given type.
- `diffuse/models.py` → `diffuse/runtime.py`:
  - `Runtime(settings).load(model_type, paths)` — refuses a second model with a clear
    error, or optionally unloads/GCs the first (see `--swap` below).
  - `Runtime.get()` returns the single loaded model or raises `NotLoaded`.
- `api.py` no longer dispatches per-model; it serves one generic `/text2image` route
  whose parameter set depends on the loaded model type.
- Free the non-resident model: call `del` + `gc.collect()` + `torch.cuda.empty_cache()`
  when swapping, since 128GB is unified but we still want determinism and no accidental
  dual residency.

### Phase 2 — CLI checkpoint args (Req 2)
- `diffuse/cli.py` adds a shared `--dit`, `--vae`, `--text-encoder` (plus optional
  `--lora`/`--lora-multiplier`) flag group.
- `config.py` shrinks to **server knobs only** (`host`, `port`, `device`, `dtype`).
  Model paths and the `krea2:`/`anima:` blocks are removed.
- Loading a model requires explicit CLI paths; nothing is read from a config block.
- Environment-variable overrides for paths are removed (CLI is now the source of truth).

### Phase 3 — Auto-detect model type (Req 3)
- Add `diffuse/models/base.py` with the `detect(f)` protocol and the
  `MODEL_CATALOG` registry (no VAE/encoder detection — only the main model).
- `Runtime.resolve()` opens the DiT file once (`safetensors.safe_open`) and passes the
  handle to each class's `detect()` until one matches.
- CLI flow: if `--model` omitted, resolve from the `--dit` handle. Validate `--model`
  when given. Wire detection into `Runtime.load()` so HTTP and CLI share one code path.
- Implement `AnimaModel.detect()` fully (validated against the on-disk checkpoint);
  leave `Krea2Model.detect()` as the coarse stub until the Krea2 checkpoint is
  available.

### Phase 4 — CLI beside HTTP (Req 4)
- Add a `diffuse generate` subcommand:
  ```
  diffuse generate --dit ... --vae ... --text-encoder ... \
      --prompt "a fox in the snow" [--steps 8] [--width 1024] [--height 1024] \
      [--seed 42] [--out out.png] [--num-images 4]
  ```
- Reuses the exact same adapter `generate()` methods as the HTTP API → no logic drift.
- Saves PNGs to `--out` (or `out_<n>.png` for multiple images), prints the seed used.
- `diffuse serve` keeps the HTTP surface; the two share `cli.py` argument parsing and
  the `Runtime`.
- Retire `scripts/debug_anima.py` (its purpose is now `diffuse generate`).

### Phase 5 — Remove vendored libraries, own implementation (Req 5)

This is the largest phase and is split into sub-steps. The guiding principle: **move
only the code on the actual inference path into `diffuse/`, under a coherent namespace,
deduplicating what both vendors share, and drop everything dead.**

1. **Deduplicate shared modules first** — the same file appears in both vendors:
   | Shared concept | Current duplicate | New home |
   |---|---|---|
   | Qwen-Image VAE | both `qwen_image_autoencoder_kl.py` | `diffuse/vae/qwen_image_vae.py` |
   | attention backend | both `attention.py` | `diffuse/utils/attention.py` |
   | safetensors loader | both `safetensors_utils.py` | `diffuse/utils/safetensors.py` |
   | device helpers | both `device_utils.py` | `diffuse/utils/device.py` |
   | LoRA merge + networks | both `lora_utils.py` + `networks/` | `diffuse/utils/lora.py` + `diffuse/networks/` |
   | flow-math sampler/rotary | `sd_scripts/hunyuan_image_utils.py` + `krea2_sampling.py` | `diffuse/sampling/` |
   | tokenize/text-encode strategies | `sd_scripts/strategy_{base,anima}.py` | `diffuse/strategies/` |

2. **Move the Krea2 path** into `diffuse/dit/krea2*` + `diffuse/vae`:
   - `krea2_mmdit.py`, `krea2_encoder.py` (Qwen3-VL conditioner), the loaders from
     `krea2_utils.py`, and `encode_prompts`/`sample` from `krea2_sampling.py`.
   - **Keep** LoRA merge plumbing (`lora_utils`, `networks/`) — LoRAs are supported.
   - Drop fp8 paths and block-swap stubs (both unnecessary on 128GB unified RAM / bf16).

3. **Move the Anima path** into `diffuse/dit/anima_models.py` + `diffuse/strategies`:
   - `anima_models.py` (Anima DiT + LLMAdapter), the loaders from `anima_utils.py`,
     the tokenize/encode strategies, and the LoRA support (`networks/`).
   - Drop fp8, block-swap, and unused sampler helpers (`MomentumBuffer`,
     `AdaptiveProjectedGuidance`, `normalized_guidance_apg`, etc.).

4. **Rewrite the two adapters** (`diffuse/krea2.py`, `diffuse/anima.py`) to import from
   the new `diffuse.*` modules instead of `musubi_tuner.*` / `sd_scripts.*`.

5. **Delete `vendor/`**, remove the `sys.path` mutation in `diffuse/__init__.py`, and
   update `pyproject.toml` (`[tool.setuptools]` packages → list the new `diffuse`
   subpackages).

6. **Keep the vendored tokenizer configs** (`configs/qwen3_06b`, `configs/t5_old`) —
   the text encoder's safetensors loading needs them. Move them under `diffuse/`
   as package data (`diffuse/strategies/configs/...`) and declare them in
   `pyproject.toml` so they ship with the wheel.

### Phase 6 — Cleanup & docs
- Update `README.md`: new CLI usage, single-model semantics, auto-detection, no more
  vendoring.
- Update `config.example.json` → remove model blocks (or delete it in favor of flags).
- Update `tests/`:
  - config tests → server-knobs-only.
  - add detection scaffolding tests using synthetic key sets passed to each
    `detect()` (no real weights; verify the handle-protocol and catalog resolution).
  - add CLI parsing tests for `serve`/`generate`.
- Provide the user with the exact commands to re-validate a golden image for each model
  end-to-end after Phases 1, 4 and 5 (run on the GPU box, not in this VM).

---

## Risks & mitigations

- **Krea2 detect() body unverified** (no Krea2 checkpoint on disk). Mitigated: it is
  a coarse stub until the checkpoint is supplied; scaffolding is validated with
  synthetic key sets.
- **De-vendoring is the highest-risk change** — a subtle divergence in a copied file can
  silently change numerics. Mitigate: golden-image validation on the GPU box right
  after Phase 5, keep the move mechanical (verbatim copy + prune), land one reviewable
  commit per model.
- **Behavior drift between HTTP and CLI** — mitigated by sharing the exact same
  `generate()` methods; the CLI is a thin wrapper.
- **Tokenizer configs must ship with the wheel** — must be declared as package data or
  the Anima text encoder breaks in installed (non-source) use.
- **LoRAs are kept** (must keep `lora_utils` + `networks/` in the own-implementation);
  **fp8 and block-swap are dropped** — their code paths and monkey-patches are removed.

## Suggested commit sequence

1. Phase 1 — single-model runtime + generic HTTP route
2. Phase 2 — CLI checkpoint args (config shrinks to server knobs)
3. Phase 3 — auto-detect model type
4. Phase 4 — `diffuse generate` CLI (+ retire debug script)
5. Phase 5a — deduplicate shared modules into `diffuse/`
6. Phase 5b — migrate Krea2, Phase 5c — migrate Anima, Phase 5d — delete `vendor/`
7. Phase 6 — docs, config, tests

Each phase is independently mergeable and leaves the tree green.
