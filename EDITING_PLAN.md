# Instruction-based editing: Flux2 Klein + Qwen Image Edit

## Goal

Add instruction-based image editing (an input image + a text instruction → an edited
image) to `thenoise`, first for **Flux.2 Klein**, then for a future **Qwen Image Edit**
model. The two share a single generic editing core; only per-model kernels differ.

This is **not** img2img (no noise injection). Editing generates from noise and guides
it with the input image via the Flux-family **reference-latent** mechanism
(`reference_latents` conditioning, `process_img` → concat → slice, generate-from-noise).

## Guiding principles

- **Follow existing patterns**: each DiT lives in
  `thenoise/dit/<model>/{models,sampling,utils,__init__}.py`; each model adapter in
  `thenoise/models/<model>.py`; VAEs in `thenoise/vae/`; orchestration in
  `PipelineController`.
- **One generic editing core** (Stage 1) shared by both models — the only truly shared
  machinery.
- **Per-model**: only the DiT forward signature, VAE encode, and (for Qwen) the
  multimodal text encoder differ.

## Model comparison (why this is shared)

| Parameter | Flux2 Klein | Qwen Image Edit |
|---|---|---|
| position axes | 4 (t/h/w/l) | 3 (t/h/w) |
| ref index step | +10 (`ref_index_scale`) | +1 |
| spatial centering | no | yes (h/w centered) |
| latent channels | 128 packed @16x | 16 @8x |
| VAE encoder in `thenoise` | missing | present |
| text encoder | text-only Qwen3 | multimodal Qwen2-VL (sees image) |

The editing *mechanism* is identical (conditioning keys, `process_img` → concat →
slice, generate-from-noise); only the above differ. The text conditioning is the one
fundamentally non-shared piece.

---

## Stage 1 — Generic shared parts (both models)

### 1.1 New shared module `thenoise/dit/reference.py`
The generic version of ComfyUI's `process_img` id-building + the concat/slice pattern,
parameterized by the only things that differ:

```python
def build_reference_ids(h, w, *, axes, index, center=False) -> torch.Tensor
def concat_reference(img, img_ids, ref_tokens, ref_ids) -> tuple[tensor, tensor]
def slice_reference_output(out, num_img_tokens) -> tensor
```

- Flux2 uses `axes=4` (t/h/w/l), `index=10` (`ref_index_scale`), `center=False`.
- Qwen uses `axes=3` (t/h/w), `index=1`, `center=True` (h/w minus `h_len//2`).

This is the single reusable piece — both models call it, neither re-implements the id
scheme.

### 1.2 Base-class hooks in `thenoise/models/base.py`
Add **optional** (default no-op) hooks so non-editing models are untouched:

```python
supports_edit: bool = False
def encode_reference(self, image, params) -> torch.Tensor: ...  # VAE encode -> canonical latent
def pack_reference_latent(self, latents, index) -> tuple | None: ...  # canonical -> (tokens, ids)
```

`prepare_latent`/`denoise_step` get an optional ref path (default `None`) so editing
models stash/pass ref tokens+ids without disturbing the t2i loop. `Conditioning` stays
as-is (ref tokens are per-request, not cached with the prompt).

### 1.3 Request/config in `thenoise/models/config.py`
Extend `GenerateRequest` with `image` (base64/bytes) + `ref_latents_method` (default
`"index"`). No new tensor conversion is written: the **existing**
`pil_to_pixels` in `thenoise/utils/image_tensor.py` (PIL RGB → `[C,H,W]` fp32 in
`[-1,1]`) is reused as-is by both models. The only net-new wiring is decoding the
wire bytes → PIL (`base64.b64decode` + `Image.open(io.BytesIO(...))`), which lives at
the **API/CLI layer** (`thenoise/api.py`), not in a shared helper.

### 1.4 Shared pipeline path in `thenoise/pipeline.py`
Add one model-agnostic `edit()` that **reuses the existing**
`_denoise`/`decode`/`postprocess`/`resize` helpers:
`encode_reference(image)` → `encode_prompt(prompt, image=…)` →
`pack_reference_latent` → `_denoise` → `decode`. Because both models edit from noise
with ref conditioning, the loop is identical; only the model kernels differ.

### 1.5 Surface: `thenoise/api.py` + `thenoise/cli.py`
`/edit` endpoint (or extend `/text2image` with an optional `image` field) + a `--image`
CLI flag on `generate`.

---

## Stage 2 — Flux2 Klein image edit

### 2.1 Flux.2 VAE encoder — `thenoise/vae/flux2.py` *(the blocker)*
- Add `encoder` + `quant_conv` modules (exact mirror of the existing `_Decoder`).
- Add `encode_pixels_to_latents(pixels)` following the **existing signature/convention**
  in `AutoencoderKLQwenImage`: pixels → encoder → quant_conv → 32ch@8x → patchify to
  128ch@16x → BatchNorm normalize via the existing `bn` (inverse of `decode_to_pixels`).
- Update `load_flux2_vae` to also keep `encoder.*`/`quant_conv.*` keys.
- ⚠ Verify `encoder.*`/`quant_conv.*` exist in the downloaded `flux2-vae.safetensors`
  (ComfyUI loads it as a full AutoencoderKL, but confirm locally).

### 2.2 Flux2 DiT ref path — `thenoise/dit/flux2/models.py`
Add `ref_tokens`/`ref_ids` args to `Flux2.forward`; `concat_reference(img, pe_x, ref_tokens,
ref_ids)`, run, `slice_reference_output(out, num_img_tokens)`. Use the Stage 1 helpers.

### 2.3 Flux2 ref packing — `thenoise/dit/flux2/sampling.py`
`prc_img(latents, t_coord=torch.tensor([index]))` already supports the `t`-axis override —
use it for the index method (index=10). No new sampling code needed.

### 2.4 `FluxKleinModel` — `thenoise/models/flux_klein.py`
Set `supports_edit=True`; implement `encode_reference` (Flux2 VAE encode) +
`pack_reference_latent` (`prc_img` with index); stash ref tokens/ids in `prepare_latent`,
pass them in `denoise_step`. Schedule/Euler/`init_latents`/`finalize_latent` unchanged.

### 2.5 Tests — `tests/test_flux_klein.py` + new `tests/test_edit.py`
VAE encode/decode roundtrip, ref id building, DiT forward with ref tokens, adapter edit
path (no weights/GPU).

---

## Stage 3 — Qwen Image Edit (generate + edit)

### 3.1 New DiT package `thenoise/dit/qwen_image/`
Follow the existing `thenoise/dit/{zimage,anima}` structure: `models.py` (the
`QwenImageTransformer2DModel`, ported from ComfyUI `comfy/ldm/qwen_image/model.py`),
`sampling.py` (packing + schedule), `utils.py` (loaders), `__init__.py`. Use
`build_reference_ids` for its 3-axis/index/centered ref path.

### 3.2 Multimodal text encoder *(the net-new, model-specific piece)*
Port the Qwen2-VL text encoder that consumes the input image as vision tokens (the
`llama_template` + `tokenize(prompt, images=images)` flow from ComfyUI `nodes_qwen.py`).
This is **not** shared with Flux2 — it's the one genuinely new component. New
`scripts/download_qwen_image_edit.py` for the DiT + Qwen2-VL text encoder weights.

### 3.3 `thenoise/models/qwen_image_edit.py`
Follows the `flux_klein.py`/`anima.py` adapter pattern:
- `name = "qwen_image_edit"`, `detect(f)` on the DiT's distinctive keys, register in
  `MODEL_CATALOG`.
- Reuses the **existing full Qwen-Image VAE** (`load_qwen_vae`,
  `_upscale_format="wan21"`, 16ch@8x canonical latent) — its `encode_pixels_to_latents`
  already exists, so **no VAE work here**.
- `encode_prompt(prompt, negative, guidance, image=None)`: **generate** (no image) =
  text-only; **edit** (image) = feed vision tokens + attach ref latent. One adapter
  covers both modes.
- `encode_reference` (VAE encode) + `pack_reference_latent` (3-axis/index/centered).
- `prepare_latent`/`denoise_step` follow the Flux2 pattern (stash/pass ref tokens+ids).

### 3.4 Tests
Detection, both generate and edit conditioning, ref packing, DiT forward with ref,
VAE encode roundtrip.

---

## What's shared vs. net-new (repetition audit)

| Piece | Shared (Stage 1) | Flux2 (Stage 2) | Qwen Edit (Stage 3) |
|---|---|---|---|
| ref id-building + concat/slice | ✅ `dit/reference.py` | calls it | calls it |
| VAE `encode_pixels_to_latents` | interface convention | **add encoder** | already exists |
| pipeline edit orchestration | ✅ `pipeline.py` | reuse | reuse |
| request/API/CLI surface | ✅ | reuse | reuse |
| DiT forward ref path | — | add args | in new DiT |
| text conditioning | — | text-only Qwen3 (unchanged) | **multimodal Qwen2-VL (net-new)** |
| DiT architecture | — | exists | **new `dit/qwen_image/`** |
| model adapter | — | extend `flux_klein.py` | new `qwen_image_edit.py` |

The only genuinely model-specific additions are: the Flux2 VAE encoder (Stage 2), and
the Qwen Image Edit DiT + multimodal text encoder (Stage 3). Everything else is the one
shared core from Stage 1.
