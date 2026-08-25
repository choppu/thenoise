const $ = id => document.getElementById(id);
let currentUrl = null;
let currentMeta = null;
let history = []; // newest first, capped at 10: {url, meta, time}

// Show a range slider's value in a companion element (formatted to N decimals).
function bindRange(sliderId, valId, digits = 2) {
  $(sliderId).addEventListener('input', e =>
    $(valId).textContent = parseFloat(e.target.value).toFixed(digits));
}
bindRange('film_grain', 'film_grain_val');
bindRange('sharpening', 'sharpening_val');

// Set a slider's value and mirror it in its companion label.
function setRange(sliderId, valId, value, digits = 2) {
  $(sliderId).value = value;
  $(valId).textContent = value.toFixed(digits);
}
// Mirror a slider's current value into its companion label.
function rangeLabel(sliderId, valId, digits = 2) {
  $(valId).textContent = parseFloat($(sliderId).value).toFixed(digits);
}
// Cap a slider at `max`, clamping the current value if it exceeds it.
function clampSlider(slider, max) {
  slider.max = max;
  if (parseFloat(slider.value) > max) slider.value = max;
}

const LATENT_SCALE = 2; // latent refiner multiplier

function updateFactorVal() {
  rangeLabel('upscale_factor', 'upscale_factor_val');
}

// Render "W × H px" for dims scaled by `factor` into resEl; clear when dims unknown.
function renderRes(resEl, w, h, factor) {
  resEl.textContent = (w && h)
    ? ` ${Math.round(w * factor)} \u00d7 ${Math.round(h * factor)} px`
    : '';
}

// Show the final output resolution (base width/height x factor) under the slider.
function updateFinalRes() {
  const w = parseInt($('width').value, 10);
  const h = parseInt($('height').value, 10);
  const factor = parseFloat($('upscale_factor').value) || 1;
  renderRes($('final_res'), w, h, factor);
}

// Max slider value follows the selected pixel upscaler's native scale:
//   refined    -> 2x (latent refiner) * pixel scale
//   no-refiner -> pixel scale only
// With no pixel upscaler, refined caps at the latent 2x and no-refiner at 1x.
function updateUpscaleMax() {
  const slider = $('upscale_factor');
  const type = $('upscale_type').value;
  const name = $('pixel_upscaler').value;
  const scale = name ? (upscalerScales[name] || 2) : 0;
  const max = type === 'no-refiner'
    ? (scale || 1)
    : (scale ? LATENT_SCALE * scale : LATENT_SCALE);
  clampSlider(slider, max);
  updateFactorVal();
  updateFinalRes();
}

$('upscale_type').addEventListener('change', updateUpscaleMax);
$('pixel_upscaler').addEventListener('change', updateUpscaleMax);
$('upscale_factor').addEventListener('input', () => { updateFactorVal(); updateFinalRes(); });
$('width').addEventListener('input', updateFinalRes);
$('height').addEventListener('input', updateFinalRes);

function setTimer(elId, textId, state, text) {
  $(elId).className = 'timer ' + state;
  $(textId).textContent = text;
}

// Reads a PNG tEXt chunk (Latin-1 encoded, uncompressed).
function readPngText(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const texts = {};
  if (dv.getUint32(0) !== 0x89504e47) return texts; // not a PNG
  let off = 8; // skip signature
  while (off + 8 <= bytes.byteLength) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(
      dv.getUint8(off + 4), dv.getUint8(off + 5),
      dv.getUint8(off + 6), dv.getUint8(off + 7)
    );
    const start = off + 8;
    if (type === 'tEXt') {
      // keyword: null-terminated Latin-1; text: rest of chunk
      const chunk = bytes.subarray(start, start + len);
      let sep = -1;
      for (let i = 0; i < chunk.length; i++) if (chunk[i] === 0) { sep = i; break; }
      if (sep >= 0) {
        const keyword = decodeLatin1(chunk.subarray(0, sep));
        const value = decodeLatin1(chunk.subarray(sep + 1));
        texts[keyword] = value;
      }
    }
    off = start + len + 4; // skip data + crc
  }
  return texts;
}

function decodeLatin1(bytes) {
  // PNG tEXt chunks are Latin-1 encoded; decode each byte into a string.
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return s;
}

// Parse the JSON stored under a given tEXt keyword ("generation_data" / "upscale_data").
async function decodeMeta(blob, keyword) {
  const buf = await blob.arrayBuffer();
  const texts = readPngText(new Uint8Array(buf));
  let meta = null;
  try { meta = texts[keyword] ? JSON.parse(texts[keyword]) : null; }
  catch { meta = null; }
  return meta;
}

const FIELD_LABELS = {
  model: 'Model', prompt: 'Prompt', negative_prompt: 'Negative prompt',
  width: 'Width', height: 'Height', steps: 'Steps',
  guidance_scale: 'CFG scale', seed: 'Seed', upscale: 'Upscale',
  upscale_factor: 'Upscale factor', upscale_type: 'Upscale type',
  sampler: 'Sampler', qwen_vae_enhance: 'Reduce grid pattern',
  film_grain: 'Film grain', sharpening: 'Sharpening', lora_specs: 'LoRA',
};

// Append one dt/dd pair to a dl.info-grid.
function addRow(grid, label, value, cls) {
  const dt = document.createElement('dt'); dt.textContent = label;
  const dd = document.createElement('dd'); if (cls) dd.className = cls;
  if (Array.isArray(value)) dd.textContent = value.join(', ') || '—';
  else dd.textContent = value ?? '—';
  grid.append(dt, dd);
}

// Render image metadata into the shared info grid (used by the info popup).
function renderInfo(meta) {
  const grid = $('info_modal_grid');
  grid.innerHTML = '';
  if (!meta || typeof meta !== 'object') {
    const dd = document.createElement('dd');
    dd.className = 'none';
    dd.textContent = 'No metadata found in this image.';
    grid.appendChild(dd);
    return;
  }
  for (const key of ['model','prompt','negative_prompt','width','height','steps','sampler','guidance_scale','seed']) {
    if (key in meta) addRow(grid, FIELD_LABELS[key], meta[key], key === 'prompt' || key === 'negative_prompt' ? 'prompt' : '');
  }
  if ('upscale' in meta) addRow(grid, 'Upscale', meta.upscale);
  if ('upscale_factor' in meta) addRow(grid, 'Upscale factor', meta.upscale_factor);
  if ('upscale_type' in meta) addRow(grid, 'Upscale type', meta.upscale_type);
  if ('pixel_upscaler' in meta && meta.pixel_upscaler) addRow(grid, 'Pixel upscaler', meta.pixel_upscaler);
  if ('qwen_vae_enhance' in meta) addRow(grid, 'Reduce grid pattern', meta.qwen_vae_enhance);
  for (const key of ['film_grain','sharpening','lora_specs']) {
    if (key in meta && (key !== 'film_grain' || meta.film_grain) && (key !== 'sharpening' || meta.sharpening)) {
      addRow(grid, FIELD_LABELS[key], meta[key], '');
    }
  }
}

function addToHistory(url, meta) {
  history.unshift({ url, meta, time: Date.now() });
  if (history.length > 10) {
    const removed = history.pop();
    if (removed.url) URL.revokeObjectURL(removed.url);
  }
  renderHistory();
}

function renderHistory() {
  const container = $('history_items');
  container.innerHTML = '';
  history.forEach((item, i) => {
    const div = document.createElement('div');
    div.className = 'hist-item' + (i === 0 ? ' current' : '');
    const img = document.createElement('img');
    img.src = item.url;
    img.alt = 'generation ' + (i + 1);
    div.appendChild(img);
    const seed = document.createElement('span');
    seed.className = 'hist-seed';
    seed.textContent = item.meta && item.meta.seed != null ? item.meta.seed : '';
    div.appendChild(seed);
    div.addEventListener('click', () => selectHistory(i));
    container.appendChild(div);
  });
  $('history').classList.toggle('hidden', history.length === 0);
}

function selectHistory(i) {
  const item = history[i];
  if (!item) return;
  currentUrl = item.url;
  currentMeta = item.meta;
  $('image').src = item.url;
  $('image').style.display = 'block';
  $('placeholder').style.display = 'none';
  renderInfo(item.meta);
  applySettings(item.meta);
  setStageActions($('download'), $('info_btn'), true);
  const items = $('history_items').querySelectorAll('.hist-item');
  items.forEach((el, idx) => el.classList.toggle('current', idx === i));
}

function applySettings(meta) {
  if (!meta) return;
  $('prompt').value = meta.prompt ?? '';
  $('negative_prompt').value = meta.negative_prompt ?? '';
  if (meta.width != null) $('width').value = meta.width;
  if (meta.height != null) $('height').value = meta.height;
  if (meta.steps != null) $('steps').value = meta.steps;
  if (meta.guidance_scale != null) $('guidance_scale').value = meta.guidance_scale;
  if (meta.seed != null) $('seed').value = meta.seed;
  if (meta.sampler) $('sampler').value = meta.sampler;
  if (meta.upscale_factor != null) {
    setRange('upscale_factor', 'upscale_factor_val', meta.upscale_factor);
  } else if (meta.upscale === true) {
    // legacy metadata: 'upscale: true' == 2x refined
    setRange('upscale_factor', 'upscale_factor_val', 2);
    $('upscale_type').value = 'refined';
  }
  if (meta.upscale_type) $('upscale_type').value = meta.upscale_type;
  if (meta.qwen_vae_enhance != null) $('qwen_vae_enhance').checked = meta.qwen_vae_enhance;
  if (meta.film_grain != null) setRange('film_grain', 'film_grain_val', meta.film_grain);
  if (meta.sharpening != null) setRange('sharpening', 'sharpening_val', meta.sharpening);
  if (Array.isArray(meta.lora_specs)) {
    $('lora_specs').value = meta.lora_specs.join('\n');
  }
  if (meta.pixel_upscaler) {
    $('pixel_upscaler').value = meta.pixel_upscaler;
  }
  updateUpscaleMax();
}

$('swap').addEventListener('click', () => {
  const w = $('width'), h = $('height');
  const tmp = w.value; w.value = h.value; h.value = tmp;
});

function download(url, filename) {
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// Toggle the download + info buttons together for a stage's image.
function setStageActions(dlBtn, infoBtn, enabled) {
  dlBtn.disabled = !enabled;
  infoBtn.disabled = !enabled;
}

$('download').addEventListener('click', () => {
  const seed = currentMeta && currentMeta.seed != null ? currentMeta.seed : 'x';
  download(currentUrl, `thenoise_${seed}.png`);
});

let loras = [];

async function loadLoras() {
  const data = await fetchJSON('/lora');
  loras = data ? (data.loras || []).sort() : [];
}

let upscalerScales = {};  // name -> detected native scale (2/4)

function fillSelect(sel, names) {
  sel.innerHTML = '';
  for (const name of names) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  }
}

// Fetch JSON with graceful error handling; returns null on failure.
async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (res.ok) return await res.json();
  } catch (e) { /* fall through to null */ }
  return null;
}

async function loadUpscalers() {
  const sel = $('pixel_upscaler');
  const none = sel.firstElementChild; // the empty 'none' option
  sel.innerHTML = '';
  sel.appendChild(none);
  let names = [];
  const data = await fetchJSON('/upscalers');
  if (data) {
    names = (data.upscalers || []).sort();
    upscalerScales = data.scales || {};
    fillSelect(sel, names);
    fillSelect($('upscaler_model'), names);
    applyUpscalerDefaults();
  }
  // No upscaler models found: block the Upscale tab, leave the Generate
  // tab's pixel-upscaler dropdown empty (refiner-only), hide the selector
  // and drop the no-refiner upscale_type option.
  const noRefiner = $('no_refiner_opt');
  const noUpscalers = names.length === 0;
  $('no_upscaler').classList.toggle('hidden', !noUpscalers);
  $('pixel_upscaler_field').classList.toggle('hidden', noUpscalers);
  if (noUpscalers) {
    if (noRefiner.parentElement) noRefiner.remove();
    if ($('upscale_type').value === 'no-refiner') $('upscale_type').value = 'refined';
  } else {
    $('upscale_type').appendChild(noRefiner);
  }
  updateUpscaleMax();
}

const acEl = () => $('lora_ac');

function openAc() { acEl().classList.add('open'); }
function closeAc() { acEl().classList.remove('open'); acEl().innerHTML = ''; }

function tokenAtCursor(ta) {
  const s = ta.selectionStart;
  const before = ta.value.slice(0, s);
  const nl = before.lastIndexOf('\n');
  const lineStart = nl === -1 ? 0 : nl + 1;
  const sp = before.lastIndexOf(' ');
  const tokenStart = sp > lineStart ? sp + 1 : lineStart;
  return { token: ta.value.slice(tokenStart, s).trim(), tokenStart };
}

function insertLora(name, tokenStart) {
  const ta = $('lora_specs');
  const s = ta.selectionStart;
  ta.value = ta.value.slice(0, tokenStart) + name + ta.value.slice(s);
  const pos = tokenStart + name.length;
  ta.setSelectionRange(pos, pos);
  ta.focus();
}

function renderAc(items, tokenStart) {
  const ac = acEl();
  ac.innerHTML = '';
  if (!items.length) {
    const d = document.createElement('div');
    d.className = 'empty';
    d.textContent = 'No matching LoRAs';
    ac.appendChild(d);
    openAc();
    return;
  }
  items.forEach(name => {
    const d = document.createElement('div');
    d.className = 'item';
    d.dataset.name = name;
    d.dataset.tokenStart = tokenStart;
    d.textContent = name;
    ac.appendChild(d);
  });
  openAc();
}

$('lora_specs').addEventListener('input', () => {
  const ta = $('lora_specs');
  const { token, tokenStart } = tokenAtCursor(ta);
  if (token.length < 2) { closeAc(); return; }
  const q = token.toLowerCase();
  renderAc(loras.filter(n => n.toLowerCase().includes(q)), tokenStart);
});

acEl().addEventListener('mousedown', e => {
  const item = e.target.closest('.item');
  if (!item) return;
  e.preventDefault();
  insertLora(item.dataset.name, parseInt(item.dataset.tokenStart, 10));
  closeAc();
});

$('lora_specs').addEventListener('keydown', e => {
  if (!acEl().classList.contains('open')) return;
  if (e.key === 'Escape') { closeAc(); return; }
  const items = acEl().querySelectorAll('.item');
  if (!items.length) return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    let idx = -1;
    items.forEach((it, i) => { if (it.classList.contains('sel')) idx = i; });
    idx = e.key === 'ArrowDown'
      ? (idx + 1) % items.length
      : (idx === -1 ? items.length - 1 : (idx - 1 + items.length) % items.length);
    items.forEach((it, i) => it.classList.toggle('sel', i === idx));
  } else if (e.key === 'Enter' || e.key === 'Tab') {
    const sel = acEl().querySelector('.item.sel');
    if (sel) {
      e.preventDefault();
      insertLora(sel.dataset.name, parseInt(sel.dataset.tokenStart, 10));
      closeAc();
    }
  }
});

document.addEventListener('click', e => {
  if (!e.target.closest('.lora-field')) closeAc();
});

loadLoras();
loadUpscalers();
updateUpscaleMax();

// If the server is running without a loaded model, show a notice on the
// Generate tab explaining that it is unavailable (Upscale stays usable).
async function applyModelState() {
  let hasModel = true;
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      hasModel = (data.models || []).length > 0;
    }
  } catch (e) { /* assume a model is present on network errors */ }
  $('no_model').classList.toggle('hidden', hasModel);
}
applyModelState();

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return await res.blob();
}

// Runs a request while the button is disabled, the overlay shown and a live
// timer ticking; resets everything in finally and reports done/error states.
async function runWithBusy({ btn, overlay, timerEl, timerTextEl, request, onSuccess }) {
  const set = (state, text) => setTimer(timerEl, timerTextEl, state, text);
  btn.disabled = true;
  overlay.classList.remove('hidden');
  const start = Date.now();
  const elapsed = () => ((Date.now() - start) / 1000).toFixed(1) + 's';
  set('running', '0.0s');
  const timer = setInterval(() => set('running', elapsed()), 100);
  try {
    const result = await request();
    await onSuccess(result);
    set('done', elapsed());
  } catch (e) {
    set('error', 'error: ' + e.message);
  } finally {
    clearInterval(timer);
    overlay.classList.add('hidden');
    btn.disabled = false;
  }
}

$('generate').addEventListener('click', () => {
  const MAX_DIM = 4096;
  for (const f of ['width', 'height']) {
    const v = $(f).value === '' ? null : parseInt($(f).value, 10);
    if (v !== null && (v < 0 || v > MAX_DIM)) {
      alert(`error: ${f} must be between 0 and ${MAX_DIM} (got ${v}).`);
      return;
    }
  }
  runWithBusy({
    btn: $('generate'),
    overlay: $('overlay'),
    timerEl: 'timer',
    timerTextEl: 'timer_text',
    request: async () => {
      const body = {
        prompt: $('prompt').value,
        negative_prompt: $('negative_prompt').value,
        upscale_factor: parseFloat($('upscale_factor').value),
        upscale_type: $('upscale_type').value,
        pixel_upscaler: $('pixel_upscaler').value || null,
        qwen_vae_enhance: $('qwen_vae_enhance').checked,
        film_grain: parseFloat($('film_grain').value),
        sharpening: parseFloat($('sharpening').value),
        lora_specs: $('lora_specs').value.trim()
          ? $('lora_specs').value.split('\n').map(l => l.trim()).filter(Boolean)
          : null,
      };
      for (const f of ['width', 'height', 'steps', 'seed']) {
        const v = $(f).value;
        if (v !== '') body[f] = parseInt(v, 10);
      }
      for (const f of ['guidance_scale']) {
        const v = $(f).value;
        if (v !== '') body[f] = parseFloat(v);
      }
      const samplerVal = $('sampler').value;
      if (samplerVal) body.sampler = samplerVal;

      return await postJSON('/text2image', body);
    },
    onSuccess: async (blob) => {
      currentUrl = URL.createObjectURL(blob);
      setStageActions($('download'), $('info_btn'), true);
      $('image').src = currentUrl;
      $('image').style.display = 'block';
      $('placeholder').style.display = 'none';
      const meta = await decodeMeta(blob, 'generation_data');
      currentMeta = meta;
      renderInfo(meta);
      addToHistory(currentUrl, meta);
    },
  });
});

document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('hidden', v.id !== 'view-' + t.dataset.tab));
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
}));

let uInputB64 = null;      // base64 (no prefix) sent to /upscale
let uInputDims = null;     // {w, h} of the input
let uOutUrl = null;        // object URL of the upscaled result
let uOutMeta = null;       // upscale metadata

const dz = $('dropzone');
const fileInput = $('file');
const dzName = $('dz_name');

function loadUpscaleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    const img = new Image();
    img.onload = () => {
      uInputB64 = dataUrl.split(',')[1];
      uInputDims = { w: img.naturalWidth, h: img.naturalHeight };
      updateUpscaleTabFactor();
      // reset any previous result
      if (uOutUrl) { URL.revokeObjectURL(uOutUrl); uOutUrl = null; }
      uOutMeta = null;
      setStageActions($('udownload'), $('uinfo_btn'), false);
      // show the selected image as a single image in the stage
      $('usingle_img').src = dataUrl;
      $('usingle').classList.remove('hidden');
      $('uresult').classList.add('hidden');
      $('uplaceholder').style.display = 'none';
      // show the file name in the dropzone instead of a preview image
      dzName.textContent = file.name;
      dzName.style.display = 'block';
      $('dz_hint').style.display = 'none';
      $('upscale').disabled = false;
      applyUpscalerDefaults();
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

dz.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => loadUpscaleFile(e.target.files[0]));
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  loadUpscaleFile(e.dataTransfer.files[0]);
});

/* Upscale factor slider: capped at the selected upscaler's native scale. */
function updateUpscaleTabFactor() {
  const factor = parseFloat($('u_factor').value) || 1;
  $('u_factor_val').textContent = factor.toFixed(2);
  renderRes($('u_final_res'),
    uInputDims && uInputDims.w, uInputDims && uInputDims.h, factor);
}
function updateUpscaleTabMax() {
  const model = $('upscaler_model').value;
  const max = model ? (upscalerScales[model] || 2) : 1;
  clampSlider($('u_factor'), max);
  updateUpscaleTabFactor();
}

/* Pre-fill the upscale factor from the selected upscaler's native scale. */
function applyUpscalerDefaults() {
  const model = $('upscaler_model').value;
  const scale = upscalerScales[model];
  if (scale) $('u_factor').value = scale;
  updateUpscaleTabMax();
}
$('u_factor').addEventListener('input', updateUpscaleTabFactor);
$('upscaler_model').addEventListener('change', applyUpscalerDefaults);

/* upscale info panel (reuses addRow for the dl grid) */
function renderUpscaleInfo(meta) {
  const grid = $('info_modal_grid');
  grid.innerHTML = '';
  addRow(grid, 'Input resolution', uInputDims ? uInputDims.w + ' × ' + uInputDims.h : '—');
  const up = $('u_img');
  if (up.naturalWidth) addRow(grid, 'Output resolution', up.naturalWidth + ' × ' + up.naturalHeight);
  if (meta) {
    addRow(grid, 'Upscaler model', meta.upscaler_model);
    addRow(grid, 'Upscale factor', meta.upscale_factor);
  }
}

$('upscale').addEventListener('click', () => {
  const model = $('upscaler_model').value;
  const factor = Math.max(1, parseFloat($('u_factor').value) || 1);
  if (!model || !uInputB64) {
    setTimer('utimer', 'utimer_text', 'error',
      !model ? 'error: select an upscaler model' : 'error: load an input image');
    return;
  }

  runWithBusy({
    btn: $('upscale'),
    overlay: $('uoverlay'),
    timerEl: 'utimer',
    timerTextEl: 'utimer_text',
    request: async () => {
      return await postJSON('/upscale', {
        image_b64: uInputB64,
        upscale_factor: factor,
        pixel_upscaler: model,
      });
    },
    onSuccess: async (blob) => {
      if (uOutUrl) URL.revokeObjectURL(uOutUrl);
      uOutUrl = URL.createObjectURL(blob);
      const up = $('u_img');
      up.src = uOutUrl;
      // hide single view + placeholder, show the upscaled image
      $('usingle').classList.add('hidden');
      $('uplaceholder').style.display = 'none';
      $('uresult').classList.remove('hidden');
      setStageActions($('udownload'), $('uinfo_btn'), true);
      uOutMeta = await decodeMeta(blob, 'upscale_data');
      renderUpscaleInfo(uOutMeta);
    },
  });
});

$('udownload').addEventListener('click', () => {
  download(uOutUrl, 'upscaled.png');
});

/* ---------- info popup ---------- */
function openInfo(title, render) {
  render();
  $('info_modal_title').textContent = title;
  $('info_modal').classList.remove('hidden');
}
function closeInfo() {
  $('info_modal').classList.add('hidden');
}
$('info_btn').addEventListener('click', () => openInfo('Image info', () => renderInfo(currentMeta)));
$('uinfo_btn').addEventListener('click', () => openInfo('Upscale info', () => renderUpscaleInfo(uOutMeta)));
$('info_modal_close').addEventListener('click', closeInfo);
$('info_backdrop').addEventListener('click', closeInfo);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeInfo(); });
