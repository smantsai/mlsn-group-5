const state = {
  config: null,
  datasetAvailable: false,
  split: "test",
  search: "",
  offset: 0,
  limit: 60,
  total: 0,
  thresholdPct: 60,
  opacityPct: 55,
  viewMode: "overlay",
  visibleClasses: new Set([0, 1, 2, 3]),
  current: null, // { width, height, origRgba, predClasses, confidences, gtClasses, source, meanConfidence }
  activeSampleIndex: null,
};

const els = {
  modelMeta: document.getElementById("modelMeta"),
  randomTestBtn: document.getElementById("randomTestBtn"),
  fileInput: document.getElementById("fileInput"),
  dropZone: document.getElementById("dropZone"),
  sourceInfo: document.getElementById("sourceInfo"),
  thresholdSlider: document.getElementById("thresholdSlider"),
  thresholdNumber: document.getElementById("thresholdNumber"),
  thresholdText: document.getElementById("thresholdText"),
  opacitySlider: document.getElementById("opacitySlider"),
  opacityValue: document.getElementById("opacityValue"),
  classList: document.getElementById("classList"),
  coverageBadge: document.getElementById("coverageBadge"),
  datasetTotal: document.getElementById("datasetTotal"),
  splitTabs: document.getElementById("splitTabs"),
  datasetSearch: document.getElementById("datasetSearch"),
  datasetList: document.getElementById("datasetList"),
  prevPageBtn: document.getElementById("prevPageBtn"),
  nextPageBtn: document.getElementById("nextPageBtn"),
  pageLabel: document.getElementById("pageLabel"),
  viewModes: document.getElementById("viewModes"),
  gtModeBtn: document.getElementById("gtModeBtn"),
  downloadOverlayBtn: document.getElementById("downloadOverlayBtn"),
  downloadMaskBtn: document.getElementById("downloadMaskBtn"),
  metricsStrip: document.getElementById("metricsStrip"),
  emptyState: document.getElementById("emptyState"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  canvasGrid: document.getElementById("canvasGrid"),
  primaryCard: document.getElementById("primaryCard"),
  secondaryCard: document.getElementById("secondaryCard"),
  primaryLabel: document.getElementById("primaryLabel"),
  secondaryLabel: document.getElementById("secondaryLabel"),
  primaryCanvas: document.getElementById("primaryCanvas"),
  secondaryCanvas: document.getElementById("secondaryCanvas"),
  hoverReadout: document.getElementById("hoverReadout"),
  imageDims: document.getElementById("imageDims"),
};

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to decode image payload."));
    img.src = src;
  });
}

function imageToRgba(img, width, height) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height).data;
}

function getClassColor(clsId) {
  const colors = state.config?.colors || {};
  return colors[String(clsId)] || [180, 180, 180];
}

function rgbCss(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function confidenceRamp(t) {
  // t in [0, 1]: dark blue -> teal -> amber -> bright mint
  const stops = [
    [0.0, [30, 27, 75]],
    [0.45, [14, 116, 144]],
    [0.75, [245, 158, 11]],
    [1.0, [74, 222, 128]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const u = (t - t0) / Math.max(1e-6, t1 - t0);
      return [
        Math.round(c0[0] + u * (c1[0] - c0[0])),
        Math.round(c0[1] + u * (c1[1] - c0[1])),
        Math.round(c0[2] + u * (c1[2] - c0[2])),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

async function init() {
  const res = await fetch("/api/config");
  const data = await res.json();
  state.config = data.model;
  state.datasetAvailable = Boolean(data.dataset?.available);

  const miou = state.config.test_miou != null ? `Test mIoU ${state.config.test_miou}` : "";
  els.modelMeta.textContent = `${state.config.encoder} • ${state.config.input_size}×${state.config.input_size} • ${state.config.device_name} ${miou ? "• " + miou : ""}`;

  buildClassList();
  bindEvents();

  if (state.datasetAvailable) {
    updateSplitTabCounts(data.dataset.counts || {});
    await fetchDatasetPage();
    await loadRandomSample("test");
  } else {
    els.randomTestBtn.disabled = true;
    els.datasetList.innerHTML = '<div class="muted">No dataset zip/CSV found.</div>';
  }
}

function buildClassList() {
  els.classList.innerHTML = "";
  const names = state.config.class_names;
  names.forEach((name, idx) => {
    const rgb = getClassColor(idx);
    const row = document.createElement("div");
    row.className = "class-row";
    row.innerHTML = `
      <div class="class-top">
        <label class="class-label">
          <input type="checkbox" data-cls="${idx}" checked />
          <span class="swatch" style="background:${rgbCss(rgb)}"></span>
          <span>${name}</span>
        </label>
        <span class="mono muted" id="clsPct-${idx}">0.0%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" id="clsBar-${idx}" style="background:${rgbCss(rgb)}"></div></div>
    `;
    const cb = row.querySelector("input");
    cb.addEventListener("change", () => {
      if (cb.checked) state.visibleClasses.add(idx);
      else state.visibleClasses.delete(idx);
      render();
    });
    els.classList.appendChild(row);
  });

  // Unlabeled row
  const unlRow = document.createElement("div");
  unlRow.className = "class-row";
  unlRow.innerHTML = `
    <div class="class-top">
      <div class="class-label">
        <span class="swatch" style="background:#334155"></span>
        <span>Unlabeled (&lt; threshold)</span>
      </div>
      <span class="mono muted" id="clsPct-unlabeled">0.0%</span>
    </div>
    <div class="bar-track"><div class="bar-fill" id="clsBar-unlabeled" style="background:#64748b"></div></div>
  `;
  els.classList.appendChild(unlRow);
}

function updateSplitTabCounts(counts) {
  els.splitTabs.querySelectorAll(".tab-btn").forEach((btn) => {
    const split = btn.dataset.split;
    const label = split.charAt(0).toUpperCase() + split.slice(1);
    const c = counts[split];
    btn.textContent = c != null ? `${label} (${c})` : label;
  });
}

function bindEvents() {
  els.fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    e.target.value = "";
  });

  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("drag-over");
  });
  els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("drag-over"));
  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) uploadFile(file);
  });

  els.randomTestBtn.addEventListener("click", () => loadRandomSample(state.split === "all" ? "test" : state.split));

  const setThreshold = (val) => {
    const clamped = Math.max(0, Math.min(100, Math.round(Number(val) || 0)));
    state.thresholdPct = clamped;
    els.thresholdSlider.value = String(clamped);
    els.thresholdNumber.value = String(clamped);
    els.thresholdText.textContent = `${clamped}%`;
    document.querySelectorAll(".preset-btn").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.threshold) === clamped);
    });
    render();
  };

  els.thresholdSlider.addEventListener("input", (e) => setThreshold(e.target.value));
  els.thresholdNumber.addEventListener("input", (e) => setThreshold(e.target.value));
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => setThreshold(btn.dataset.threshold));
  });

  els.opacitySlider.addEventListener("input", (e) => {
    state.opacityPct = Number(e.target.value);
    els.opacityValue.textContent = `${state.opacityPct}%`;
    render();
  });

  els.viewModes.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      state.viewMode = btn.dataset.mode;
      els.viewModes.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      render();
    });
  });

  els.splitTabs.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.split = btn.dataset.split;
      state.offset = 0;
      els.splitTabs.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      fetchDatasetPage();
    });
  });

  let searchTimer = null;
  els.datasetSearch.addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.search = e.target.value;
      state.offset = 0;
      fetchDatasetPage();
    }, 180);
  });

  els.prevPageBtn.addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.limit);
    fetchDatasetPage();
  });

  els.nextPageBtn.addEventListener("click", () => {
    if (state.offset + state.limit < state.total) {
      state.offset += state.limit;
      fetchDatasetPage();
    }
  });

  els.downloadOverlayBtn.addEventListener("click", () => downloadResult("overlay"));
  els.downloadMaskBtn.addEventListener("click", () => downloadResult("mask"));

  [els.primaryCanvas, els.secondaryCanvas].forEach((canvas) => {
    canvas.addEventListener("mousemove", (e) => handleHover(canvas, e));
    canvas.addEventListener("mouseleave", () => {
      els.hoverReadout.textContent = "Hover over the image to inspect per-pixel class & confidence.";
    });
  });
}

async function fetchDatasetPage() {
  const params = new URLSearchParams({
    split: state.split,
    search: state.search,
    offset: String(state.offset),
    limit: String(state.limit),
  });
  const res = await fetch(`/api/dataset?${params.toString()}`);
  const data = await res.json();
  state.total = data.total;
  els.datasetTotal.textContent = `${data.total} items`;

  const currentPage = Math.floor(state.offset / state.limit) + 1;
  const totalPages = Math.max(1, Math.ceil(data.total / state.limit));
  els.pageLabel.textContent = `Page ${currentPage} / ${totalPages}`;
  els.prevPageBtn.disabled = state.offset <= 0;
  els.nextPageBtn.disabled = state.offset + state.limit >= data.total;

  els.datasetList.innerHTML = "";
  if (!data.items.length) {
    els.datasetList.innerHTML = '<div class="muted" style="padding:8px;">No matching images.</div>';
    return;
  }

  for (const item of data.items) {
    const btn = document.createElement("button");
    btn.className = "dataset-item" + (item.index === state.activeSampleIndex ? " active" : "");
    btn.innerHTML = `
      <span class="dataset-name" title="${item.filename}">${item.filename}</span>
      <span class="split-pill">${item.split}</span>
    `;
    btn.addEventListener("click", () => loadDatasetSample(item.index));
    els.datasetList.appendChild(btn);
  }
}

function setLoading(isLoading) {
  els.loadingOverlay.classList.toggle("hidden", !isLoading);
}

async function uploadFile(file) {
  setLoading(true);
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/predict", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Inference failed.");
    }
    const payload = await res.json();
    state.activeSampleIndex = null;
    highlightActiveDatasetItem();
    await applyPredictionPayload(payload);
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
}

async function loadRandomSample(split) {
  setLoading(true);
  try {
    const res = await fetch(`/api/dataset/random?split=${encodeURIComponent(split)}`);
    if (!res.ok) throw new Error("Failed to load random sample.");
    const payload = await res.json();
    state.activeSampleIndex = payload.source?.index ?? null;
    highlightActiveDatasetItem();
    await applyPredictionPayload(payload);
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
}

async function loadDatasetSample(index) {
  setLoading(true);
  try {
    const res = await fetch(`/api/dataset/sample/${index}`);
    if (!res.ok) throw new Error("Failed to load sample.");
    const payload = await res.json();
    state.activeSampleIndex = index;
    highlightActiveDatasetItem();
    await applyPredictionPayload(payload);
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
}

function highlightActiveDatasetItem() {
  // Refresh visible list highlight without re-fetching
  const items = els.datasetList.querySelectorAll(".dataset-item");
  items.forEach((el) => el.classList.remove("active"));
  if (state.activeSampleIndex == null) return;
  const targetName = state.current?.source?.filename;
  if (!targetName) return;
  items.forEach((el) => {
    if (el.querySelector(".dataset-name")?.textContent === targetName) {
      el.classList.add("active");
    }
  });
}

async function applyPredictionPayload(payload) {
  const { width, height } = payload;
  const [origImg, predImg, gtImg] = await Promise.all([
    loadImageElement(payload.image_data_url),
    loadImageElement(payload.prediction_data_url),
    payload.ground_truth_data_url ? loadImageElement(payload.ground_truth_data_url) : Promise.resolve(null),
  ]);

  const origRgba = imageToRgba(origImg, width, height);
  const packedRgba = imageToRgba(predImg, width, height);
  const nPixels = width * height;
  const predClasses = new Uint8Array(nPixels);
  const confidences = new Float32Array(nPixels);

  for (let i = 0; i < nPixels; i++) {
    const p = i * 4;
    predClasses[i] = packedRgba[p];
    confidences[i] = ((packedRgba[p + 1] << 8) | packedRgba[p + 2]) / 65535.0;
  }

  let gtClasses = null;
  if (gtImg) {
    const gtRgba = imageToRgba(gtImg, width, height);
    gtClasses = new Uint8Array(nPixels);
    for (let i = 0; i < nPixels; i++) {
      gtClasses[i] = gtRgba[i * 4];
    }
  }

  state.current = {
    width,
    height,
    originalWidth: payload.original_width,
    originalHeight: payload.original_height,
    origRgba,
    predClasses,
    confidences,
    gtClasses,
    source: payload.source,
    meanConfidence: payload.mean_confidence,
  };

  els.gtModeBtn.disabled = !gtClasses;
  if (!gtClasses && state.viewMode === "gt") {
    state.viewMode = "overlay";
    els.viewModes.querySelectorAll(".mode-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === "overlay");
    });
  }

  els.downloadOverlayBtn.disabled = false;
  els.downloadMaskBtn.disabled = false;
  els.emptyState.classList.add("hidden");
  els.canvasGrid.classList.remove("hidden");

  updateSourceInfo(payload);
  highlightActiveDatasetItem();
  render();
}

function updateSourceInfo(payload) {
  const s = payload.source || {};
  els.sourceInfo.classList.remove("hidden");
  if (s.type === "dataset") {
    els.sourceInfo.innerHTML = `
      <div><strong>${s.filename}</strong></div>
      <div class="muted">Split: <strong>${s.split}</strong> • GT labeled: ${(s.labeled_frac * 100).toFixed(1)}%</div>
    `;
  } else {
    els.sourceInfo.innerHTML = `
      <div><strong>${s.filename || "Uploaded image"}</strong></div>
      <div class="muted">Mean confidence: ${(payload.mean_confidence * 100).toFixed(1)}%</div>
    `;
  }

  const dimNote =
    payload.width === payload.original_width && payload.height === payload.original_height
      ? `${payload.width} × ${payload.height}px`
      : `${payload.width} × ${payload.height}px (from ${payload.original_width} × ${payload.original_height}px)`;
  els.imageDims.textContent = dimNote;
}

function drawOverlayToCanvas(canvas, { maskOnly = false } = {}) {
  const { width, height, origRgba, predClasses, confidences } = state.current;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(width, height);
  const dst = out.data;

  const threshold = state.thresholdPct / 100.0;
  const alpha = state.opacityPct / 100.0;
  const invAlpha = 1.0 - alpha;
  const nPixels = width * height;
  const counts = new Uint32Array(state.config.num_classes);
  let labeledCount = 0;

  for (let i = 0; i < nPixels; i++) {
    const p = i * 4;
    const cls = predClasses[i];
    const conf = confidences[i];
    const passes = conf >= threshold;

    if (passes) {
      counts[cls]++;
      labeledCount++;
    }

    const showClass = passes && state.visibleClasses.has(cls);
    if (maskOnly) {
      if (showClass) {
        const rgb = getClassColor(cls);
        dst[p] = rgb[0];
        dst[p + 1] = rgb[1];
        dst[p + 2] = rgb[2];
        dst[p + 3] = 255;
      } else {
        dst[p] = 14;
        dst[p + 1] = 19;
        dst[p + 2] = 29;
        dst[p + 3] = 255;
      }
    } else if (showClass) {
      const rgb = getClassColor(cls);
      dst[p] = Math.round(origRgba[p] * invAlpha + rgb[0] * alpha);
      dst[p + 1] = Math.round(origRgba[p + 1] * invAlpha + rgb[1] * alpha);
      dst[p + 2] = Math.round(origRgba[p + 2] * invAlpha + rgb[2] * alpha);
      dst[p + 3] = 255;
    } else {
      dst[p] = origRgba[p];
      dst[p + 1] = origRgba[p + 1];
      dst[p + 2] = origRgba[p + 2];
      dst[p + 3] = 255;
    }
  }

  ctx.putImageData(out, 0, 0);
  return { counts, labeledCount, nPixels };
}

function drawOriginalToCanvas(canvas) {
  const { width, height, origRgba } = state.current;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(new Uint8ClampedArray(origRgba), width, height), 0, 0);
}

function drawConfidenceMapToCanvas(canvas) {
  const { width, height, confidences } = state.current;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(width, height);
  const dst = out.data;
  const threshold = state.thresholdPct / 100.0;
  const nPixels = width * height;

  for (let i = 0; i < nPixels; i++) {
    const p = i * 4;
    const conf = confidences[i];
    const rgb = confidenceRamp(conf);
    const dim = conf < threshold ? 0.35 : 1.0;
    dst[p] = Math.round(rgb[0] * dim);
    dst[p + 1] = Math.round(rgb[1] * dim);
    dst[p + 2] = Math.round(rgb[2] * dim);
    dst[p + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

function drawGroundTruthToCanvas(canvas) {
  const { width, height, origRgba, gtClasses } = state.current;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(width, height);
  const dst = out.data;
  const alpha = state.opacityPct / 100.0;
  const invAlpha = 1.0 - alpha;
  const nPixels = width * height;

  for (let i = 0; i < nPixels; i++) {
    const p = i * 4;
    const cls = gtClasses[i];
    if (cls < state.config.num_classes && state.visibleClasses.has(cls)) {
      const rgb = getClassColor(cls);
      dst[p] = Math.round(origRgba[p] * invAlpha + rgb[0] * alpha);
      dst[p + 1] = Math.round(origRgba[p + 1] * invAlpha + rgb[1] * alpha);
      dst[p + 2] = Math.round(origRgba[p + 2] * invAlpha + rgb[2] * alpha);
      dst[p + 3] = 255;
    } else {
      dst[p] = origRgba[p];
      dst[p + 1] = origRgba[p + 1];
      dst[p + 2] = origRgba[p + 2];
      dst[p + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
}

function computeStats() {
  const { predClasses, confidences, gtClasses } = state.current;
  const nPixels = predClasses.length;
  const numClasses = state.config.num_classes;
  const threshold = state.thresholdPct / 100.0;
  const counts = new Uint32Array(numClasses);
  let labeledCount = 0;

  const inter = new Uint32Array(numClasses);
  const union = new Uint32Array(numClasses);
  let gtValidCount = 0;
  let bothLabeledCount = 0;
  let correctCount = 0;

  for (let i = 0; i < nPixels; i++) {
    const passes = confidences[i] >= threshold;
    const pred = predClasses[i];
    if (passes) {
      counts[pred]++;
      labeledCount++;
    }
    if (gtClasses) {
      const gt = gtClasses[i];
      if (gt < numClasses) {
        gtValidCount++;
        if (passes) {
          bothLabeledCount++;
          if (pred === gt) correctCount++;
        }
        for (let c = 0; c < numClasses; c++) {
          const pMatch = passes && pred === c;
          const gMatch = gt === c;
          if (pMatch && gMatch) inter[c]++;
          if (pMatch || gMatch) union[c]++;
        }
      }
    }
  }

  let gtMetrics = null;
  if (gtClasses && gtValidCount > 0) {
    let iouSum = 0;
    let iouCount = 0;
    for (let c = 0; c < numClasses; c++) {
      if (union[c] > 0) {
        iouSum += inter[c] / union[c];
        iouCount++;
      }
    }
    gtMetrics = {
      miou: iouCount > 0 ? iouSum / iouCount : 0,
      accuracyOnLabeled: bothLabeledCount > 0 ? correctCount / bothLabeledCount : null,
      gtRetention: bothLabeledCount / gtValidCount,
    };
  }

  return { counts, labeledCount, nPixels, gtMetrics };
}

function updateStatsUi(stats) {
  const { counts, labeledCount, nPixels, gtMetrics } = stats;
  const labeledPct = (labeledCount / nPixels) * 100;
  const unlabeledPct = 100 - labeledPct;

  els.coverageBadge.textContent = `${labeledPct.toFixed(1)}% labeled`;

  for (let c = 0; c < state.config.num_classes; c++) {
    const pct = (counts[c] / nPixels) * 100;
    document.getElementById(`clsPct-${c}`).textContent = `${pct.toFixed(1)}%`;
    document.getElementById(`clsBar-${c}`).style.width = `${pct.toFixed(2)}%`;
  }
  document.getElementById("clsPct-unlabeled").textContent = `${unlabeledPct.toFixed(1)}%`;
  document.getElementById("clsBar-unlabeled").style.width = `${unlabeledPct.toFixed(2)}%`;

  els.metricsStrip.classList.remove("hidden");
  let html = `
    <div class="metric-chip"><span class="muted">Threshold:</span><strong>&ge; ${state.thresholdPct}%</strong></div>
    <div class="metric-chip"><span class="muted">Labeled:</span><strong>${labeledPct.toFixed(1)}%</strong></div>
    <div class="metric-chip"><span class="muted">Unlabeled:</span><strong>${unlabeledPct.toFixed(1)}%</strong></div>
    <div class="metric-chip"><span class="muted">Mean Conf:</span><strong>${(state.current.meanConfidence * 100).toFixed(1)}%</strong></div>
  `;

  if (gtMetrics) {
    const accStr = gtMetrics.accuracyOnLabeled != null ? `${(gtMetrics.accuracyOnLabeled * 100).toFixed(1)}%` : "—";
    html += `
      <div class="metric-chip"><span class="muted">Acc (labeled GT):</span><strong>${accStr}</strong></div>
      <div class="metric-chip"><span class="muted">GT Kept:</span><strong>${(gtMetrics.gtRetention * 100).toFixed(1)}%</strong></div>
      <div class="metric-chip"><span class="muted">mIoU (@${state.thresholdPct}%):</span><strong>${gtMetrics.miou.toFixed(4)}</strong></div>
    `;
  }
  els.metricsStrip.innerHTML = html;
}

function render() {
  if (!state.current) return;

  const stats = computeStats();
  updateStatsUi(stats);

  const mode = state.viewMode;
  const twoUp = mode === "side-by-side" || mode === "gt";
  els.canvasGrid.classList.toggle("two-up", twoUp);
  els.secondaryCard.classList.toggle("hidden", !twoUp);

  if (mode === "overlay") {
    els.primaryLabel.textContent = `Segmentation Overlay (≥ ${state.thresholdPct}% confidence)`;
    drawOverlayToCanvas(els.primaryCanvas, { maskOnly: false });
  } else if (mode === "mask") {
    els.primaryLabel.textContent = `Label Mask Only (≥ ${state.thresholdPct}% confidence)`;
    drawOverlayToCanvas(els.primaryCanvas, { maskOnly: true });
  } else if (mode === "side-by-side") {
    els.primaryLabel.textContent = `Original Image`;
    els.secondaryLabel.textContent = `Segmentation Overlay (≥ ${state.thresholdPct}%)`;
    drawOriginalToCanvas(els.primaryCanvas);
    drawOverlayToCanvas(els.secondaryCanvas, { maskOnly: false });
  } else if (mode === "confidence") {
    els.primaryLabel.textContent = `Confidence Heatmap (dimmed below ${state.thresholdPct}%)`;
    drawConfidenceMapToCanvas(els.primaryCanvas);
  } else if (mode === "gt") {
    els.primaryLabel.textContent = `Model Prediction (≥ ${state.thresholdPct}%)`;
    els.secondaryLabel.textContent = `Ground Truth Label`;
    drawOverlayToCanvas(els.primaryCanvas, { maskOnly: false });
    drawGroundTruthToCanvas(els.secondaryCanvas);
  }
}

function handleHover(canvas, event) {
  if (!state.current) return;
  const rect = canvas.getBoundingClientRect();
  const { width, height, predClasses, confidences, gtClasses } = state.current;
  const x = Math.min(width - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * width)));
  const y = Math.min(height - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * height)));
  const idx = y * width + x;
  const cls = predClasses[idx];
  const confPct = confidences[idx] * 100;
  const passes = confPct >= state.thresholdPct;
  const clsName = state.config.class_names[cls] || `class_${cls}`;

  let text = `(${x}, ${y}) • `;
  if (passes) {
    text += `Labeled: ${clsName.toUpperCase()} (${confPct.toFixed(1)}% conf)`;
  } else {
    text += `UNLABELED — top guess "${clsName}" at ${confPct.toFixed(1)}% < ${state.thresholdPct}% threshold`;
  }

  if (gtClasses) {
    const gt = gtClasses[idx];
    const gtName = gt < state.config.num_classes ? state.config.class_names[gt] : "unlabeled";
    text += ` • GT: ${gtName}`;
  }
  els.hoverReadout.textContent = text;
}

function downloadResult(kind) {
  if (!state.current) return;
  const { width, height, origRgba, predClasses, confidences, source } = state.current;
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = width;
  exportCanvas.height = height;
  const ctx = exportCanvas.getContext("2d");
  const out = ctx.createImageData(width, height);
  const dst = out.data;

  const threshold = state.thresholdPct / 100.0;
  const alpha = state.opacityPct / 100.0;
  const invAlpha = 1.0 - alpha;
  const nPixels = width * height;

  for (let i = 0; i < nPixels; i++) {
    const p = i * 4;
    const cls = predClasses[i];
    const show = confidences[i] >= threshold && state.visibleClasses.has(cls);

    if (kind === "mask") {
      if (show) {
        const rgb = getClassColor(cls);
        dst[p] = rgb[0];
        dst[p + 1] = rgb[1];
        dst[p + 2] = rgb[2];
        dst[p + 3] = 255;
      } else {
        dst[p + 3] = 0;
      }
    } else if (show) {
      const rgb = getClassColor(cls);
      dst[p] = Math.round(origRgba[p] * invAlpha + rgb[0] * alpha);
      dst[p + 1] = Math.round(origRgba[p + 1] * invAlpha + rgb[1] * alpha);
      dst[p + 2] = Math.round(origRgba[p + 2] * invAlpha + rgb[2] * alpha);
      dst[p + 3] = 255;
    } else {
      dst[p] = origRgba[p];
      dst[p + 1] = origRgba[p + 1];
      dst[p + 2] = origRgba[p + 2];
      dst[p + 3] = 255;
    }
  }

  ctx.putImageData(out, 0, 0);
  const baseName = (source?.filename || "image").replace(/\.[^.]+$/, "");
  const link = document.createElement("a");
  link.download = `${baseName}_${kind}_thr${state.thresholdPct}.png`;
  link.href = exportCanvas.toDataURL("image/png");
  link.click();
}

init();
