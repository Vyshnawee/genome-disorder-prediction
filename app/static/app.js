// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const CLASS_STYLE = {
  "Thalassemia": { short: "Thalassemia", color: "var(--c-thal)" },
  "Hemophilia": { short: "Hemophilia", color: "var(--c-hemo)" },
  "Breast Cancer (BRCA1/p53)": { short: "Breast Cancer", color: "var(--c-brca)" },
  "Sickle Cell Anemia": { short: "Sickle Cell", color: "var(--c-sickle)" },
  "Cystic Fibrosis": { short: "Cystic Fibrosis", color: "var(--c-cf)" },
};
const shortName = (name) => CLASS_STYLE[name]?.short ?? name;
const classColor = (name) => CLASS_STYLE[name]?.color ?? "var(--primary)";
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;

const ICON = {
  warn: '<svg viewBox="0 0 24 24"><path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
  ok: '<svg viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg>',
  bad: '<svg viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>',
};

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.add("hidden"), 3500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => (typeof d === "string" ? d : d.msg)).join(" ")
      : data.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

// ---------------------------------------------------------
// Tabs
// ---------------------------------------------------------

function showView(name) {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  history.replaceState(null, "", `#${name}`);
  if (name === "performance") loadMetrics();
}

$$(".tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view)));

// ---------------------------------------------------------
// Patient form (built from /api/model-info)
// ---------------------------------------------------------

let modelInfo = null;

function fieldHtml(name, field) {
  const id = `f-${name.replace(/[^a-z0-9]/gi, "_")}`;

  if (field.type === "number") {
    return `
      <div class="field">
        <label for="${id}">${escapeHtml(field.label)}</label>
        <input type="number" id="${id}" data-name="${escapeHtml(name)}"
               min="0" step="${field.step}" placeholder="Not tested">
        <div class="hint">${field.hint ? `${escapeHtml(field.hint)} · ` : ""}data: ${field.min} – ${field.max}</div>
      </div>`;
  }

  const options = '<option value="">Unknown</option>' + field.options
    .map((option) => `<option value="${option.value}">${escapeHtml(option.label)}</option>`)
    .join("");
  return `
    <div class="field">
      <label for="${id}">${escapeHtml(field.label)}</label>
      <select id="${id}" data-name="${escapeHtml(name)}">${options}</select>
    </div>`;
}

function buildForm(info) {
  const html = info.sections.map((section, index) => {
    const body = `<div class="fields">${section.fields.map((n) => fieldHtml(n, info.fields[n])).join("")}</div>`;
    return `
      <div class="card section-card">
        <div class="section-title"><span class="section-num">${index + 1}</span><h3>${escapeHtml(section.title)}</h3></div>
        ${body}
      </div>`;
  }).join("");

  $("#form-sections").innerHTML = html;

  $$("#form-sections select").forEach((select) => {
    select.addEventListener("change", () => markUnknown(select));
  });
  resetForm();
}

function markUnknown(select) {
  select.classList.toggle("is-unknown", select.value === "");
}

function resetForm() {
  $$("#form-sections input").forEach((input) => (input.value = ""));
  $$("#form-sections select").forEach((select) => {
    select.value = "";
    markUnknown(select);
  });
}

function collectForm() {
  const patient = {};
  $$("#form-sections [data-name]").forEach((el) => {
    const name = el.dataset.name;
    patient[name] = el.value === "" ? null : Number(el.value);
  });
  return patient;
}

function fillForm(patient) {
  resetForm();
  $$("#form-sections [data-name]").forEach((el) => {
    const value = patient[el.dataset.name];
    if (value === null || value === undefined) return;
    if (el.tagName === "INPUT") {
      el.value = Number.isInteger(value) ? value : Number(value).toFixed(2);
    } else {
      el.value = String(Math.round(value));
      markUnknown(el);
    }
  });
}

// ---------------------------------------------------------
// Prediction result
// ---------------------------------------------------------

function renderResult(result, actual = null) {
  const color = classColor(result.prediction);
  const level = result.confidence_level.toLowerCase();
  const sorted = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);

  const bars = sorted.map(([name, p]) => `
    <div class="prob-row">
      <div class="top"><span>${escapeHtml(shortName(name))}</span><b>${pct(p)}</b></div>
      <div class="bar"><span style="background:${classColor(name)}" data-w="${(p * 100).toFixed(1)}"></span></div>
    </div>`).join("");

  let note;
  if (result.is_confident) {
    note = `<div class="note ok">${ICON.ok}<span>High-confidence prediction. These are the model's most reliable answers.</span></div>`;
  } else {
    note = `<div class="note warn">${ICON.warn}<span><b>Uncertain.</b> The two most likely categories are
      <b>${escapeHtml(shortName(result.top2[0]))}</b> or <b>${escapeHtml(shortName(result.top2[1]))}</b>. Confirmatory testing is recommended.</span></div>`;
  }

  let actualNote = "";
  if (actual) {
    const correct = actual === result.prediction;
    const inTop2 = result.top2.includes(actual);
    actualNote = `<div class="note ${correct ? "ok" : inTop2 ? "info" : "bad"}">${correct ? ICON.ok : inTop2 ? ICON.info : ICON.bad}
      <span>Actual (sample patient): <b>${escapeHtml(shortName(actual))}</b>, ${correct ? "prediction correct" : inTop2 ? "in the top-2" : "prediction missed"}.</span></div>`;
  }

  $("#result-body").innerHTML = `
    <div class="result-label">Most likely category</div>
    <div class="result-title"><span class="dot" style="background:${color}"></span>${escapeHtml(result.prediction)}</div>
    ${modelInfo?.class_info?.[result.prediction] ? `<p class="disease-info">${escapeHtml(modelInfo.class_info[result.prediction])}</p>` : ""}

    <div class="gauge">
      <div class="ring" style="--c:${color}; --p:${(result.confidence * 100).toFixed(1)}">
        <div class="ring-inner">${pct(result.confidence, 0)}</div>
      </div>
      <div class="gauge-text">
        <div class="big">Confidence</div>
        <span class="badge ${level}">${result.confidence_level}</span>
      </div>
    </div>

    <div class="result-label" style="margin-bottom:10px">Probability by category</div>
    <div class="probs">${bars}</div>
    ${note}
    ${actualNote}`;

  $("#result-empty").classList.add("hidden");
  $("#result-body").classList.remove("hidden");
  requestAnimationFrame(() => $$("#result-body .bar > span").forEach((bar) => (bar.style.width = `${bar.dataset.w}%`)));

  if (window.innerWidth < 1000) $("#result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runPrediction(actual = null) {
  const button = $("#btn-predict");
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span><span>Predicting…</span>';
  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectForm()),
    });
    renderResult(result, actual);
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = '<span>Predict disease</span><svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
  }
}

$("#patient-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});

$("#btn-reset").addEventListener("click", () => {
  resetForm();
  $("#result-body").classList.add("hidden");
  $("#result-empty").classList.remove("hidden");
});

$("#btn-sample").addEventListener("click", async () => {
  try {
    const sample = await api("/api/sample");
    fillForm(sample.patient);
    await runPrediction(sample.actual);
  } catch (error) {
    toast(error.message);
  }
});

// ---------------------------------------------------------
// Batch prediction
// ---------------------------------------------------------

let batchResults = [];

async function uploadBatch(file) {
  if (!file) return;
  const zone = $("#dropzone");
  $(".dz-title", zone).innerHTML = `Predicting <b>${escapeHtml(file.name)}</b>…`;

  const form = new FormData();
  form.append("file", file);

  try {
    const data = await api("/api/predict-batch", { method: "POST", body: form });
    batchResults = data.results;
    renderBatch(data, file.name);
  } catch (error) {
    toast(error.message);
  } finally {
    $(".dz-title", zone).innerHTML = "Drop a CSV here or <u>browse</u>";
    $("#batch-file").value = "";
  }
}

function renderBatch(data, fileName) {
  const counts = {};
  let confident = 0;
  data.results.forEach((r) => {
    counts[r.prediction] = (counts[r.prediction] || 0) + 1;
    if (r.is_confident) confident += 1;
  });

  const classKpis = (modelInfo?.classes ?? Object.keys(CLASS_STYLE)).map((name) => `
    <div class="kpi">
      <div class="k-label"><span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${classColor(name)};margin-right:6px"></span>${escapeHtml(shortName(name))}</div>
      <div class="k-value">${(counts[name] || 0).toLocaleString()}</div>
      <div class="k-sub">${pct((counts[name] || 0) / data.rows)} of patients</div>
    </div>`).join("");

  const accuracyKpi = data.accuracy == null ? "" : `
    <div class="kpi accent">
      <div class="k-label">Accuracy vs answers in file</div>
      <div class="k-value">${pct(data.accuracy)}</div>
      <div class="k-sub">${data.checked_rows.toLocaleString()} patients had a known disease</div>
    </div>`;

  $("#batch-kpis").innerHTML = `
    ${accuracyKpi}
    <div class="kpi">
      <div class="k-label">Patients predicted</div>
      <div class="k-value">${data.rows.toLocaleString()}</div>
      <div class="k-sub">${escapeHtml(fileName)} · ${data.matched_columns}/${data.total_columns} features found</div>
    </div>
    ${classKpis}
    <div class="kpi">
      <div class="k-label">High confidence</div>
      <div class="k-value">${confident.toLocaleString()}</div>
      <div class="k-sub">${pct(confident / data.rows)} of patients</div>
    </div>`;

  const shown = data.results.slice(0, 300);
  $("#batch-note").textContent = data.rows > shown.length ? `(showing first ${shown.length} of ${data.rows.toLocaleString()})` : "";
  $("#batch-table").innerHTML = `
    <thead><tr><th>Patient</th><th>Prediction</th>${data.accuracy != null ? "<th>Actual</th>" : ""}<th>Confidence</th><th>Level</th><th>Top-2</th></tr></thead>
    <tbody>${shown.map((r) => `
      <tr>
        <td>${escapeHtml(r.id)}</td>
        <td><span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${classColor(r.prediction)};margin-right:8px"></span>${escapeHtml(shortName(r.prediction))}</td>
        ${data.accuracy != null ? `<td>${r.actual ? `${escapeHtml(shortName(r.actual))} ${r.actual === r.prediction ? "✅" : "❌"}` : "–"}</td>` : ""}
        <td class="num">${pct(r.confidence)}</td>
        <td><span class="badge ${r.confidence_level.toLowerCase()}">${r.confidence_level}</span></td>
        <td class="muted">${r.top2.map(shortName).join(" / ")}</td>
      </tr>`).join("")}
    </tbody>`;

  $("#batch-output").classList.remove("hidden");
}

function downloadBatch() {
  const classes = modelInfo?.classes ?? Object.keys(CLASS_STYLE);
  const quote = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const hasActual = batchResults.some((r) => "actual" in r);
  const header = ["Patient Id", "Predicted Disease", ...(hasActual ? ["Actual Disease"] : []), "Confidence", "Confidence Level", ...classes.map((c) => `P(${shortName(c)})`)];
  const lines = batchResults.map((r) => [
    r.id, r.prediction, ...(hasActual ? [r.actual ?? ""] : []), r.confidence, r.confidence_level, ...classes.map((c) => r.probabilities[c]),
  ].map(quote).join(","));
  const blob = new Blob([[header.map(quote).join(","), ...lines].join("\n")], { type: "text/csv" });
  const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "predictions.csv" });
  link.click();
  URL.revokeObjectURL(link.href);
}

$("#batch-file").addEventListener("change", (e) => uploadBatch(e.target.files[0]));
$("#btn-download").addEventListener("click", downloadBatch);
const zone = $("#dropzone");
["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
zone.addEventListener("drop", (e) => uploadBatch(e.dataTransfer.files[0]));

// ---------------------------------------------------------
// Performance page
// ---------------------------------------------------------

let metricsLoaded = false;

function hbar(label, value, max, color, display, sub = "") {
  return `
    <div class="hbar">
      <span class="lbl" title="${escapeHtml(label)}">${escapeHtml(label)}${sub ? `<small class="sub">${escapeHtml(sub)}</small>` : ""}</span>
      <div class="bar"><span style="background:${color}" data-w="${Math.max(0, (value / max) * 100).toFixed(1)}"></span></div>
      <span class="val">${display}</span>
    </div>`;
}

async function loadMetrics() {
  if (metricsLoaded) return;
  try {
    const m = await api("/api/metrics");
    metricsLoaded = true;
    renderMetrics(m);
  } catch (error) {
    $("#perf-content").innerHTML = `<div class="card">Could not load metrics: ${escapeHtml(error.message)}</div>`;
  }
}

function renderMetrics(m) {
  const threshold = m.confidence_table.find((r) => r.threshold === m.confidence_threshold) ?? m.confidence_table.at(-1);
  $("#perf-sub").innerHTML = `Model: <b>${escapeHtml(m.model_name)}</b> · tested on <b>${m.holdout_size.toLocaleString()}</b> patients the model never saw during training.`;

  const kpis = `
    <div class="kpis">
      <div class="kpi accent">
        <div class="k-label">Top-2 accuracy</div>
        <div class="k-value">${pct(m.top2_accuracy)}</div>
        <div class="k-sub">True category is in the top 2 answers</div>
      </div>
      <div class="kpi">
        <div class="k-label">High-confidence accuracy</div>
        <div class="k-value">${pct(threshold.accuracy)}</div>
        <div class="k-sub">When confidence ≥ ${pct(threshold.threshold, 0)} (${pct(threshold.coverage, 0)} of patients)</div>
      </div>
      <div class="kpi">
        <div class="k-label">Overall accuracy</div>
        <div class="k-value">${pct(m.accuracy)}</div>
        <div class="k-sub">vs ${pct(m.baseline_accuracy)} by always guessing the most common class</div>
      </div>
      <div class="kpi">
        <div class="k-label">AUC (macro)</div>
        <div class="k-value">${m.auc_macro.toFixed(3)}</div>
        <div class="k-sub">Ranking quality (0.5 = random, 1 = perfect)</div>
      </div>
      <div class="kpi">
        <div class="k-label">Macro F1</div>
        <div class="k-value">${m.macro_f1.toFixed(3)}</div>
        <div class="k-sub">Balanced score across all classes</div>
      </div>
    </div>`;

  const confidenceBars = m.confidence_table.map((r) =>
    hbar(`Confidence ≥ ${pct(r.threshold, 0)}`, r.accuracy ?? 0, 1, "var(--primary)", r.accuracy == null ? "n/a" : pct(r.accuracy), `${pct(r.coverage, 0)} of patients`)
  ).join("") + hbar("All patients", m.accuracy, 1, "var(--muted)", pct(m.accuracy), "100% of patients");

  const classBars = m.classes.map((name) => {
    const s = m.per_class[name];
    return hbar(`${shortName(name)} (recall)`, s.recall, 1, classColor(name), pct(s.recall)) +
           hbar(`${shortName(name)} (AUC)`, m.auc_per_class[name], 1, classColor(name), m.auc_per_class[name].toFixed(3));
  }).join("");

  const rowTotals = m.confusion_matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const cmCells = m.confusion_matrix.map((row, i) => {
    const cells = row.map((count, j) => {
      const share = count / rowTotals[i];
      const color = i === j ? "var(--good)" : "var(--bad)";
      const strong = share > 0.45;
      return `<div class="cell" style="background:color-mix(in srgb, ${color} ${Math.round(8 + share * 72)}%, var(--surface-2)); color:${strong ? "#fff" : "var(--text)"}">
        ${count.toLocaleString()}<small>${pct(share, 0)}</small></div>`;
    }).join("");
    return `<div class="h rh">${escapeHtml(shortName(m.classes[i]))}</div>${cells}`;
  }).join("");

  const maxImportance = Math.max(...m.feature_importance.map((f) => f.importance), 1e-9);
  const importanceBars = m.feature_importance.slice(0, 10).map((f) =>
    hbar(f.label, f.importance, maxImportance, "var(--primary)", `${(f.importance * 100).toFixed(2)}`)
  ).join("");

  const comparison = `
    <table class="table">
      <thead><tr><th>Model</th><th>CV accuracy</th><th>CV macro F1</th><th>CV log loss ↓</th><th>Train accuracy</th></tr></thead>
      <tbody>${m.model_comparison.map((r) => `
        <tr class="${r.Model === m.model_name ? "best" : ""}">
          <td><b>${escapeHtml(r.Model)}</b>${r.Model === m.model_name ? ' <span class="badge high">selected</span>' : ""}</td>
          <td class="num">${pct(r["CV Accuracy"])} <span class="muted small">± ${pct(r["CV Accuracy Std"])}</span></td>
          <td class="num">${r["CV Macro F1"].toFixed(3)}</td>
          <td class="num">${r["CV Log Loss"].toFixed(3)}</td>
          <td class="num">${pct(r["Train Accuracy"])}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;

  $("#perf-content").innerHTML = `
    ${kpis}
    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h3>Accuracy by confidence</h3></div>
        <p class="desc muted">The more confident the model is, the more often it is right.</p>
        <div class="hbars">${confidenceBars}</div>
      </div>
      <div class="card">
        <div class="card-head"><h3>Per-class results</h3></div>
        <p class="desc muted">Recall = share of each category the model finds. AUC = how well it separates that category from the others.</p>
        <div class="hbars">${classBars}</div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h3>Confusion matrix</h3></div>
        <p class="desc muted">Rows = actual category, columns = predicted. Green diagonal = correct.</p>
        <div class="cm" style="grid-template-columns: 110px repeat(${m.classes.length}, 1fr)">
          <div></div>${m.classes.map((c) => `<div class="h">${escapeHtml(shortName(c))}</div>`).join("")}
          ${cmCells}
        </div>
        <div class="cm-axis">Predicted →</div>
      </div>
      <div class="card">
        <div class="card-head"><h3>Most important inputs</h3></div>
        <p class="desc muted">Accuracy drop (in % points) when the input is shuffled. Higher means more important.</p>
        <div class="hbars">${importanceBars}</div>
      </div>
    </div>
    <div class="card">
      <div class="card-head"><h3>Model comparison</h3><span class="muted small">5-fold cross-validation on the training split · lowest log loss wins (most reliable confidence)</span></div>
      <div class="table-wrap">${comparison}</div>
    </div>`;

  requestAnimationFrame(() => $$("#perf-content .bar > span").forEach((bar) => (bar.style.width = `${bar.dataset.w}%`)));
}

// ---------------------------------------------------------
// Model scores card (Predict page)
// ---------------------------------------------------------

function renderScoresCard(m) {
  const classes = m.classes;
  const mean = (key) => classes.reduce((sum, c) => sum + m.per_class[c][key], 0) / classes.length;

  const rows = classes.map((name) => {
    const s = m.per_class[name];
    return `<tr>
      <td><span class="dot" style="background:${classColor(name)}"></span>${escapeHtml(shortName(name))}</td>
      <td>${pct(s.precision)}</td><td>${pct(s.recall)}</td><td>${s.f1.toFixed(3)}</td>
    </tr>`;
  }).join("");

  $("#model-scores").innerHTML = `
    <div class="scores-head">
      <h3>Model scores</h3>
      <span class="muted small">${escapeHtml(m.model_name)}</span>
    </div>
    <div class="score-tiles">
      <div class="score-tile"><div class="v">${pct(m.accuracy)}</div><div class="l">Accuracy</div></div>
      <div class="score-tile"><div class="v">${pct(mean("precision"))}</div><div class="l">Precision</div></div>
      <div class="score-tile"><div class="v">${pct(mean("recall"))}</div><div class="l">Recall</div></div>
      <div class="score-tile"><div class="v">${m.macro_f1.toFixed(3)}</div><div class="l">F1 score</div></div>
    </div>
    <table class="scores-table">
      <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>
        ${rows}
        <tr><td>Average (macro)</td><td>${pct(mean("precision"))}</td><td>${pct(mean("recall"))}</td><td>${m.macro_f1.toFixed(3)}</td></tr>
      </tbody>
    </table>
    <div class="scores-foot">Tested on ${m.holdout_size.toLocaleString()} patients the model never saw during training.</div>`;
}

// ---------------------------------------------------------
// Start
// ---------------------------------------------------------

(async function init() {
  try {
    modelInfo = await api("/api/model-info");
    buildForm(modelInfo);
  } catch (error) {
    $("#form-sections").innerHTML = `<div class="card">Could not load the model. Run <code>python main.py</code> first, then restart the server.<br><span class="muted small">${escapeHtml(error.message)}</span></div>`;
  }
  try {
    renderScoresCard(await api("/api/metrics"));
  } catch (error) {
    $("#model-scores").innerHTML = `<div class="muted small">Model scores unavailable: ${escapeHtml(error.message)}</div>`;
  }
  const start = location.hash.replace("#", "");
  if (["predict", "batch", "performance", "about"].includes(start)) showView(start);
})();
