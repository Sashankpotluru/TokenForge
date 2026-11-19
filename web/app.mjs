import {Controller} from "./controller.mjs";
import {loadManifest, readVerified, validateRequest} from "./runtime.mjs";

const el = id => document.getElementById(id);
const set = (id, value) => { el(id).textContent = value; };
const format = n => n.toLocaleString("en-US");
const encoder = new TextEncoder();
const examples = {
  prose: el("source").value,
  code: "def square(x):\n    return x ** 2\n",
  unicode: "Cafe\u0301 meets café. 藍 🪶 مرحبًا",
  markers: "Read [MASK] and [UNK] as ordinary text."
};
const visible = text => text.replace(/[ \t\n\r\x00-\x1f\x7f]/g, c =>
  ({" ": "␠", "\t": "⇥", "\n": "↵", "\r": "␍"}[c] ?? "\\x" + c.charCodeAt(0).toString(16).padStart(2, "0")));
const hex = bytes => Array.from(bytes, n => n.toString(16).padStart(2, "0")).join(" ");
const labels = {bpe: "Byte BPE", wordpiece: "WordPiece", pretrained_reference: "cl100k_base"};
const controls = {
  encode: {button: "encode", status: "encode-status", cancel: "cancel-encode"},
  train: {button: "train-button", status: "train-status", cancel: "cancel-train"}
};
let output = null;
let sourceBytes = null;
let summary = null;
let pendingResults = null;
let view;

function measureInput(field, counter, limit) {
  const value = el(field).value;
  const length = [...value].length;
  set(counter, `${format(length)} / ${format(limit)} scalars · ${format(encoder.encode(value).length)} UTF-8 bytes`);
  el(counter).classList.toggle("error", length > limit);
  return length <= limit;
}

function estimate() {
  const context = el("context").valueAsNumber, rate = el("rate").valueAsNumber;
  if (!output) return set("estimate", "Encode text to calculate an estimate.");
  if (!Number.isSafeInteger(context) || context <= 0 || !Number.isFinite(rate) || rate < 0) {
    return set("estimate", "Enter a positive whole context window and a nonnegative input rate.");
  }
  const cost = (output.count / 1_000_000) * rate;
  set("estimate", `${format(output.count)} tokens · ${(output.count / context * 100).toFixed(2)}% of context · ${cost.toLocaleString("en-US", {style: "currency", currency: "USD", maximumFractionDigits: 8})} estimated input cost`);
}

function clearEncoding() {
  output = sourceBytes = null;
  el("detail").hidden = true;
  el("ribbon").replaceChildren();
  for (const id of ["shown", "token-stat", "decoded", "decode-match"]) set(id, "");
  const bpe = el("algorithm").value === "bpe";
  set("decode-note", bpe ? "Byte BPE concatenates token bytes before decoding. It preserves the original text."
    : "WordPiece joins pretokens with one space. A whole pretoken outside its vocabulary becomes [UNK].");
  estimate();
}

function selectToken(index, focus = false) {
  const ribbon = el("ribbon"), token = output.tokens[index];
  for (const [i, chip] of Array.from(ribbon.children).entries()) {
    chip.setAttribute("aria-pressed", String(index === i));
    chip.tabIndex = index === i ? 0 : -1;
  }
  if (focus) ribbon.children[index].focus();
  el("detail").hidden = false;
  set("selected-piece", visible(token.piece));
  set("selected-id", String(token.id));
  set("selected-span", `[${token.start}, ${token.end})`);
  const bytes = sourceBytes.subarray(token.start, token.end);
  const preview = bytes.subarray(0, 48);
  const omitted = bytes.length > preview.length ? ` … (first 48 of ${format(bytes.length)} bytes)` : "";
  set("selected-hex", hex(preview) + omitted);
  let note = "";
  if (token.unknown) note = "One [UNK] replaces this entire pretoken.";
  else if (token.bytes_hex !== null) {
    try { new TextDecoder("utf-8", {fatal: true}).decode(bytes); }
    catch { note = "This piece contains part of a UTF-8 character. Its bytes join adjacent pieces during decoding."; }
  }
  set("selected-note", note);
}

function renderEncoding(value) {
  output = value;
  const source = el("source").value;
  sourceBytes = encoder.encode(source);
  set("token-stat", `${format(value.count)} token${value.count === 1 ? "" : "s"}`);
  const ribbon = el("ribbon");
  ribbon.replaceChildren();
  value.tokens.forEach((token, i) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "piece";
    chip.dataset.tone = String(i % 5);
    if (token.unknown) chip.dataset.unknown = "";
    const piece = document.createElement("b"), id = document.createElement("small");
    piece.textContent = visible(token.piece);
    piece.dir = "auto";
    id.textContent = String(token.id);
    chip.setAttribute("aria-label", `Piece ${i + 1}, ${piece.textContent}, ID ${token.id}, bytes ${token.start} to ${token.end}`);
    chip.append(piece, id);
    chip.onclick = () => selectToken(i);
    chip.onkeydown = event => {
      const to = {ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: value.tokens.length - 1}[event.key];
      if (to !== undefined) {
        event.preventDefault();
        selectToken(Math.max(0, Math.min(value.tokens.length - 1, to)), true);
      }
    };
    ribbon.append(chip);
  });
  if (value.tokens.length) selectToken(0);
  set("shown", value.count > value.tokens.length ? `Showing the first ${value.tokens.length} of ${format(value.count)} tokens. The full count and decode include every token.`
    : (value.count === 0 ? "This input produced no tokens." : ""));
  set("decoded", value.decoded);
  set("decode-match", value.decoded === source ? "Exact match" : "Canonical output");
  if (el("algorithm").value === "wordpiece") {
    set("decode-note", `WordPiece joins pretokens with one space. ${format(value.unknown)} whole pretokens became [UNK].`);
  }
  estimate();
}

function clearTraining() {
  set("trace-summary", "Train the corpus to see the selected pairs.");
  el("trace-wrap").hidden = true;
  el("trace").replaceChildren();
  const rule = el("train-algorithm").value === "bpe" ? "Byte BPE selects the most frequent pair."
    : "Educational WordPiece scores each pair by its count divided by the product of its symbol counts.";
  set("trace-method", rule + " Up to 12 merges, with a minimum pair count of two.");
}

function row(parent, cells) {
  const tr = document.createElement("tr");
  for (const value of cells) {
    const td = document.createElement("td");
    if (value instanceof Node) td.append(value);
    else td.textContent = String(value);
    tr.append(td);
  }
  parent.append(tr);
}

function renderTraining(value) {
  set("trace-summary", `${value.trace.length} of 12 possible merges · ${value.base} base entries · ${value.size} final entries.` +
    (value.trace.length < 12 ? " No eligible pair remains." : ""));
  const body = el("trace");
  body.replaceChildren();
  for (const step of value.trace) {
    const pair = document.createElement("code"), result = document.createElement("code");
    pair.textContent = step.pair.map(visible).join(" + ");
    result.textContent = visible(step.piece);
    row(body, [step.rank + 1, pair, result, step.count, step.score_den === 1 ? step.score_num : `${step.score_num}/${step.score_den}`]);
  }
  el("trace-wrap").hidden = value.trace.length === 0;
}

function updateButtons() {
  for (const kind of ["encode", "train"]) {
    const encoding = kind === "encode";
    const valid = encoding ? measureInput("source", "source-count", 12000) : measureInput("corpus", "corpus-count", 1000);
    const busy = controller.active?.kind === kind;
    el(controls[kind].button).disabled = busy || !valid;
    el(controls[kind].cancel).hidden = !busy;
  }
}

function notify(event) {
  const status = controls[event.kind].status;
  el(status).classList.toggle("error", event.type === "error");
  if (event.type === "start") set(status, "Starting worker…");
  if (event.type === "progress") set(status, event.message);
  if (event.type === "cancel") set(status, "Cancelled. Run again when ready.");
  if (event.type === "error") {
    set(status, `Worker error. Check your connection to the Pyodide CDN, then retry. ${event.message}`);
  }
  if (event.type === "result") {
    if (event.kind === "encode") renderEncoding(event.value);
    else renderTraining(event.value);
    set(status, event.kind === "encode" ? "Encoded locally in Python." : "Trained locally in Python.");
  }
  updateButtons();
}

const controller = new Controller(() => new Worker(new URL("./worker.mjs", import.meta.url), {type: "module"}), notify);

function start(kind) {
  const request = kind === "encode"
    ? {kind, key: `${el("algorithm").value}-${el("target").value}`, text: el("source").value}
    : {kind, algorithm: el("train-algorithm").value, text: el("corpus").value};
  try { validateRequest({...request, id: 1}); }
  catch (error) {
    set(controls[kind].status, error.message);
    el(controls[kind].status).classList.add("error");
    return;
  }
  if (kind === "encode") clearEncoding(); else clearTraining();
  controller.start(request);
}

function edit(kind) {
  if (controller.active?.kind === kind) controller.cancel();
  if (kind === "encode") clearEncoding(); else clearTraining();
  set(controls[kind].status, kind === "encode" ? "Encode the edited text to update the ribbon." : "Train the edited corpus to update the trace.");
  el(controls[kind].status).classList.remove("error");
  updateButtons();
}

el("source").oninput = () => { el("sample").value = "custom"; edit("encode"); };
el("corpus").oninput = el("train-algorithm").onchange = () => edit("train");
el("algorithm").onchange = el("target").onchange = () => { edit("encode"); start("encode"); };
el("sample").onchange = () => {
  el("source").value = examples[el("sample").value];
  edit("encode");
  start("encode");
};
el("encode").onclick = () => start("encode");
el("train-button").onclick = () => start("train");
el("cancel-encode").onclick = el("cancel-train").onclick = () => controller.cancel();
el("context").oninput = el("rate").oninput = estimate;

function renderResults() {
  if (!summary) return;
  const domain = el("domain").value, data = summary.datasets[domain];
  set("domain-size", `${format(data.rows)} ${domain === "cpython" ? "files" : "rows"} · ${format(data.bytes)} UTF-8 bytes`);
  set("results-note", `Training used ${format(summary.training_bytes)} UTF-8 bytes after exact test-row exclusion.`);
  el("metrics").replaceChildren(); el("timings").replaceChildren();
  for (const model of [...summary.models, summary.reference]) {
    const m = model.datasets[domain], algorithm = labels[model.algorithm];
    const pct = x => (100 * x).toFixed(2) + "%";
    row(el("metrics"), [algorithm, format(model.vocab_size), format(m.tokens), m.bytes_per_token.toFixed(3),
      m.word_coverage === null ? "All bytes" : pct(m.word_coverage),
      m.unknown_word_rate === null ? "—" : pct(m.unknown_word_rate),
      m.known_bytes_per_known_token?.toFixed(3) ?? "—"]);
    if (model === summary.reference) continue;
    const t = model.timing[domain];
    row(el("timings"), [algorithm, format(model.vocab_size), format(Math.round(t.tokenforge.median_bytes_per_second)),
      format(Math.round(t.huggingface.median_bytes_per_second)), `${format(m.parity_rows)} / ${format(m.rows)}`]);
  }
}

function loadResults() {
  if (!pendingResults) {
    pendingResults = fetchResults().finally(() => { pendingResults = null; });
  }
  return pendingResults;
}

async function fetchResults() {
  set("results-note", "Loading recorded measurements…");
  el("retry-results").hidden = true;
  try {
    const manifest = await loadManifest();
    summary = JSON.parse(new TextDecoder().decode(await readVerified(manifest.summary)));
    renderResults();
  } catch (error) {
    summary = null;
    set("results-note", `Could not load verified measurements. ${error.message}`);
    el("retry-results").hidden = false;
  }
}

function navigate() {
  const next = ["inspect", "train", "results"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "inspect";
  if (next === view) return;
  if (controller.active) controller.cancel();
  const initial = view === undefined;
  view = next;
  for (const section of document.querySelectorAll(".view")) section.hidden = section.id !== view;
  for (const link of document.querySelectorAll(".mast nav a")) {
    if (link.hash === "#" + view) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  }
  if (!initial) {
    el("main").focus({preventScroll: true});
    window.scrollTo(0, 0);
  }
  if (view === "inspect" && !output && !el("encode-status").textContent) start("encode");
  if (view === "results" && !summary) loadResults();
}

el("retry-results").onclick = loadResults;
el("domain").onchange = renderResults;
document.querySelector(".skip").onclick = event => {
  event.preventDefault();
  el("main").focus();
};
window.addEventListener("hashchange", navigate);
clearEncoding();
clearTraining();
updateButtons();
navigate();
