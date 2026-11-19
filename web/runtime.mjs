const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";

export function validateRequest(request) {
  if (!Number.isSafeInteger(request?.id) || request.id < 1) throw Error("Invalid request ID");
  const encoding = request.kind === "encode";
  if (!encoding && request.kind !== "train") throw Error("Unknown operation");
  if (encoding && !/^(bpe|wordpiece)-(1024|2048|4096)$/.test(request.key)) {
    throw Error("Select a published vocabulary");
  }
  if (!encoding && !["bpe", "wordpiece"].includes(request.algorithm)) {
    throw Error("Select a training algorithm");
  }
  const limit = encoding ? 12000 : 1000;
  if (typeof request.text !== "string" || [...request.text].length > limit) {
    throw Error(`Use at most ${limit.toLocaleString("en-US")} Unicode scalars.`);
  }
  if (/[\uD800-\uDFFF]/u.test(request.text)) throw Error("Input contains a lone surrogate.");
}

export async function readVerified(entry, fetcher = fetch) {
  if (!entry || !/^[a-z0-9][a-z0-9./-]*$/.test(entry.path) ||
      entry.path.split("/").includes("..") || !/^[a-f0-9]{64}$/.test(entry.sha256) ||
      !Number.isSafeInteger(entry.bytes) || entry.bytes < 0) throw Error("Invalid asset manifest.");
  const response = await fetcher(new URL(entry.path, import.meta.url));
  if (!response.ok) throw Error(`Could not load ${entry.path} (HTTP ${response.status}).`);
  const data = new Uint8Array(await response.arrayBuffer());
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  const hash = Array.from(digest, n => n.toString(16).padStart(2, "0")).join("");
  if (data.length !== entry.bytes || hash !== entry.sha256) {
    throw Error(`Integrity check failed for ${entry.path}. Reload the page and retry.`);
  }
  return data;
}

export async function loadManifest() {
  const response = await fetch(new URL("./manifest.json", import.meta.url), {cache: "no-cache"});
  if (!response.ok) throw Error(`Could not load manifest (HTTP ${response.status}).`);
  const manifest = await response.json();
  if (manifest.version !== 1) throw Error("Unknown manifest version. Reload the page.");
  return manifest;
}

export async function startPython(manifest) {
  const [{loadPyodide}, core] = await Promise.all([
    import(PYODIDE + "pyodide.mjs"), readVerified(manifest.core)
  ]);
  const python = await loadPyodide({indexURL: PYODIDE});
  python.FS.writeFile("/tmp/tokenforge-core.zip", core);
  python.runPython('import sys\nsys.path.insert(0, "/tmp/tokenforge-core.zip")\nimport workbench_bridge');
  const loaded = new Set();
  return async function run(request, progress) {
    if (request.kind === "encode" && !loaded.has(request.key)) {
      progress("Checking the trained model…");
      const model = await readVerified(manifest.models[request.key]);
      python.globals.set("model_key", request.key);
      python.globals.set("model_json", new TextDecoder().decode(model));
      python.runPython("workbench_bridge.register_model(model_key, model_json)\ndel model_json");
      loaded.add(request.key);
    }
    progress(request.kind === "encode" ? "Encoding text…" : "Training vocabulary…");
    python.globals.set("request_json", JSON.stringify(request));
    return JSON.parse(python.runPython("workbench_bridge.handle(request_json)"));
  };
}

export function createHandler(start = async () => startPython(await loadManifest())) {
  let runtime;
  return async function handle(request, send) {
    const progress = message => send({id: request.id, type: "progress", message});
    try {
      validateRequest(request);
      if (!runtime) {
        progress("Loading Python and checking the core…");
        runtime = await start();
      }
      const value = await runtime(request, progress);
      send({id: request.id, type: "result", value});
    } catch (error) {
      send({id: request?.id, type: "error", message: String(error.message || error)});
    }
  };
}
