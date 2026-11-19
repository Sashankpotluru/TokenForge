import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFile} from "node:fs/promises";
import test from "node:test";

async function openApp(name, fetcher) {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const skipHref = html.match(/<a class="skip" href="([^"]+)"/)[1];
  const skip = {};
  let focused = skip;
  class FixtureNode {
    children = [];
    append(child) { this.children.push(child); }
    replaceChildren() { this.children.length = 0; }
    focus() { focused = this; }
  }
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([, id]) => [id, Object.assign(new FixtureNode(), {
    id, value: "", textContent: "", hidden: false,
    classList: {toggle() {}, remove() {}, add() {}}
  })]));
  const element = id => elements.get(id);
  element("train-algorithm").value = "bpe";
  element("domain").value = "wikitext";
  element("corpus").value = "tea tea tea";
  element("algorithm").value = "bpe";
  element("target").value = "4096";
  const views = ["inspect", "train", "results"].map(element);
  const window = new EventTarget();
  window.scrollTo = () => {};
  globalThis.window = window;
  globalThis.Node = FixtureNode;
  globalThis.document = {
    createElement: () => new FixtureNode(),
    getElementById: element,
    querySelector: selector => selector === ".skip" ? skip : null,
    querySelectorAll: selector => selector === ".view" ? views : []
  };
  globalThis.location = {hash: "#train"};
  globalThis.fetch = fetcher;
  let worker;
  globalThis.Worker = class {
    constructor() { worker = this; }
    postMessage(request) { this.request = request; }
    terminate() { this.terminated = true; }
  };

  await import(`../app.mjs?fixture=${name}`);
  const navigateTo = hash => {
    location.hash = hash;
    window.dispatchEvent(new Event("hashchange"));
  };
  function skipNavigation() {
    focused = skip;
    const click = new Event("click", {cancelable: true});
    skip.onclick?.(click);
    // Reproduce the default activation of the actual anchor if JS leaves it in place.
    if (!click.defaultPrevented) navigateTo(skipHref);
  }

  return {
    element, navigateTo, skipNavigation,
    get focused() { return focused; },
    get worker() { return worker; }
  };
}

test("skip link focuses main without leaving Train or Results or cancelling a running trainer", async () => {
  const app = await openApp("skip", () => new Promise(() => {}));
  const {element, navigateTo, skipNavigation} = app;
  element("train-button").onclick();
  const worker = app.worker;
  assert.equal(worker.request.kind, "train");
  skipNavigation();
  assert.equal(location.hash, "#train");
  assert.equal(element("train").hidden, false);
  assert.equal(app.focused, element("main"));
  assert.ok(!worker.terminated);
  assert.equal(element("train-button").disabled, true);
  assert.equal(element("cancel-train").hidden, false);

  navigateTo("#results");
  assert.equal(worker.terminated, true); // A real view change still cancels the old job.
  skipNavigation();
  assert.equal(location.hash, "#results");
  assert.equal(element("results").hidden, false);
  assert.equal(app.focused, element("main"));
});

test("Results shares a pending load across navigation and retries a failed load", async () => {
  const path = "results/site-summary.json";
  const data = await readFile(new URL("../../" + path, import.meta.url));
  const summary = JSON.parse(data);
  const descriptor = {path, bytes: data.length, sha256: createHash("sha256").update(data).digest("hex")};
  const {promise: firstRequest, reject} = Promise.withResolvers();
  const requests = [];
  const fetcher = url => {
    requests.push(url.pathname.split("/").pop());
    if (requests.length === 1) return firstRequest;
    if (url.pathname.endsWith("/manifest.json")) return Response.json({version: 1, summary: descriptor});
    return new Response(data);
  };
  const {element, navigateTo} = await openApp("results", fetcher);
  navigateTo("#results");
  navigateTo("#train");
  navigateTo("#results");
  assert.deepEqual(requests, ["manifest.json"]);

  reject(new Error("connection lost"));
  await new Promise(resolve => setImmediate(resolve));
  const retry = element("retry-results");
  assert.match(element("results-note").textContent, /connection lost/);
  assert.equal(retry.hidden, false);
  assert.equal(element("metrics").children.length, 0);

  const retryFinished = retry.onclick();
  navigateTo("#train");
  navigateTo("#results");
  await retryFinished;
  assert.equal(retry.hidden, true);
  assert.match(element("results-note").textContent, /Training used/);
  assert.equal(element("metrics").children.length, summary.models.length + 1);
  assert.equal(element("timings").children.length, summary.models.length);
  navigateTo("#train");
  navigateTo("#results");
  assert.deepEqual(requests, ["manifest.json", "manifest.json", "site-summary.json"]);
});
