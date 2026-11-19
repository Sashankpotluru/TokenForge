import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

test("skip link focuses main without leaving Train or Results or cancelling a running trainer", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const skipHref = html.match(/<a class="skip" href="([^"]+)"/)[1];
  const skip = {};
  let focused = skip;
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([, id]) => [id, {
    id, value: "", textContent: "", hidden: false,
    replaceChildren() {},
    classList: {toggle() {}, remove() {}, add() {}},
    focus() { focused = this; }
  }]));
  const element = id => elements.get(id);
  element("train-algorithm").value = "bpe";
  element("corpus").value = "tea tea tea";
  element("algorithm").value = "bpe";
  element("target").value = "4096";
  const views = ["inspect", "train", "results"].map(element);
  const window = new EventTarget();
  window.scrollTo = () => {};
  globalThis.window = window;
  globalThis.document = {
    getElementById: element,
    querySelector: selector => selector === ".skip" ? skip : null,
    querySelectorAll: selector => selector === ".view" ? views : []
  };
  globalThis.location = {hash: "#train"};
  globalThis.fetch = () => new Promise(() => {});
  let worker;
  globalThis.Worker = class {
    constructor() { worker = this; }
    postMessage(request) { this.request = request; }
    terminate() { this.terminated = true; }
  };

  await import("../app.mjs");
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

  element("train-button").onclick();
  assert.equal(worker.request.kind, "train");
  skipNavigation();
  assert.equal(location.hash, "#train");
  assert.equal(element("train").hidden, false);
  assert.equal(focused, element("main"));
  assert.ok(!worker.terminated);
  assert.equal(element("train-button").disabled, true);
  assert.equal(element("cancel-train").hidden, false);

  navigateTo("#results");
  assert.equal(worker.terminated, true); // A real view change still cancels the old job.
  skipNavigation();
  assert.equal(location.hash, "#results");
  assert.equal(element("results").hidden, false);
  assert.equal(focused, element("main"));
});
