import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {createHandler, readVerified, validateRequest} from "../runtime.mjs";

const root = new URL("../../site/", import.meta.url);
const encode = {kind: "encode", key: "bpe-4096", id: 1, text: "👩🏿‍💻"};
const train = {kind: "train", algorithm: "bpe", id: 2, text: "tea tea"};

test("scalar bounds accept complete astral characters, reject oversize and lone surrogates", () => {
  for (const [request, limit] of [[encode, 12000], [train, 1000]]) {
    validateRequest({...request, text: "𐐷".repeat(limit)});
    assert.throws(() => validateRequest({...request, text: "𐐷".repeat(limit + 1)}), /scalars/);
    assert.throws(() => validateRequest({...request, text: "\uD800"}), /surrogate/);
  }
  assert.throws(() => validateRequest({...train, algorithm: "invalid"}));
  assert.throws(() => validateRequest({...encode, key: "../secret"}));
});

test("worker validates before loading, retries startup, and sends data-shaped requests intact", async () => {
  let boots = 0;
  const sent = [];
  const handle = createHandler(async () => {
    boots++;
    if (boots === 1) throw Error("CDN offline");
    return async (req, progress) => { progress("Working"); return req.text; };
  });
  const send = event => sent.push(event);
  await handle({...train, text: "a".repeat(1001)}, send);
  assert.equal(boots, 0);
  assert.equal(sent.at(-1).type, "error");
  await handle(encode, send);
  assert.equal(sent.at(-1).message, "CDN offline");
  const text = "'\n__import__('os') #";
  await handle({...encode, text}, send);
  assert.deepEqual(sent.at(-1), {id: 1, type: "result", value: text});
  await handle(train, send);
  assert.equal(boots, 2);
  assert.deepEqual(sent.at(-1), {id: 2, type: "result", value: train.text});
});

test("release manifest matches actual core, summary and six trained models; tampering fails closed", async () => {
  const manifest = JSON.parse(await readFile(new URL("manifest.json", root)));
  assert.equal(Object.keys(manifest.models).length, 6);
  const get = async entry => readVerified(entry, async () => new Response(await readFile(new URL(entry.path, root))));
  const summary = JSON.parse(new TextDecoder().decode(await get(manifest.summary)));
  for (const model of summary.models) {
    const entry = manifest.models[model.key];
    assert.equal(entry.sha256, model.sha256);
    const data = JSON.parse(new TextDecoder().decode(await get(entry)));
    assert.equal(data.vocab.length, model.vocab_size);
  }
  const core = await get(manifest.core);
  core[0] ^= 1;
  await assert.rejects(readVerified(manifest.core, async () => new Response(core)), /Integrity/);
  await assert.rejects(readVerified({...manifest.core, path: "../outside"}), /Invalid asset manifest/);
});
