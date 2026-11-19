import test from "node:test";
import assert from "node:assert/strict";
import {Controller} from "../controller.mjs";

function setup() {
  const workers = [], events = [];
  const controller = new Controller(() => {
    const worker = {
      requests: [],
      terminate() { this.terminated = true; },
      postMessage(message) { this.requests.push(message); },
      send(data) { this.onmessage({data}); }
    };
    workers.push(worker);
    return worker;
  }, event => events.push(event));
  return {controller, workers, events};
}
const encode = {kind: "encode", key: "bpe-4096", text: "hello"};
const train = {kind: "train", algorithm: "wordpiece", text: "hello hello"};

test("cancellation terminates computation and queued old success and errors cannot own a retry", () => {
  const {controller, workers, events} = setup();
  controller.start(encode);
  const old = workers[0], first = old.requests[0].id;
  controller.cancel();
  controller.start(train);
  const current = workers[1], next = current.requests[0].id;
  assert.ok(old.terminated);
  old.send({id: first, type: "result", value: "stale"});
  old.onerror({message: "late crash"});
  current.send({id: first, type: "error", message: "out of order"});
  assert.deepEqual(controller.active, {id: next, kind: "train"});
  current.send({id: next, type: "progress", message: "Training"});
  current.send({id: next, type: "result", value: "fresh"});
  current.send({id: next, type: "result", value: "duplicate"});
  assert.equal(controller.active, null);
  assert.deepEqual(events.map(e => e.type), ["start", "cancel", "start", "progress", "result"]);
  assert.equal(events.at(-1).value, "fresh");
});

test("success reuses the warm worker, while runtime errors release and recreate it", () => {
  const {controller, workers, events} = setup();
  controller.start(encode);
  workers[0].send({id: 1, type: "result", value: {}});
  controller.start(encode);
  assert.equal(workers.length, 1);
  workers[0].send({id: 1, type: "result", value: "old"});
  workers[0].send({id: 2, type: "error", message: "hash mismatch"});
  assert.equal(events.at(-1).message, "hash mismatch");
  assert.equal(controller.active, null);
  assert.ok(workers[0].terminated);
  controller.start(train);
  workers[1].onmessageerror();
  assert.equal(events.at(-1).kind, "train");
  assert.ok(workers[1].terminated);
});

test("a blocked worker constructor restores idle state and surfaces failure", () => {
  const events = [];
  const controller = new Controller(() => { throw Error("unavailable"); }, e => events.push(e));
  controller.start(encode);
  assert.equal(controller.active, null);
  assert.equal(events.at(-1).message, "unavailable");
});
