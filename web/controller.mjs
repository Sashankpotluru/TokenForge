// Only one job owns the UI. A terminated worker can still have a queued message.
export class Controller {
  constructor(makeWorker, notify) {
    this.makeWorker = makeWorker;
    this.notify = notify;
    this.sequence = 0;
    this.worker = null;
    this.active = null;
  }

  start(request) {
    if (this.active) this.cancel();
    const id = ++this.sequence;
    this.active = {id, kind: request.kind};
    this.notify({type: "start", kind: request.kind});
    try {
      const worker = this.worker ??= this.makeWorker();
      worker.onmessage = ({data}) => {
        if (this.worker !== worker || data?.id !== id || this.active?.id !== id) return;
        if (data.type === "progress") this.notify({...data, kind: request.kind});
        if (data.type === "result" || data.type === "error") {
          this.active = null;
          if (data.type === "error") this.discard();
          this.notify({...data, kind: request.kind});
        }
      };
      worker.onerror = event => {
        if (this.worker !== worker || this.active?.id !== id) return;
        event.preventDefault?.();
        this.fail(request.kind, event.message || "Worker could not start.");
      };
      worker.onmessageerror = () => {
        if (this.worker === worker && this.active?.id === id) {
          this.fail(request.kind, "Could not read the worker response.");
        }
      };
      worker.postMessage({...request, id});
    } catch (error) {
      this.fail(request.kind, error.message);
    }
  }

  fail(kind, message) {
    this.active = null;
    this.discard();
    this.notify({type: "error", kind, message});
  }

  discard() {
    this.worker?.terminate();
    this.worker = null;
  }

  cancel() {
    if (!this.active) return;
    const {kind} = this.active;
    this.active = null;
    this.discard();
    this.notify({kind, type: "cancel"});
  }
}
