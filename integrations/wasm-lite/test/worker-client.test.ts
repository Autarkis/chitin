import { beforeEach, describe, expect, it } from "vitest";

import type { ConvexHull } from "../src/types.js";
import { DecomposeWorker, type WorkerLike } from "../src/worker-client.js";
import type { WorkerRequest, WorkerResponse } from "../src/worker-protocol.js";

// A programmable stand-in for a Web Worker: it records what the client posts and
// lets the test drive responses back through the client's onmessage handler.
class FakeWorker implements WorkerLike {
  onmessage: ((ev: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  posted: WorkerRequest[] = [];
  terminated = false;

  // Set to make the next decompose postMessage throw synchronously, the way a
  // real Worker does on a detached transfer buffer or an uncloneable config.
  throwOnDecompose: Error | null = null;

  postMessage(message: WorkerRequest): void {
    if (message.type === "decompose" && this.throwOnDecompose) {
      const err = this.throwOnDecompose;
      this.throwOnDecompose = null;
      throw err;
    }
    this.posted.push(message);
  }
  terminate(): void {
    this.terminated = true;
  }

  emit(msg: WorkerResponse): void {
    this.onmessage?.({ data: msg } as MessageEvent<WorkerResponse>);
  }
  emitError(): void {
    this.onerror?.(new Error("boom"));
  }
  lastDecomposeId(): number {
    for (let i = this.posted.length - 1; i >= 0; i--) {
      const m = this.posted[i];
      if (m.type === "decompose") return m.id;
    }
    throw new Error("no decompose message posted");
  }
}

const V = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const F = new Int32Array([0, 1, 2]);
const HULLS: ConvexHull[] = [
  { vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), indices: new Uint32Array([0, 1, 2]) },
];

let fakes: FakeWorker[];
function makeWorker(instanceOnState?: (s: string) => void) {
  fakes = [];
  const factory = () => {
    const f = new FakeWorker();
    fakes.push(f);
    return f;
  };
  return new DecomposeWorker(
    { js: "coacd.mjs", wasm: "coacd.wasm" },
    { workerFactory: factory, onState: instanceOnState },
  );
}

beforeEach(() => {
  fakes = [];
});

describe("DecomposeWorker", () => {
  it("sends init with the wasm URLs before the first decompose", () => {
    const w = makeWorker();
    void w.decompose(V, F).catch(() => {});
    const init = fakes[0].posted[0];
    expect(init).toEqual({ type: "init", wasmJsUrl: "coacd.mjs", wasmBinaryUrl: "coacd.wasm" });
    w.terminate();
  });

  it("resolves with hulls and reports state transitions in order", async () => {
    const states: string[] = [];
    const w = makeWorker();
    const p = w.decompose(V, F, {}, { onState: (s) => states.push(s) });
    const f = fakes[0];
    const id = f.lastDecomposeId();
    f.emit({ type: "state", id, state: "loading-wasm" });
    f.emit({ type: "state", id, state: "decomposing" });
    f.emit({ type: "state", id, state: "done" });
    f.emit({ type: "result", id, hulls: HULLS });
    const res = await p;
    expect(res.hulls).toBe(HULLS);
    expect(states).toEqual(["loading-wasm", "decomposing", "done"]);
  });

  it("reuses the same worker (no re-init) for a second sequential call", async () => {
    const w = makeWorker();
    const p1 = w.decompose(V, F);
    const f = fakes[0];
    f.emit({ type: "result", id: f.lastDecomposeId(), hulls: HULLS });
    await p1;

    const p2 = w.decompose(V, F);
    expect(fakes).toHaveLength(1); // no new worker spawned
    expect(f.posted.filter((m) => m.type === "init")).toHaveLength(1);
    f.emit({ type: "result", id: f.lastDecomposeId(), hulls: HULLS });
    await p2;
  });

  it("rejects a concurrent call while one is in progress", async () => {
    const w = makeWorker();
    const p1 = w.decompose(V, F);
    await expect(w.decompose(V, F)).rejects.toThrow(/already in progress/);
    const f = fakes[0];
    f.emit({ type: "result", id: f.lastDecomposeId(), hulls: HULLS });
    await p1;
  });

  it("aborts by terminating the worker and rejecting with CANCELLED", async () => {
    const w = makeWorker();
    const ac = new AbortController();
    const p = w.decompose(V, F, {}, { signal: ac.signal });
    const f = fakes[0];
    ac.abort();
    await expect(p).rejects.toMatchObject({ code: "CANCELLED" });
    expect(f.terminated).toBe(true);
  });

  it("spawns a fresh worker for the call after an abort", async () => {
    const w = makeWorker();
    const ac = new AbortController();
    const p = w.decompose(V, F, {}, { signal: ac.signal });
    ac.abort();
    await expect(p).rejects.toMatchObject({ code: "CANCELLED" });

    const p2 = w.decompose(V, F);
    expect(fakes).toHaveLength(2);
    fakes[1].emit({ type: "result", id: fakes[1].lastDecomposeId(), hulls: HULLS });
    await p2;
  });

  it("rejects immediately if the signal is already aborted", async () => {
    const w = makeWorker();
    const ac = new AbortController();
    ac.abort();
    await expect(w.decompose(V, F, {}, { signal: ac.signal })).rejects.toMatchObject({
      code: "CANCELLED",
    });
    expect(fakes).toHaveLength(0); // no worker spawned
  });

  it("rejects with the worker's mapped error code", async () => {
    const w = makeWorker();
    const p = w.decompose(V, F);
    const f = fakes[0];
    f.emit({
      type: "error",
      id: f.lastDecomposeId(),
      code: "OUT_OF_MEMORY",
      message: "Cannot enlarge memory arrays",
      stage: null,
      suggestion: null,
      retryable: false,
      context: {},
    });
    await expect(p).rejects.toMatchObject({ code: "OUT_OF_MEMORY" });
    expect(f.terminated).toBe(true);

    const retry = w.decompose(V, F);
    expect(fakes).toHaveLength(2);
    fakes[1].emit({ type: "result", id: fakes[1].lastDecomposeId(), hulls: HULLS });
    await expect(retry).resolves.toMatchObject({ hulls: HULLS });
  });

  it("preserves structured topology context from the worker", async () => {
    const w = makeWorker();
    const pending = w.decompose(V, F);
    const f = fakes[0];
    f.emit({
      type: "error",
      id: f.lastDecomposeId(),
      code: "NON_MANIFOLD",
      message: "mesh has open edges",
      stage: null,
      suggestion: "repair it",
      retryable: false,
      context: { boundary_edges: 3 },
    });
    await expect(pending).rejects.toMatchObject({
      code: "NON_MANIFOLD",
      suggestion: "repair it",
      retryable: false,
      context: { boundary_edges: 3 },
    });
  });

  it("rejects with WORKER_ERROR when the worker faults", async () => {
    const w = makeWorker();
    const p = w.decompose(V, F);
    fakes[0].emitError();
    await expect(p).rejects.toMatchObject({ code: "WORKER_ERROR" });
    expect(fakes[0].terminated).toBe(true);
  });

  it("passes checkManifold through to the worker request", () => {
    const w = makeWorker();
    void w.decompose(V, F, {}, { checkManifold: true }).catch(() => {});
    const msg = fakes[0].posted.find((m) => m.type === "decompose");
    expect(msg).toMatchObject({ checkManifold: true });
    w.terminate();
  });

  it("stays usable after postMessage throws synchronously", async () => {
    // The pending slot used to survive the throw, so the client rejected every
    // later call as "already in progress" -- permanently wedged.
    const w = makeWorker();
    const p1 = w.decompose(V, F);
    const f = fakes[0];
    f.emit({ type: "result", id: f.lastDecomposeId(), hulls: HULLS });
    await p1;

    f.throwOnDecompose = new Error("DataCloneError");
    await expect(w.decompose(V, F)).rejects.toThrow(/DataCloneError/);

    const p3 = w.decompose(V, F);
    f.emit({ type: "result", id: f.lastDecomposeId(), hulls: HULLS });
    await expect(p3).resolves.toMatchObject({ hulls: HULLS });
  });

  it("releases the abort listener when postMessage throws", async () => {
    const w = makeWorker();
    const ac = new AbortController();
    // First call primes the worker; the second one throws on post.
    const p1 = w.decompose(V, F, {}, { signal: ac.signal });
    fakes[0].emit({ type: "result", id: fakes[0].lastDecomposeId(), hulls: HULLS });
    await p1;

    fakes[0].throwOnDecompose = new Error("DataCloneError");
    await expect(w.decompose(V, F, {}, { signal: ac.signal })).rejects.toThrow(
      /DataCloneError/,
    );
    // A listener left attached would fire into a null pending slot here.
    expect(() => ac.abort()).not.toThrow();
  });

  it("terminate() rejects an in-flight call with CANCELLED", async () => {
    const w = makeWorker();
    const p = w.decompose(V, F);
    w.terminate();
    await expect(p).rejects.toMatchObject({ code: "CANCELLED" });
    expect(fakes[0].terminated).toBe(true);
  });
});
