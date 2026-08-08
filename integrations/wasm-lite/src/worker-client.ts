import { ChitinError } from "./errors.js";
import type { DecomposeConfig, DecomposeResult } from "./types.js";
import type {
  DecomposeState,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.js";

/** The subset of the Web Worker interface DecomposeWorker drives. A fake
 * implementing this can be injected via {@link DecomposeWorkerOptions.workerFactory}
 * for testing without a real Worker. */
export interface WorkerLike {
  postMessage(message: WorkerRequest, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((ev: MessageEvent<WorkerResponse>) => void) | null;
  onerror: ((ev: ErrorEvent) => void) | null;
}

export interface DecomposeWorkerOptions {
  /** Build the underlying worker. Overrides `workerUrl`; used by tests. */
  workerFactory?: () => WorkerLike;
  /** URL of the built `worker.js` module. Needed when no bundler rewrites
   * `new URL("./worker.js", import.meta.url)` — e.g. loading from a CDN, pass
   * the package's `dist/worker.js` URL. */
  workerUrl?: string | URL;
  /** Called on every lifecycle transition of every call. */
  onState?: (state: DecomposeState) => void;
}

export interface DecomposeCallOptions {
  /** Abort the call: the worker is terminated and the promise rejects with
   * code `CANCELLED`. A fresh worker is spawned for the next call. */
  signal?: AbortSignal;
  /** Run the manifold precheck before CoACD; a non-manifold mesh rejects with
   * code `NON_MANIFOLD` instead of producing bad hulls. Default false. */
  checkManifold?: boolean;
  /** Called on this call's lifecycle transitions (in addition to the
   * instance-level `onState`). */
  onState?: (state: DecomposeState) => void;
  /** Transfer the input buffers to the worker (zero-copy) instead of cloning
   * them. Default true, which detaches the caller's `vertices`/`faces`. Set
   * false to keep them usable after the call. */
  transferInput?: boolean;
}

interface Pending {
  id: number;
  resolve: (r: DecomposeResult) => void;
  reject: (e: Error) => void;
  onState?: (state: DecomposeState) => void;
  cleanup: () => void;
}

/**
 * Runs `decompose()` off the main thread. CoACD is synchronous inside the
 * worker, so the UI stays responsive and a call can be cancelled by terminating
 * the worker. One call runs at a time; the worker (and its loaded wasm) is
 * reused across calls until terminated or cancelled.
 */
export class DecomposeWorker {
  private worker: WorkerLike | null = null;
  private pending: Pending | null = null;
  private nextId = 1;

  constructor(
    private readonly wasmUrls: { js: string; wasm: string },
    private readonly opts: DecomposeWorkerOptions = {},
  ) {}

  private spawn(): WorkerLike {
    let worker: WorkerLike;
    if (this.opts.workerFactory) {
      worker = this.opts.workerFactory();
    } else {
      const url = this.opts.workerUrl ?? new URL("./worker.js", import.meta.url);
      worker = new Worker(url, { type: "module" }) as unknown as WorkerLike;
    }
    worker.onmessage = (ev: MessageEvent<WorkerResponse>) => this.onMessage(ev.data);
    worker.onerror = () => this.onWorkerError();
    worker.postMessage({
      type: "init",
      wasmJsUrl: this.wasmUrls.js,
      wasmBinaryUrl: this.wasmUrls.wasm,
    });
    return worker;
  }

  private onMessage(msg: WorkerResponse): void {
    const p = this.pending;
    if (!p || msg.id !== p.id) return; // stale message from a terminated worker
    if (msg.type === "state") {
      this.opts.onState?.(msg.state);
      p.onState?.(msg.state);
      return;
    }
    if (msg.type === "result") {
      p.cleanup();
      this.pending = null;
      p.resolve({ hulls: msg.hulls });
      return;
    }
    // error
    p.cleanup();
    this.pending = null;
    // Emscripten aborts leave the module instance unusable. Discard the worker
    // so a retry gets a newly initialized WASM runtime instead of failing from
    // the poisoned instance immediately.
    if (msg.code === "WORKER_ERROR" || msg.code === "OUT_OF_MEMORY") {
      this.worker?.terminate();
      this.worker = null;
    }
    p.reject(new ChitinError(msg.code, msg.message, {
      stage: msg.stage,
      suggestion: msg.suggestion,
      retryable: msg.retryable,
      context: msg.context,
    }));
  }

  private onWorkerError(): void {
    const p = this.pending;
    // A worker-level error (e.g. the module failed to load) is unrecoverable;
    // drop the worker so the next call spawns a fresh one.
    this.worker?.terminate();
    this.worker = null;
    if (!p) return;
    p.cleanup();
    this.pending = null;
    p.reject(new ChitinError("WORKER_ERROR", "worker failed to load or crashed"));
  }

  /**
   * Decompose a mesh off the main thread. By default the input buffers are
   * transferred to the worker and detached on this side (see
   * {@link DecomposeCallOptions.transferInput}).
   */
  decompose(
    vertices: Float64Array,
    faces: Int32Array,
    config: DecomposeConfig = {},
    callOpts: DecomposeCallOptions = {},
  ): Promise<DecomposeResult> {
    if (this.pending) {
      return Promise.reject(
        new Error("a decompose call is already in progress; await it or use a second worker"),
      );
    }
    const { signal } = callOpts;
    if (signal?.aborted) {
      return Promise.reject(new ChitinError("CANCELLED", "aborted before start"));
    }

    if (!this.worker) this.worker = this.spawn();
    const worker = this.worker;
    const id = this.nextId++;

    return new Promise<DecomposeResult>((resolve, reject) => {
      let onAbort: (() => void) | undefined;
      const cleanup = () => {
        if (onAbort && signal) signal.removeEventListener("abort", onAbort);
      };
      if (signal) {
        onAbort = () => {
          // Can't preempt the synchronous CoACD call; terminate the worker.
          worker.terminate();
          if (this.worker === worker) this.worker = null;
          const p = this.pending;
          this.pending = null;
          cleanup();
          p?.reject(new ChitinError("CANCELLED", "decompose aborted by caller"));
        };
        signal.addEventListener("abort", onAbort);
      }

      this.pending = { id, resolve, reject, onState: callOpts.onState, cleanup };

      const transfer =
        callOpts.transferInput === false
          ? []
          : dedupeBuffers([vertices.buffer, faces.buffer]);
      try {
        worker.postMessage(
          {
            type: "decompose",
            id,
            vertices,
            faces,
            config,
            checkManifold: callOpts.checkManifold ?? false,
          },
          transfer,
        );
      } catch (err) {
        // postMessage can throw synchronously: a detached buffer in `transfer`
        // (reusing arrays from a previous transferring call), or a `config`
        // value structured-clone can't copy. Nothing was delivered, so the
        // worker is still good -- but the pending slot has to be released or
        // this client rejects every later call as "already in progress".
        this.pending = null;
        cleanup();
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  /** Terminate the worker and reject any in-flight call with `CANCELLED`. */
  terminate(): void {
    const p = this.pending;
    this.pending = null;
    this.worker?.terminate();
    this.worker = null;
    if (p) {
      p.cleanup();
      p.reject(new ChitinError("CANCELLED", "worker terminated"));
    }
  }
}

// Transferring the same ArrayBuffer twice throws; a mesh whose vertices and
// faces happen to share a buffer must list it once.
function dedupeBuffers(buffers: ArrayBufferLike[]): Transferable[] {
  const seen = new Set<ArrayBufferLike>();
  const out: Transferable[] = [];
  for (const b of buffers) {
    if (!seen.has(b)) {
      seen.add(b);
      out.push(b as Transferable);
    }
  }
  return out;
}
