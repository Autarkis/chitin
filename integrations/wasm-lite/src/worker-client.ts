import { ChitinError } from "./errors.js";
import type { TriangleMesh } from "./mesh.js";
import type { DecomposeConfig, DecomposeResult } from "./types.js";
import type {
  DecomposeState,
  PoissonState,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.js";

/** The subset of the Web Worker interface ChitinWorkerClient drives. A fake
 * implementing this can be injected via {@link ChitinWorkerClientOptions.workerFactory}
 * for testing without a real Worker. */
export interface WorkerLike {
  postMessage(message: WorkerRequest, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((ev: MessageEvent<WorkerResponse>) => void) | null;
  onerror: ((ev: ErrorEvent) => void) | null;
}

export interface ChitinWorkerClientOptions {
  /** Build the underlying worker. Overrides `workerUrl`; used by tests. */
  workerFactory?: () => WorkerLike;
  /** URL of the built `worker.js` module. Needed when no bundler rewrites
   * `new URL("./worker.js", import.meta.url)` — e.g. loading from a CDN, pass
   * the package's `dist/worker.js` URL. */
  workerUrl?: string | URL;
  /** Called on every lifecycle transition of every call. */
  onState?: (state: DecomposeState | PoissonState) => void;
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

interface PendingCall<TResult, TState extends string> {
  id: number;
  resolve: (r: TResult) => void;
  reject: (e: Error) => void;
  onState?: (state: TState) => void;
  cleanup: () => void;
}

type Pending =
  | ({ kind: "decompose" } & PendingCall<DecomposeResult, DecomposeState>)
  | ({ kind: "poisson" } & PendingCall<{ vertices: Float64Array; faces: Int32Array }, PoissonState>);

/**
 * Runs `decompose()` off the main thread. CoACD is synchronous inside the
 * worker, so the UI stays responsive and a call can be cancelled by terminating
 * the worker. One call runs at a time; the worker (and its loaded wasm) is
 * reused across calls until terminated or cancelled.
 */
export class ChitinWorkerClient {
  private worker: WorkerLike | null = null;
  private pending: Pending | null = null;
  private nextId = 1;

  constructor(
    private readonly wasmUrls: { js: string; wasm: string },
    private readonly opts: ChitinWorkerClientOptions = {},
    private readonly poissonUrls?: { js: string; wasm: string },
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
    if (this.poissonUrls) {
      worker.postMessage({
        type: "init-poisson",
        wasmJsUrl: this.poissonUrls.js,
        wasmBinaryUrl: this.poissonUrls.wasm,
      });
    }
    return worker;
  }

  private onMessage(msg: WorkerResponse): void {
    const p = this.pending;
    if (!p || msg.id !== p.id) return;
    if (msg.type === "state") {
      this.opts.onState?.(msg.state);
      // Safe: state values are disjoint between decompose and poisson
      (p.onState as ((s: string) => void) | undefined)?.(msg.state);
      return;
    }
    if (msg.type === "result" && p.kind === "decompose") {
      p.cleanup();
      this.pending = null;
      p.resolve({ hulls: msg.hulls });
      return;
    }
    if (msg.type === "poisson-result" && p.kind === "poisson") {
      p.cleanup();
      this.pending = null;
      p.resolve({ vertices: msg.vertices, faces: msg.faces });
      return;
    }
    if (msg.type !== "error") return;
    p.cleanup();
    this.pending = null;
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

      this.pending = { kind: "decompose", id, resolve, reject, onState: callOpts.onState, cleanup };

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

  poissonReconstruct(
    positions: Float64Array,
    normals: Float64Array,
    options: {
      depth: number;
      densityQuantile: number;
      signal?: AbortSignal;
      onState?: (state: PoissonState) => void;
    },
  ): Promise<{ vertices: Float64Array; faces: Int32Array }> {
    return new Promise((resolve, reject) => {
      if (this.pending) {
        reject(new ChitinError("WORKER_ERROR", "decompose worker already in progress"));
        return;
      }
      if (!this.worker) this.worker = this.spawn();
      const worker = this.worker;
      const id = this.nextId++;
      const cleanup = (): void => {
        signal?.removeEventListener("abort", onAbort);
      };
      this.pending = {
        kind: "poisson",
        id,
        resolve,
        reject,
        cleanup,
        onState: options.onState,
      };
      const { signal } = options;
      const onAbort = (): void => {
        worker.terminate();
        this.worker = null;
        this.pending = null;
        cleanup();
        reject(new ChitinError("CANCELLED", "poisson reconstruction cancelled"));
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      try {
        const transfer = [positions.buffer, normals.buffer];
        worker.postMessage(
          {
            type: "poisson",
            id,
            positions,
            normals,
            depth: options.depth,
            densityQuantile: options.densityQuantile,
          },
          transfer,
        );
      } catch (err) {
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
