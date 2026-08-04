import { ChitinError, type ChitinErrorCode } from "./errors.js";
import type { CompilationErrorInfo } from "./report.js";
import type { ConvexHull, DecomposeConfig } from "./types.js";

// The message contract between DecomposeWorker (main thread) and worker.ts
// (inside the Web Worker). Kept in its own module so both sides — and the
// tests' fake worker — share one source of truth.

/** Lifecycle of a single decompose call, reported as it advances. */
export type DecomposeState = "loading-wasm" | "decomposing" | "done";

export interface InitRequest {
  type: "init";
  wasmJsUrl: string;
  wasmBinaryUrl: string;
}

export interface DecomposeRequest {
  type: "decompose";
  id: number;
  vertices: Float64Array;
  faces: Int32Array;
  config: DecomposeConfig;
  checkManifold: boolean;
}

export type WorkerRequest = InitRequest | DecomposeRequest;

export interface StateMessage {
  type: "state";
  id: number;
  state: DecomposeState;
}

export interface ResultMessage {
  type: "result";
  id: number;
  hulls: ConvexHull[];
}

export interface ErrorMessage {
  type: "error";
  id: number;
  code: ChitinErrorCode;
  message: string;
  stage: CompilationErrorInfo["stage"];
  suggestion: string | null;
  retryable: boolean;
  context: CompilationErrorInfo["context"];
}

export type WorkerResponse = StateMessage | ResultMessage | ErrorMessage;

// Emscripten reports a heap that cannot grow in a few different phrasings
// depending on build flags; match the ones CoACD's module can surface.
const OOM_PATTERN = /cannot enlarge memory|out of memory|\boom\b|bad_alloc|allocation failed/i;

type WorkerErrorInfo = Omit<CompilationErrorInfo, "code"> & { code: ChitinErrorCode };

/**
 * Map an arbitrary error thrown inside the worker to a structured
 * error payload. A {@link ChitinError} keeps all of its structured context; a
 * heap-exhaustion message becomes `OUT_OF_MEMORY`; anything else is a
 * `WORKER_ERROR`.
 */
export function mapWorkerError(err: unknown): WorkerErrorInfo {
  if (err instanceof ChitinError) {
    return { ...err.toInfo(), code: err.code };
  }
  const message = err instanceof Error ? err.message : String(err);
  if (OOM_PATTERN.test(message)) {
    return { code: "OUT_OF_MEMORY", message, stage: null, suggestion: null, retryable: false, context: {} };
  }
  return { code: "WORKER_ERROR", message, stage: null, suggestion: null, retryable: false, context: {} };
}
