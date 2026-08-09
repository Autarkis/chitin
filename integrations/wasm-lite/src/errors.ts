import type { CompilationErrorInfo, CompilationStage } from "./report.js";

export type ChitinErrorCode =
  | "INVALID_GLB" // malformed GLB container or glTF document
  | "UNSUPPORTED_GLTF" // valid glTF feature that this compiler cannot preserve
  | "LOAD_ERROR" // URL input could not be fetched
  | "COMPILER_BUSY" // a reusable compiler already has an in-flight call
  | "INVALID_MESH" // malformed input geometry (shape, finiteness, index bounds)
  | "INVALID_CONFIG" // a decompose option is out of its valid range
  | "NON_MANIFOLD" // input mesh is open, non-manifold, or degenerate
  | "OUT_OF_MEMORY" // the WASM heap could not grow during decomposition
  | "CANCELLED" // aborted by the caller; the worker was terminated
  | "WORKER_ERROR"; // the worker failed unexpectedly (load or runtime fault)

export interface ChitinErrorOptions {
  stage?: CompilationStage | null;
  suggestion?: string | null;
  retryable?: boolean;
  context?: CompilationErrorInfo["context"];
  cause?: unknown;
}

export class ChitinError extends Error {
  readonly code: ChitinErrorCode;
  readonly stage: CompilationStage | null;
  readonly suggestion: string | null;
  readonly retryable: boolean;
  readonly context: CompilationErrorInfo["context"];

  constructor(code: ChitinErrorCode, message: string, options: ChitinErrorOptions = {}) {
    super(message);
    this.name = "ChitinError";
    this.code = code;
    this.stage = options.stage ?? null;
    this.suggestion = options.suggestion ?? null;
    this.retryable = options.retryable ?? false;
    this.context = { ...(options.context ?? {}) };
    if (options.cause !== undefined) this.cause = options.cause;
    // Preserve prototype chain when compiled to older targets.
    Object.setPrototypeOf(this, ChitinError.prototype);
  }

  toInfo(): CompilationErrorInfo {
    return {
      code: this.code,
      message: this.message,
      stage: this.stage,
      suggestion: this.suggestion,
      retryable: this.retryable,
      context: { ...this.context },
    };
  }
}
