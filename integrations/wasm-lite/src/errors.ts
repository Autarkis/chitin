// INVALID_MESH and INVALID_CONFIG come from the synchronous validators;
// NON_MANIFOLD, OUT_OF_MEMORY, CANCELLED, and WORKER_ERROR are emitted by the
// worker client (see worker-client.ts).
export type ChitinErrorCode =
  | "INVALID_MESH" // malformed input geometry (shape, finiteness, index bounds)
  | "INVALID_CONFIG" // a decompose option is out of its valid range
  | "NON_MANIFOLD" // input mesh fails the optional manifold precheck
  | "OUT_OF_MEMORY" // the WASM heap could not grow during decomposition
  | "CANCELLED" // aborted by the caller; the worker was terminated
  | "WORKER_ERROR"; // the worker failed unexpectedly (load or runtime fault)

export class ChitinError extends Error {
  readonly code: ChitinErrorCode;

  constructor(code: ChitinErrorCode, message: string) {
    super(message);
    this.name = "ChitinError";
    this.code = code;
    // Preserve prototype chain when compiled to older targets.
    Object.setPrototypeOf(this, ChitinError.prototype);
  }
}
