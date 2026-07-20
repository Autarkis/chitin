// Only INVALID_MESH and INVALID_CONFIG are currently emitted.
export type ChitinErrorCode =
  | "INVALID_MESH" // malformed input geometry (shape, finiteness, index bounds)
  | "INVALID_CONFIG" // a decompose option is out of its valid range
  | "NON_MANIFOLD" // reserved: input mesh is not manifold (not yet emitted)
  | "OUT_OF_MEMORY" // reserved: WASM heap exhausted (not yet mapped from CoACD)
  | "CANCELLED"; // reserved: aborted by the caller (needs the worker API)

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
