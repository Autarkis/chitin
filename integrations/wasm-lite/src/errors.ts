export type ChitinErrorCode =
  | "INVALID_MESH" // malformed input geometry (shape, finiteness, index bounds)
  | "NON_MANIFOLD" // input mesh is not manifold (reserved; set by the compiler)
  | "OUT_OF_MEMORY" // WASM heap exhausted during compilation
  | "CANCELLED"; // compilation was aborted by the caller

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
