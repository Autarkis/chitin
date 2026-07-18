export {
  decompose,
  initFromUrl,
  setModuleFactory,
  setWasmBinary,
  validateMeshInput,
} from "./decompose.js";
export { quantizeHulls, validateHull, writePhys } from "./phys-writer.js";
export { ChitinError } from "./errors.js";
export type { ChitinErrorCode } from "./errors.js";
export type {
  ConvexHull,
  DecomposeConfig,
  DecomposeResult,
  QuantizedHull,
} from "./types.js";
