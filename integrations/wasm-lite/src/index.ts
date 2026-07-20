export {
  decompose,
  initFromUrl,
  setModuleFactory,
  setWasmBinary,
  validateConfig,
  validateMeshInput,
} from "./decompose.js";
export { checkManifold } from "./manifold.js";
export { DecomposeWorker } from "./worker-client.js";
export type {
  DecomposeCallOptions,
  DecomposeWorkerOptions,
  WorkerLike,
} from "./worker-client.js";
export { mapWorkerError } from "./worker-protocol.js";
export type {
  DecomposeState,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.js";
export { quantizeHulls, validateHull, writePhys } from "./phys-writer.js";
export { ChitinError } from "./errors.js";
export type { ChitinErrorCode } from "./errors.js";
export type {
  ConvexHull,
  DecomposeConfig,
  DecomposeResult,
  QuantizedHull,
} from "./types.js";
