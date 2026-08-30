export {
  decompose,
  initFromUrl,
  setModuleFactory,
  setWasmBinary,
  validateConfig,
  validateMeshInput,
} from "./decompose.js";
export { analyzeManifold, checkManifold } from "./manifold.js";
export type { ManifoldAnalysis } from "./manifold.js";
export { ChitinWorkerClient } from "./worker-client.js";
export type {
  DecomposeCallOptions,
  ChitinWorkerClientOptions,
  WorkerLike,
} from "./worker-client.js";
export { mapWorkerError } from "./worker-protocol.js";
export type {
  DecomposeState,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.js";
export { quantizeHulls, validateHull, writePhys } from "./phys-writer.js";
export {
  COMPILATION_REPORT_VERSION,
  createCompilationReport,
  validateCompilationReport,
} from "./report.js";
export type {
  CompilationCheck,
  CompilationErrorInfo,
  CompilationMetric,
  CompilationProgress,
  CompilationReport,
  CompilationRuntime,
  CompilationStage,
  CompilationVerdict,
  CompilationWarning,
  CreateCompilationReportOptions,
} from "./report.js";
export { ChitinError } from "./errors.js";
export type { ChitinErrorCode, ChitinErrorOptions } from "./errors.js";
export { parseGlb } from "./glb.js";
export type { ParsedGlbMesh } from "./glb.js";
export { ChitinCompiler, compileGlb, compileGaussianField } from "./compiler.js";
export type {
  ChitinCompilerOptions,
  CompileGaussianFieldOptions,
  CompileGlbOptions,
  CompileGlbResult,
  GlbInput,
  InteractiveComponentPolicy,
  OneShotCompileGaussianFieldOptions,
  OneShotCompileGlbOptions,
  WasmAssetUrls,
} from "./compiler.js";
export type {
  GaussianFieldInput,
  GaussianFieldReconstructionOptions,
  CanonicalGaussianField,
  ScaleEncoding,
  OpacityEncoding,
  QuaternionOrder,
} from "./splat-preprocess.js";
export {
  canonicalizeGaussianField,
  preprocessGaussianField,
} from "./splat-preprocess.js";
export { evaluateColliderQuality } from "./quality.js";
export type {
  ColliderQualityComponentResult,
  ColliderQualityOptions,
  ColliderQualityResult,
} from "./quality.js";
export { CHITIN_LITE_VERSION } from "./version.js";
export type {
  ConvexHull,
  DecomposeConfig,
  DecomposeResult,
  QuantizedHull,
} from "./types.js";
