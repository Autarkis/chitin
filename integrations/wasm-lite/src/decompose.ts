import { ChitinError } from "./errors.js";
import type { ConvexHull, DecomposeConfig, DecomposeResult } from "./types.js";

// Embind handles must be released with delete(); the wrapper types make that
// explicit so cleanup is enforced in finally blocks.
interface EmScalarVector {
  size(): number;
  get(i: number): number;
  delete(): void;
}

interface EmHull {
  vertices: EmScalarVector;
  indices: EmScalarVector;
  delete?(): void;
}

interface EmHullVector {
  size(): number;
  get(i: number): EmHull;
  delete(): void;
}

interface CoACDResult {
  hulls: EmHullVector;
  delete?(): void;
}

interface CoACDModule {
  decompose(
    vertices: Float64Array,
    faces: Int32Array,
    threshold: number,
    maxConvexHull: number,
    prepResolution: number,
    sampleResolution: number,
    mctsNodes: number,
    mctsIteration: number,
    mctsMaxDepth: number,
    maxChVertex: number,
    merge: boolean,
  ): CoACDResult;
}

type ModuleFactory = (opts?: { wasmBinary?: ArrayBuffer }) => Promise<CoACDModule>;

let modulePromise: Promise<CoACDModule> | null = null;
let factory: ModuleFactory | null = null;

export function setModuleFactory(f: ModuleFactory): void {
  factory = f;
  modulePromise = null;
}

export function setWasmBinary(binary: ArrayBuffer): void {
  if (!factory) throw new Error("Call setModuleFactory before setWasmBinary");
  modulePromise = factory({ wasmBinary: binary });
}

async function getModule(): Promise<CoACDModule> {
  if (modulePromise) return modulePromise;
  if (!factory) throw new Error("Call setModuleFactory or initFromUrl first");
  modulePromise = factory();
  return modulePromise;
}

export async function initFromUrl(wasmJsUrl: string, wasmBinaryUrl: string): Promise<void> {
  const jsModule = await import(/* webpackIgnore: true */ wasmJsUrl);
  const createCoACD = jsModule.default || jsModule;
  const response = await fetch(wasmBinaryUrl);
  const wasmBinary = await response.arrayBuffer();
  factory = createCoACD;
  modulePromise = createCoACD({ wasmBinary });
  await modulePromise;
}

export function validateMeshInput(
  vertices: Float64Array,
  faces: Int32Array,
): void {
  if (vertices.length === 0 || faces.length === 0) {
    throw new ChitinError("INVALID_MESH", "empty vertices or faces");
  }
  if (vertices.length % 3 !== 0) {
    throw new ChitinError(
      "INVALID_MESH",
      `vertex array length ${vertices.length} is not a multiple of 3`,
    );
  }
  if (faces.length % 3 !== 0) {
    throw new ChitinError(
      "INVALID_MESH",
      `face array length ${faces.length} is not a multiple of 3`,
    );
  }
  // Every vertex coordinate must be finite; NaN/Infinity crash or hang CoACD.
  for (let i = 0; i < vertices.length; i++) {
    if (!Number.isFinite(vertices[i])) {
      throw new ChitinError(
        "INVALID_MESH",
        `vertex coordinate at position ${i} is not finite (${vertices[i]})`,
      );
    }
  }
  // Every face index must reference a real vertex; an out-of-range index reads
  // past the buffer inside native code.
  const vertexCount = vertices.length / 3;
  for (let i = 0; i < faces.length; i++) {
    const idx = faces[i];
    if (idx < 0 || idx >= vertexCount) {
      throw new ChitinError(
        "INVALID_MESH",
        `face index ${idx} at position ${i} is out of range [0, ${vertexCount})`,
      );
    }
  }
}

/** Reject decompose options that fall outside their valid range. */
export function validateConfig(config: DecomposeConfig): void {
  const { threshold, maxConvexHull } = config;
  if (threshold !== undefined && (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1)) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `threshold must be in (0, 1], got ${threshold}`,
    );
  }
  if (
    maxConvexHull !== undefined &&
    (!Number.isInteger(maxConvexHull) || maxConvexHull === 0 || maxConvexHull < -1)
  ) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `maxConvexHull must be -1 (unlimited) or a positive integer, got ${maxConvexHull}`,
    );
  }
  // CoACD 1.0.11 (public/coacd.cpp) explicitly rejects prep_resolution outside
  // [5, 1000]; a generic "positive" check would let native code throw instead.
  const { prepResolution } = config;
  if (
    prepResolution !== undefined &&
    (!Number.isInteger(prepResolution) || prepResolution < 5 || prepResolution > 1000)
  ) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `prepResolution must be an integer in [5, 1000] (CoACD's supported range), got ${prepResolution}`,
    );
  }
  for (const key of [
    "sampleResolution",
    "mctsNodes",
    "mctsIteration",
    "mctsMaxDepth",
    "maxChVertex",
  ] as const) {
    const v = config[key];
    if (v !== undefined && (!Number.isInteger(v) || v <= 0)) {
      throw new ChitinError(
        "INVALID_CONFIG",
        `${key} must be a positive integer, got ${v}`,
      );
    }
  }
}

export async function decompose(
  vertices: Float64Array,
  faces: Int32Array,
  config: DecomposeConfig = {},
): Promise<DecomposeResult> {
  validateMeshInput(vertices, faces);
  validateConfig(config);
  const mod = await getModule();

  const result = mod.decompose(
    vertices,
    faces,
    config.threshold ?? 0.05,
    config.maxConvexHull ?? -1,
    config.prepResolution ?? 50,
    config.sampleResolution ?? 2000,
    config.mctsNodes ?? 20,
    config.mctsIteration ?? 150,
    config.mctsMaxDepth ?? 3,
    config.maxChVertex ?? 256,
    config.merge ?? true,
  );

  // Every Embind handle we obtain (the result, its hull vector, each hull, and
  // each hull's vertex/index vectors) must be delete()d or the WASM heap grows
  // on every compile. Capture each handle once and release it in finally.
  const hulls: ConvexHull[] = [];
  const hullVec = result.hulls;
  try {
    const count = hullVec.size();
    for (let i = 0; i < count; i++) {
      const h = hullVec.get(i);
      const hv = h.vertices;
      const hi = h.indices;
      try {
        const nv = hv.size();
        const verts = new Float32Array(nv);
        for (let j = 0; j < nv; j++) verts[j] = hv.get(j);

        const ni = hi.size();
        const idx = new Uint32Array(ni);
        for (let j = 0; j < ni; j++) idx[j] = hi.get(j);

        hulls.push({ vertices: verts, indices: idx });
      } finally {
        hv.delete();
        hi.delete();
        h.delete?.();
      }
    }
  } finally {
    hullVec.delete();
    result.delete?.();
  }

  return { hulls };
}
