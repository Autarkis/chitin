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
}

export async function decompose(
  vertices: Float64Array,
  faces: Int32Array,
  config: DecomposeConfig = {},
): Promise<DecomposeResult> {
  validateMeshInput(vertices, faces);
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
