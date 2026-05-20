import type { ConvexHull, DecomposeConfig, DecomposeResult } from "./types.js";

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
  ): {
    hulls: {
      size(): number;
      get(i: number): { vertices: { size(): number; get(i: number): number }; indices: { size(): number; get(i: number): number } };
    };
  };
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

export async function decompose(
  vertices: Float64Array,
  faces: Int32Array,
  config: DecomposeConfig = {},
): Promise<DecomposeResult> {
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

  const hulls: ConvexHull[] = [];
  const count = result.hulls.size();

  for (let i = 0; i < count; i++) {
    const h = result.hulls.get(i);
    const nv = h.vertices.size();
    const ni = h.indices.size();

    const verts = new Float32Array(nv);
    for (let j = 0; j < nv; j++) verts[j] = h.vertices.get(j);

    const idx = new Uint32Array(ni);
    for (let j = 0; j < ni; j++) idx[j] = h.indices.get(j);

    hulls.push({ vertices: verts, indices: idx });
  }

  return { hulls };
}
