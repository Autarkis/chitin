import { decompose, initFromUrl, validateMeshInput } from "./decompose.js";
import { checkManifold } from "./manifold.js";
import {
  mapWorkerError,
  type WorkerRequest,
  type WorkerResponse,
} from "./worker-protocol.js";

// Entry point that runs inside the Web Worker. ChitinWorkerClient (worker-client.ts)
// spawns this module, sends one `init` then `decompose` messages, and receives
// `state` / `result` / `error` back. CoACD runs synchronously here, so the only
// way to cancel is for the client to terminate the worker.

// The default lib types `self` as a Window, whose postMessage signature differs
// from a worker's. Narrow it to just what this module uses.
interface WorkerContext {
  onmessage: ((ev: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
}
const ctx = self as unknown as WorkerContext;

let wasmJsUrl: string | null = null;
let wasmBinaryUrl: string | null = null;
let initialized = false;

let poissonWasmJsUrl: string | null = null;
let poissonWasmBinaryUrl: string | null = null;
let poissonInitialized = false;

interface EmbindVector {
  size(): number;
  get(i: number): number;
  delete?(): void;
}

let poissonModule: {
  poissonReconstruct(
    positions: Float64Array,
    normals: Float64Array,
    depth: number,
    densityQuantile: number,
  ): { vertices: EmbindVector; indices: EmbindVector };
} | null = null;

async function loadPoissonModule(): Promise<typeof poissonModule> {
  if (poissonModule) return poissonModule;
  if (!poissonWasmJsUrl || !poissonWasmBinaryUrl) {
    throw new Error("worker received poisson before init-poisson");
  }
  const wasmResponse = await fetch(poissonWasmBinaryUrl);
  const wasmBinary = await wasmResponse.arrayBuffer();
  const factory = (await import(/* webpackIgnore: true */ poissonWasmJsUrl)).default;
  poissonModule = await factory({ wasmBinary });
  poissonInitialized = true;
  return poissonModule;
}

async function handlePoisson(req: Extract<WorkerRequest, { type: "poisson" }>): Promise<void> {
  const { id } = req;
  try {
    if (!poissonInitialized) {
      ctx.postMessage({ type: "state", id, state: "loading-poisson-wasm" });
      await loadPoissonModule();
    }
    ctx.postMessage({ type: "state", id, state: "reconstructing" });
    const raw = poissonModule!.poissonReconstruct(
      req.positions, req.normals, req.depth, req.densityQuantile,
    );
    const vertCount = raw.vertices.size();
    const idxCount = raw.indices.size();
    const vertices = new Float64Array(vertCount);
    for (let i = 0; i < vertCount; i++) vertices[i] = raw.vertices.get(i);
    const faces = new Int32Array(idxCount);
    for (let i = 0; i < idxCount; i++) faces[i] = raw.indices.get(i);
    raw.vertices.delete?.();
    raw.indices.delete?.();
    ctx.postMessage({ type: "state", id, state: "done" });
    const transfer = [vertices.buffer, faces.buffer];
    ctx.postMessage({ type: "poisson-result", id, vertices, faces }, transfer);
  } catch (err) {
    ctx.postMessage({ type: "error", id, ...mapWorkerError(err) });
  }
}

async function handleDecompose(req: Extract<WorkerRequest, { type: "decompose" }>): Promise<void> {
  const { id } = req;
  try {
    // Validate before loading or entering CoACD. The lightweight WASM build has
    // no OpenVDB repair, so an open component can only fail downstream.
    if (req.checkManifold) {
      validateMeshInput(req.vertices, req.faces);
      checkManifold(req.vertices, req.faces);
    }
    if (!initialized) {
      if (!wasmJsUrl || !wasmBinaryUrl) {
        throw new Error("worker received decompose before init");
      }
      ctx.postMessage({ type: "state", id, state: "loading-wasm" });
      await initFromUrl(wasmJsUrl, wasmBinaryUrl);
      initialized = true;
    }
    ctx.postMessage({ type: "state", id, state: "decomposing" });
    const { hulls } = await decompose(req.vertices, req.faces, req.config);
    ctx.postMessage({ type: "state", id, state: "done" });
    // Move the hull buffers back to the main thread instead of copying them.
    const transfer = hulls.flatMap((h) => [h.vertices.buffer, h.indices.buffer]);
    ctx.postMessage({ type: "result", id, hulls }, transfer);
  } catch (err) {
    ctx.postMessage({ type: "error", id, ...mapWorkerError(err) });
  }
}

ctx.onmessage = (ev: MessageEvent<WorkerRequest>): void => {
  const req = ev.data;
  if (req.type === "init") {
    wasmJsUrl = req.wasmJsUrl;
    wasmBinaryUrl = req.wasmBinaryUrl;
    return;
  }
  if (req.type === "init-poisson") {
    poissonWasmJsUrl = req.wasmJsUrl;
    poissonWasmBinaryUrl = req.wasmBinaryUrl;
    return;
  }
  if (req.type === "decompose") {
    void handleDecompose(req);
  }
  if (req.type === "poisson") {
    void handlePoisson(req);
  }
};
