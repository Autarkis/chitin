import type { CompileGlbOptions, CompileGlbResult, GlbInput } from "../src/compiler.js";
import { ChitinCompiler } from "../src/compiler.js";
import type { ConvexHull, DecomposeConfig } from "../src/types.js";
import type { WorkerLike } from "../src/worker-client.js";
import type { WorkerRequest, WorkerResponse } from "../src/worker-protocol.js";
import { packGlb } from "./glb-fixture.js";

export function trackedSignal(controller: AbortController): {
  signal: AbortSignal;
  counts: () => { adds: number; removes: number };
} {
  const signal = controller.signal;
  let adds = 0;
  let removes = 0;
  const originalAdd = signal.addEventListener.bind(signal);
  const originalRemove = signal.removeEventListener.bind(signal);
  Object.defineProperty(signal, "addEventListener", {
    value: (...args: Parameters<typeof originalAdd>) => {
      adds++;
      return originalAdd(...args);
    },
  });
  Object.defineProperty(signal, "removeEventListener", {
    value: (...args: Parameters<typeof originalRemove>) => {
      removes++;
      return originalRemove(...args);
    },
  });
  return { signal, counts: () => ({ adds, removes }) };
}

export function internalCompile(
  compiler: ChitinCompiler,
  input: GlbInput,
  options: CompileGlbOptions,
): Promise<CompileGlbResult> {
  const withInternal = compiler as unknown as {
    compileGlbInternal(input: GlbInput, options: CompileGlbOptions): Promise<CompileGlbResult>;
  };
  return withInternal.compileGlbInternal(input, options);
}

export const HULLS: ConvexHull[] = [
  {
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    indices: new Uint32Array([0, 1, 2]),
  },
];

export class CompletingWorker implements WorkerLike {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  terminated = false;
  initialized = false;
  configs: DecomposeConfig[] = [];
  manifoldChecks: boolean[] = [];

  postMessage(message: WorkerRequest): void {
    if (message.type !== "decompose") return;
    this.configs.push(message.config);
    this.manifoldChecks.push(message.checkManifold);
    queueMicrotask(() => {
      if (!this.initialized) {
        this.emit({ type: "state", id: message.id, state: "loading-wasm" });
        this.initialized = true;
      }
      this.emit({ type: "state", id: message.id, state: "decomposing" });
      this.emit({
        type: "result",
        id: message.id,
        hulls: HULLS.map((hull) => ({
          vertices: hull.vertices.slice(),
          indices: hull.indices.slice(),
        })),
      });
    });
  }

  terminate(): void {
    this.terminated = true;
  }

  private emit(response: WorkerResponse): void {
    this.onmessage?.({ data: response } as MessageEvent<WorkerResponse>);
  }
}

export class HoldingWorker implements WorkerLike {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  terminated = false;
  postMessage(): void {}
  terminate(): void {
    this.terminated = true;
  }
}

export function compilerWith(factory: () => WorkerLike): ChitinCompiler {
  return new ChitinCompiler({
    wasm: { js: "coacd.mjs", wasm: "coacd.wasm", version: "0.2.0" },
    workerFactory: factory,
    maxWorkers: 1,
  });
}

export function makeScaledTetrahedraGlb(): ArrayBuffer {
  const vertices = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]);
  const indices = new Uint16Array([
    0, 2, 1,
    0, 1, 3,
    0, 3, 2,
    1, 2, 3,
  ]);
  const indexOffset = vertices.byteLength;
  const binary = new ArrayBuffer(indexOffset + indices.byteLength);
  new Float32Array(binary, 0, vertices.length).set(vertices);
  new Uint16Array(binary, indexOffset, indices.length).set(indices);
  return packGlb({
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0, 1] }],
    nodes: [
      { mesh: 0 },
      { mesh: 0, translation: [3, 0, 0], scale: [0.05, 0.05, 0.05] },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 4, type: "VEC3" },
      { bufferView: 1, componentType: 5123, count: 12, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: vertices.byteLength },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength },
    ],
    buffers: [{ byteLength: binary.byteLength }],
  }, binary);
}
