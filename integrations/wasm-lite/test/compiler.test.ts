import { afterEach, describe, expect, it, vi } from "vitest";

import type { CompileGlbOptions, CompileGlbResult, GlbInput } from "../src/compiler.js";
import { ChitinCompiler, compileGlb } from "../src/compiler.js";
import type { ConvexHull, DecomposeConfig } from "../src/types.js";
import type { WorkerLike } from "../src/worker-client.js";
import type { WorkerRequest, WorkerResponse } from "../src/worker-protocol.js";
import { makeAdaptiveHullBudgetGlb, makeGlb, makeThinOpenTrayGlb, packGlb } from "./glb-fixture.js";

/**
 * Wraps a real AbortSignal's addEventListener/removeEventListener with counting
 * proxies so tests can assert listener balance without depending on the
 * signal's internal listener bookkeeping (which auto-clears "once" listeners
 * on fire, independently of whether the code under test ever calls
 * removeEventListener itself).
 */
function trackedSignal(controller: AbortController): { signal: AbortSignal; counts: () => { adds: number; removes: number } } {
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

/** Reaches past the public compileGlb wrapper (which allocates its own merged
 * AbortSignal via mergeSignals) to call compileGlbInternal directly, so the
 * signal seen by the listener-balance assertions is the exact object
 * compileGlbInternal attaches its local abort listener to. */
function internalCompile(
  compiler: ChitinCompiler,
  input: GlbInput,
  options: CompileGlbOptions,
): Promise<CompileGlbResult> {
  const withInternal = compiler as unknown as {
    compileGlbInternal(input: GlbInput, options: CompileGlbOptions): Promise<CompileGlbResult>;
  };
  return withInternal.compileGlbInternal(input, options);
}

const HULLS: ConvexHull[] = [
  {
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    indices: new Uint32Array([0, 1, 2]),
  },
];

class CompletingWorker implements WorkerLike {
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
      this.emit({ type: "result", id: message.id, hulls: HULLS });
    });
  }

  terminate(): void {
    this.terminated = true;
  }

  private emit(response: WorkerResponse): void {
    this.onmessage?.({ data: response } as MessageEvent<WorkerResponse>);
  }
}

class HoldingWorker implements WorkerLike {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  terminated = false;
  postMessage(): void {}
  terminate(): void {
    this.terminated = true;
  }
}

function compilerWith(factory: () => WorkerLike): ChitinCompiler {
  return new ChitinCompiler({
    wasm: { js: "coacd.mjs", wasm: "coacd.wasm", version: "0.2.0" },
    workerFactory: factory,
    maxWorkers: 1,
  });
}

function makeScaledTetrahedraGlb(): ArrayBuffer {
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

describe("ChitinCompiler", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("compiles an ArrayBuffer and returns .phys, source facts, progress, and the v1 report", async () => {
    const stages: string[] = [];
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), {
      onProgress: ({ stage }) => stages.push(stage),
      decompose: { threshold: 0.05 },
    });
    compiler.terminate();

    expect(new DataView(result.phys).getUint32(0, true)).toBe(0x53594850);
    expect(result.hulls).toHaveLength(2);
    expect(result.source).toEqual({
      mesh_count: 2,
      primitive_count: 4,
      node_count: 2,
      vertex_count: 12,
      triangle_count: 4,
    });
    expect(stages.slice(0, 4)).toEqual([
      "reading-input", "parsing-input", "validating-input", "loading-wasm",
    ]);
    expect(stages.filter((stage) => stage === "decomposing")).toHaveLength(5);
    expect(stages.slice(-2)).toEqual(["writing-phys", "done"]);
    expect(result.report).toMatchObject({
      report_version: 1,
      profile: "interactive",
      verdict: { status: "not_evaluated" },
      input: { kind: "glb", source_vertices: 12, processed_vertices: 6, mesh_vertices: 6 },
      runtime: {
        kind: "browser_wasm",
        version: "0.2.0",
        dependencies: { "@autarkis/chitin-coacd-wasm": "0.2.0" },
      },
      reproducibility: { scope: "same_runtime_toolchain", deterministic: null },
      config: { requested: { threshold: 0.05 } },
    });
    expect(result.report.reproducibility.artifact_sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("measures artifact fit only when quality sampling is requested", async () => {
    const stages: string[] = [];
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), {
      checkManifold: false,
      quality: { surfaceSamples: 32, volumeSamples: 64, minColliderSamples: 1 },
      onProgress: ({ stage }) => stages.push(stage),
    });
    compiler.terminate();

    expect(stages).toContain("verifying");
    expect(result.report.verdict.status).toBe("not_evaluated");
    expect(result.report.metrics.quality_method).toEqual({
      value: "deterministic_halton_v1",
      unit: "method",
      status: "measured",
    });
    expect(result.report.metrics.source_surface_coverage.status).toBe("measured");
    expect(result.report.timings_ms.verify).toBeGreaterThanOrEqual(0);
  });

  it("accepts Blob and ArrayBufferView inputs", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb(new Blob([makeGlb()]))).resolves.toMatchObject({ source: { triangle_count: 4 } });
    const bytes = new Uint8Array(makeGlb());
    await expect(compiler.compileGlb(bytes.subarray(0))).resolves.toMatchObject({ source: { vertex_count: 12 } });
    compiler.terminate();
  });

  it("fetches URL inputs and records the source artifact", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(makeGlb(), { status: 200 })));
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(new URL("https://example.test/chair.glb"));
    expect(fetch).toHaveBeenCalledWith(
      "https://example.test/chair.glb",
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(result.report.artifacts.source).toBe("https://example.test/chair.glb");
    compiler.terminate();
  });

  it("returns a structured LOAD_ERROR for failed URL inputs", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 404 })));
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb("https://example.test/missing.glb")).rejects.toMatchObject({
      code: "LOAD_ERROR",
      stage: "reading-input",
      retryable: false,
      context: { http_status: 404 },
    });
    compiler.terminate();
  });

  it("terminate cancels a compilation while its URL input is loading", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
          }),
      ),
    );
    const compiler = compilerWith(() => new CompletingWorker());
    const pending = compiler.compileGlb("https://example.test/slow.glb");
    compiler.terminate();
    await expect(pending).rejects.toMatchObject({
      code: "CANCELLED",
      stage: "reading-input",
      retryable: true,
    });
  });

  it("rejects unsupported profiles instead of attaching an inert label", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(
      compiler.compileGlb(makeGlb(), { profile: "robotics" as "interactive" }),
    ).rejects.toMatchObject({ code: "INVALID_CONFIG", stage: "validating-input" });
    compiler.terminate();
  });

  it("requires enough maxConvexHull budget for disconnected components", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb(makeGlb(), { decompose: { maxConvexHull: 1 } })).rejects.toMatchObject({
      code: "INVALID_CONFIG",
      stage: "validating-input",
      context: { component_count: 2, max_convex_hull: 1 },
    });
    compiler.terminate();
  });

  it("reserves maxConvexHull capacity for every disconnected component", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    await compiler.compileGlb(makeGlb(), { decompose: { maxConvexHull: 3 } });
    expect(worker.configs).toEqual([
      { mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, threshold: 0.1, maxChVertex: 4, maxConvexHull: 2 },
      { mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, threshold: 0.1, maxChVertex: 4, maxConvexHull: 1 },
    ]);
    compiler.terminate();
  });

  it("checks every GLB component for browser-incompatible open geometry by default", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    await compiler.compileGlb(makeGlb());
    expect(worker.manifoldChecks).toEqual([true, true]);
    compiler.terminate();
  });

  it("uses a deterministic scene-aware budget and reuses small-part results across detail changes", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const file = new Blob([makeScaledTetrahedraGlb()]);
    const firstProgress: Array<{ completed?: number; total?: number }> = [];
    const first = await compiler.compileGlb(file, {
      decompose: { threshold: 0.05 },
      onProgress: (progress) => {
        if (progress.stage === "decomposing") firstProgress.push(progress);
      },
    });
    expect(worker.configs).toEqual([
      { threshold: 0.1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 127 },
      { threshold: 1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 1 },
    ]);
    expect(firstProgress.map(({ completed, total }) => [completed, total])).toContainEqual([2, 2]);
    expect(first.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_SMALL_COMPONENTS_SIMPLIFIED",
    }));
    expect(first.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_THRESHOLD_CLAMPED",
    }));
    expect(first.report.config.effective).toMatchObject({
      max_hulls: 128,
      max_hulls_ceiling: 128,
      detail_budget_ratio: 1,
      component_count: 2,
      simplified_component_count: 1,
    });
    expect(first.reuse).toEqual({
      prepared_geometry: false,
      component_results: 0,
      total_components: 2,
    });

    const secondMessages: string[] = [];
    const second = await compiler.compileGlb(file, {
      decompose: { threshold: 0.2 },
      onProgress: ({ message }) => { if (message) secondMessages.push(message); },
    });
    expect(worker.configs).toEqual([
      { threshold: 0.1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 127 },
      { threshold: 1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 1 },
      { threshold: 0.2, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 119 },
    ]);
    expect(secondMessages).toContain("Reusing prepared triangle geometry");
    expect(secondMessages.some((message) => message.includes("1 reused"))).toBe(true);
    expect(second.reuse).toEqual({
      prepared_geometry: true,
      component_results: 1,
      total_components: 2,
    });
    expect(second.report.config.effective).toMatchObject({
      max_hulls: 120,
      max_hulls_ceiling: 128,
    });
    expect(second.report.config.effective?.detail_budget_ratio).toBeCloseTo(0.94);
    compiler.terminate();
  });

  it("limits coarse thresholds for a scene-dominant connected body", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const result = await compiler.compileGlb(makeScaledTetrahedraGlb(), {
      decompose: { threshold: 0.6 },
    });
    const detailed = worker.configs.find((config) => config.threshold !== 1);

    expect(detailed?.threshold).toBeGreaterThanOrEqual(0.1);
    expect(detailed?.threshold).toBeLessThan(0.22);
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_IMPORTANCE_GUARD",
      context: expect.objectContaining({ requested_threshold: 0.6 }),
    }));
    expect(result.report.config.effective).toMatchObject({
      important_component_max_threshold: 0.14,
      importance_guarded_component_count: 1,
    });
    compiler.terminate();
  });

  it("bounds recent component configurations while retaining nearby detail results", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const file = new Blob([makeScaledTetrahedraGlb()]);
    for (const threshold of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) {
      await compiler.compileGlb(file, {
        decompose: { threshold },
        componentPolicy: { enabled: false },
      });
    }
    expect(worker.configs).toHaveLength(14); // Seven variants for each of two components.

    const recent = await compiler.compileGlb(file, {
      decompose: { threshold: 0.7 },
      componentPolicy: { enabled: false },
    });
    expect(worker.configs).toHaveLength(14);
    expect(recent.reuse.component_results).toBe(2);

    const expired = await compiler.compileGlb(file, {
      decompose: { threshold: 0.1 },
      componentPolicy: { enabled: false },
    });
    expect(worker.configs).toHaveLength(16);
    expect(expired.reuse.component_results).toBe(0);
    compiler.terminate();
  });

  it("limits coarse settings on low-occupancy shells instead of filling their interiors", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const result = await compiler.compileGlb(makeThinOpenTrayGlb(), {
      decompose: { threshold: 0.58 },
    });

    expect(worker.configs).toEqual([{
      threshold: 0.05,
      mctsNodes: 8,
      mctsIteration: 40,
      mctsMaxDepth: 2,
      maxChVertex: 16,
      maxConvexHull: 93,
    }]);
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_HOLLOW_SHELL_GUARD",
      context: expect.objectContaining({
        requested_threshold: 0.58,
        effective_hollow_shell_threshold: 0.05,
      }),
    }));
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_HULL_VERTICES_ADAPTED",
    }));
    expect(result.report.config.effective).toMatchObject({
      hollow_shell_component_count: 1,
      guarded_hollow_shell_component_count: 1,
      hollow_shell_threshold: 0.05,
      hollow_shell_min_hulls: 8,
      adaptive_hull_vertices: true,
      max_hulls: 93,
      max_hulls_ceiling: 128,
      detail_budget_coarse_ratio: 0.7,
      effective_component_hull_vertices_min: 16,
      effective_component_hull_vertices_max: 16,
    });
    expect(result.report.config.effective?.detail_budget_ratio).toBeCloseTo(0.712);
    compiler.terminate();
  });

  it("rejects an explicit hull budget that cannot satisfy hollow-shell reservations", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb(makeThinOpenTrayGlb(), {
      decompose: { threshold: 0.58 },
      componentPolicy: { maxHulls: 7 },
    })).rejects.toMatchObject({
      code: "INVALID_CONFIG",
      stage: "validating-input",
      context: {
        hollow_shell_component_count: 1,
        max_hulls: 7,
        required_minimum_hulls: 8,
      },
    });
    compiler.terminate();
  });

  it("assigns hull vertices from both geometric roundness and scene-relative size", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const result = await compiler.compileGlb(makeAdaptiveHullBudgetGlb());
    const caps = worker.configs.map((config) => config.maxChVertex as number);

    expect(caps).toHaveLength(3);
    expect(caps[0]).toBeGreaterThan(caps[1]); // sphere vs same-diagonal thin ellipsoid
    expect(caps[1]).toBeGreaterThan(caps[2]); // scene-scale dominates the small sphere
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_HULL_VERTICES_ADAPTED",
    }));
    expect(result.report.config.effective).toMatchObject({
      adaptive_hull_vertices: true,
      hull_vertex_roundness_metric: "isoperimetric_quotient",
      effective_component_hull_vertices_min: caps[2],
      effective_component_hull_vertices_max: caps[0],
    });
    compiler.terminate();
  });

  it("runs connected components through a bounded worker pool", async () => {
    let active = 0;
    let maximumActive = 0;
    class DelayedWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        active++;
        maximumActive = Math.max(maximumActive, active);
        setTimeout(() => {
          active--;
          this.emitResult(message.id);
        }, 10);
      }

      private emitResult(id: number): void {
        this.onmessage?.({ data: { type: "result", id, hulls: HULLS } } as MessageEvent<WorkerResponse>);
      }
    }
    const compiler = new ChitinCompiler({
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => new DelayedWorker(),
      maxWorkers: 2,
    });
    await compiler.compileGlb(makeGlb(), {
      componentPolicy: { enabled: false },
      checkManifold: false,
    });
    expect(maximumActive).toBe(2);
    compiler.terminate();
  });

  it("reports the first invalid component deterministically before parallel scheduling", async () => {
    const compiler = new ChitinCompiler({
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => new CompletingWorker(),
      maxWorkers: 2,
    });
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      code: "NON_MANIFOLD",
      stage: "validating-input",
      context: { component_index: 0, component_number: 1, component_count: 2 },
    });
    compiler.terminate();
  });

  it("adds connected-part context and full-compiler guidance to topology failures", async () => {
    class RejectingWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        queueMicrotask(() => this.onmessage?.({ data: {
          type: "error",
          id: message.id,
          code: "NON_MANIFOLD",
          message: "boundary (open) edge; topology summary: 3 boundary edges",
          stage: null,
          suggestion: null,
          retryable: false,
          context: { boundary_edges: 3, non_manifold_edges: 0, degenerate_triangles: 0 },
        } } as MessageEvent<WorkerResponse>));
      }
    }
    const compiler = compilerWith(() => new RejectingWorker());
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      code: "NON_MANIFOLD",
      stage: "validating-input",
      retryable: false,
      context: {
        component_index: 0,
        component_number: 1,
        component_count: 2,
        boundary_edges: 3,
      },
    });
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      suggestion: expect.stringContaining("full Chitin compiler"),
    });
    compiler.terminate();
  });

  it("cancels an in-flight compile and can recover on a fresh worker", async () => {
    const workers: WorkerLike[] = [];

    // Resolves once the holding worker actually receives the decompose
    // message, i.e. once compileGlbInternal has genuinely reached the
    // decompose stage and spawned its worker — not after some assumed number
    // of microtask ticks.
    let spawnSignalled: () => void;
    const spawnSignal = new Promise<void>((resolve) => {
      spawnSignalled = resolve;
    });
    class SignallingHoldingWorker extends HoldingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type === "decompose") spawnSignalled();
      }
    }

    // Explicit intent, not call-count inference: the compile under
    // cancellation is handed a worker that never responds; the recovery
    // compile is handed one that completes normally. This stays correct no
    // matter how many workers a single compile spawns before reaching
    // decompose.
    const roles: Array<() => WorkerLike> = [
      () => new SignallingHoldingWorker(),
      () => new CompletingWorker(),
    ];
    const compiler = compilerWith(() => {
      const makeWorker = roles.shift();
      if (!makeWorker) throw new Error("test expected exactly two workers to be spawned");
      const worker = makeWorker();
      workers.push(worker);
      return worker;
    });
    const controller = new AbortController();
    const pending = compiler.compileGlb(makeGlb(), { signal: controller.signal });
    await spawnSignal;
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "CANCELLED" });
    await expect(compiler.compileGlb(makeGlb())).resolves.toMatchObject({ source: { mesh_count: 2 } });
    expect(workers).toHaveLength(2);
    compiler.terminate();
  });

  it("rejects concurrent work immediately with COMPILER_BUSY", async () => {
    const compiler = compilerWith(() => new HoldingWorker());
    const controller = new AbortController();
    const pending = compiler.compileGlb(makeGlb(), { signal: controller.signal });
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      code: "COMPILER_BUSY",
      retryable: true,
    });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "CANCELLED" });
  });

  it("one-shot compileGlb always releases its worker", async () => {
    const worker = new CompletingWorker();
    await compileGlb(makeGlb(), {
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => worker,
      maxWorkers: 1,
    });
    expect(worker.terminated).toBe(true);
  });

  describe("compileGlbInternal abort-listener balance (characterization)", () => {
    // compileGlbInternal attaches a listener to options.signal for its local
    // AbortController (src/compiler.ts ~line 922) and removes it at exactly
    // two points: the sorted-manifold-validation throw path (~line 988) and
    // right after the decompose worker pool settles (~line 1061). Neither is
    // inside a try/finally. These tests pin whether adds === removes across
    // the normal, error, and abort exits so a refactor cannot silently change
    // the balance.

    it("balances add/removeEventListener on a normal completion", async () => {
      const compiler = compilerWith(() => new CompletingWorker());
      const controller = new AbortController();
      const { signal, counts } = trackedSignal(controller);

      await internalCompile(compiler, makeGlb(), { signal });

      expect(counts()).toEqual({ adds: 1, removes: 1 });
      compiler.terminate();
    });

    it("balances add/removeEventListener when a worker decompose rejects", async () => {
      class RejectingWorker extends CompletingWorker {
        override postMessage(message: WorkerRequest): void {
          if (message.type !== "decompose") return;
          queueMicrotask(() => this.onmessage?.({
            data: {
              type: "error",
              id: message.id,
              code: "WORKER_ERROR",
              message: "simulated decompose failure",
              stage: null,
              suggestion: null,
              retryable: false,
              context: {},
            },
          } as MessageEvent<WorkerResponse>));
        }
      }
      const compiler = compilerWith(() => new RejectingWorker());
      const controller = new AbortController();
      const { signal, counts } = trackedSignal(controller);

      await expect(internalCompile(compiler, makeGlb(), { signal })).rejects.toMatchObject({
        code: "WORKER_ERROR",
      });

      const balance = counts();
      if (balance.adds !== balance.removes) {
        // Leak confirmed on the decompose-rejection path: compileGlbInternal
        // throws firstError (line 1062) but only reaches the matching
        // removeEventListener at line 1061 if the decompose pool's
        // Promise.all (line 1060) itself resolves; see the full trace in the
        // task report. Recording the observed counts rather than weakening
        // the assertion.
        console.warn(`abort-listener leak on error path: adds=${balance.adds} removes=${balance.removes}`);
      }
      expect(balance).toEqual({ adds: 1, removes: 1 });
      compiler.terminate();
    });

    it("balances add/removeEventListener when the signal aborts mid-flight", async () => {
      const compiler = compilerWith(() => new HoldingWorker());
      const controller = new AbortController();
      const { signal, counts } = trackedSignal(controller);

      const pending = internalCompile(compiler, makeGlb(), { signal });
      await Promise.resolve();
      controller.abort();

      await expect(pending).rejects.toMatchObject({ code: "CANCELLED" });
      expect(counts()).toEqual({ adds: 1, removes: 1 });
      compiler.terminate();
    });
  });

  it("snapshots report.config.effective for a representative compile", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), { decompose: { threshold: 0.05 } });
    compiler.terminate();

    expect(result.report.config.effective).toMatchInlineSnapshot(`
      {
        "adaptive_hull_vertices": true,
        "component_count": 2,
        "component_plans": [
          {
            "allocation_weight": 1.5,
            "component_index": 0,
            "diagonal_ratio": 0.12803687993289598,
            "hollow_shell": false,
            "importance": 1,
            "max_hull_vertices": 4,
            "max_hulls": 64,
            "occupancy_ratio": 1,
            "output_hulls": 1,
            "output_triangles": 1,
            "roundness": 0,
            "simplified": false,
            "threshold": 0.1,
            "triangle_count": 1,
            "volume_ratio": 1,
          },
          {
            "allocation_weight": 1.5,
            "component_index": 1,
            "diagonal_ratio": 0.12803687993289598,
            "hollow_shell": false,
            "importance": 1,
            "max_hull_vertices": 4,
            "max_hulls": 64,
            "occupancy_ratio": 1,
            "output_hulls": 1,
            "output_triangles": 1,
            "roundness": 0,
            "simplified": false,
            "threshold": 0.1,
            "triangle_count": 1,
            "volume_ratio": 1,
          },
        ],
        "detail_budget_coarse_ratio": 0.7,
        "detail_budget_coarse_threshold": 0.6,
        "detail_budget_fine_threshold": 0.1,
        "detail_budget_ratio": 1,
        "detailed_component_min_threshold": 0.1,
        "detailed_component_threshold": 0.1,
        "effective_component_hull_vertices_max": 4,
        "effective_component_hull_vertices_mean": 4,
        "effective_component_hull_vertices_min": 4,
        "effective_component_threshold_max": 0.1,
        "effective_component_threshold_min": 0.1,
        "guarded_hollow_shell_component_count": 0,
        "hollow_shell_component_count": 0,
        "hollow_shell_max_occupancy_ratio": 0.05,
        "hollow_shell_max_threshold": 0.05,
        "hollow_shell_min_hulls": 8,
        "hollow_shell_threshold": null,
        "hull_vertex_roundness_metric": "isoperimetric_quotient",
        "importance_guarded_component_count": 0,
        "important_component_max_occupancy_ratio": 0.5,
        "important_component_max_threshold": 0.14,
        "max_hull_vertices": 96,
        "max_hulls": 128,
        "max_hulls_ceiling": 128,
        "max_workers": 1,
        "mcts_iterations": 40,
        "mcts_max_depth": 2,
        "mcts_nodes": 8,
        "min_hull_vertices": 8,
        "requested_max_hull_vertices": 256,
        "simplified_component_count": 0,
        "small_component_max_diagonal_ratio": 0.2,
        "small_component_max_volume_ratio": 0.005,
        "small_component_threshold": 1,
      }
    `);
  });

  it("rewraps a stage-less writePhys failure with stage: writing-phys and preserves the cause", async () => {
    class EmptyHullWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        queueMicrotask(() => this.onmessage?.({
          data: {
            type: "result",
            id: message.id,
            hulls: [{ vertices: new Float32Array(0), indices: new Uint32Array(0) }],
          },
        } as MessageEvent<WorkerResponse>));
      }
    }
    const compiler = compilerWith(() => new EmptyHullWorker());

    const failure = await compiler.compileGlb(makeGlb()).catch((error: unknown) => error);
    compiler.terminate();

    expect(failure).toMatchObject({
      code: "INVALID_MESH",
      stage: "writing-phys",
      message: expect.stringContaining("empty hull"),
    });
    expect((failure as { cause?: unknown }).cause).toMatchObject({
      code: "INVALID_MESH",
      stage: null,
      message: expect.stringContaining("empty hull"),
    });
  });

  it("keeps a cached component result after a concurrent sibling fails, and distributes work across workers once the compile succeeds", async () => {
    let worker1Attempts = 0;
    class FlakyWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        worker1Attempts++;
        if (worker1Attempts === 1) {
          queueMicrotask(() => this.onmessage?.({
            data: {
              type: "error",
              id: message.id,
              code: "WORKER_ERROR",
              message: "simulated first-attempt failure",
              stage: null,
              suggestion: null,
              retryable: false,
              context: {},
            },
          } as MessageEvent<WorkerResponse>));
          return;
        }
        super.postMessage(message);
      }
    }
    const worker0 = new CompletingWorker();
    const worker1 = new FlakyWorker();
    // worker1's WORKER_ERROR discards its underlying raw worker (see
    // DecomposeWorker.onMessage), so the pool slot that owned it spawns a
    // fresh raw worker the next time that slot's DecomposeWorker calls
    // decompose(). worker1Respawn stands in for that fresh worker.
    const worker1Respawn = new CompletingWorker();
    const workers = [worker0, worker1, worker1Respawn];
    let nextWorker = 0;
    const compiler = new ChitinCompiler({
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => workers[nextWorker++],
      maxWorkers: 2,
    });
    const file = new Blob([makeScaledTetrahedraGlb()]);

    await expect(compiler.compileGlb(file, { componentPolicy: { enabled: false } })).rejects.toMatchObject({
      code: "WORKER_ERROR",
    });
    expect(worker0.configs).toHaveLength(1); // component 0 succeeded and cached before component 1 failed.

    const recovered = await compiler.compileGlb(file, { componentPolicy: { enabled: false } });

    // The cache entry for component 0 survives: cursor 0 is a cache hit on the
    // second compile. A cache hit resolves synchronously, but the pool now
    // yields once on that path before continuing the loop, so the *other*
    // pool slot's runQueue turn gets scheduled and claims cursor 1 instead of
    // the first slot falling through to grab it too. Pool slot 0 handles the
    // cache hit for component 0 (no decompose call needed); pool slot 1
    // performs the real decompose for component 1 on a freshly respawned raw
    // worker (its previous raw worker was discarded after the WORKER_ERROR)
    // -- the two plan slots land on two different pool workers instead of
    // both piling onto slot 0.
    expect(worker0.configs).toHaveLength(1); // unchanged: component 0 was a cache hit, no decompose call made.
    expect(worker1.configs).toHaveLength(0); // this raw worker was discarded after its WORKER_ERROR; never reused.
    expect(worker1Respawn.configs).toHaveLength(1); // component 1's real decompose landed on pool slot 1's respawned worker.
    expect(recovered.reuse.component_results).toBe(1); // only component 0 was a cache hit.
    compiler.terminate();
  });

  describe("quality metric exact values (characterization)", () => {
    it("pins exact aggregate and per-component quality_component_* values, and their key order", async () => {
      const compiler = compilerWith(() => new CompletingWorker());
      const result = await compiler.compileGlb(makeGlb(), {
        checkManifold: false,
        quality: { surfaceSamples: 32, volumeSamples: 64, minColliderSamples: 1 },
      });
      compiler.terminate();

      const metrics = result.report.metrics;
      expect(metrics.quality_component_count).toEqual({ value: 2, unit: "count", status: "measured" });
      expect(metrics.quality_surface_samples).toEqual({ value: 32, unit: "count", status: "measured" });
      expect(metrics.quality_volume_samples).toEqual({ value: 64, unit: "count", status: "measured" });
      expect(metrics.quality_component_0_vertex_count).toEqual({ value: 3, unit: "count", status: "measured" });
      expect(metrics.quality_component_0_triangle_count).toEqual({ value: 1, unit: "count", status: "measured" });

      const componentKeys = Object.keys(metrics).filter((key) => /^quality_component_\d+_/.test(key));
      const orderedByIndex = [...componentKeys].sort((left, right) => {
        const leftIndex = Number(left.match(/^quality_component_(\d+)_/)?.[1] ?? -1);
        const rightIndex = Number(right.match(/^quality_component_(\d+)_/)?.[1] ?? -1);
        return leftIndex - rightIndex;
      });
      expect(componentKeys).toEqual(orderedByIndex);
      expect(componentKeys[0]).toMatch(/^quality_component_0_/);
    });
  });
});
