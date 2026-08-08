import { ChitinError } from "./errors.js";
import {
  mergeSignals,
  readInput,
  throwIfAborted,
  type GlbInput,
} from "./compiler-input.js";
import { ComponentResultCache } from "./component-cache.js";
import {
  componentConfig,
  configCacheKey,
  planGlbComponents,
  type ComponentPlan,
  type InteractiveComponentPolicy,
  type ResolvedComponentPolicy,
} from "./interactive-policy.js";
import { assembleCompilationResult, sourceSummary } from "./compiler-report.js";
import { parseGlb } from "./glb.js";
import { checkManifold as validateManifold } from "./manifold.js";
import {
  canonicalizeMesh,
  splitMeshComponents,
  type CanonicalizedMesh,
  type TriangleMesh,
} from "./mesh.js";
import { writePhys } from "./phys-writer.js";
import { evaluateQualityMetrics } from "./quality-report.js";
import {
  type CompilationMetric,
  type CompilationProgress,
  type CompilationReport,
  type CompilationStage,
} from "./report.js";
import type { ColliderQualityOptions } from "./quality.js";
import type { ProfileName } from "./shared-constants.js";
import type { ConvexHull, DecomposeConfig } from "./types.js";
import {
  DecomposeWorker,
  type DecomposeWorkerOptions,
  type WorkerLike,
} from "./worker-client.js";

export type { GlbInput } from "./compiler-input.js";
export type { InteractiveComponentPolicy } from "./interactive-policy.js";

type EmitFn = (
  stage: CompilationStage,
  message: string,
  stageStarted?: number,
  detail?: Pick<CompilationProgress, "completed" | "total" | "eta_ms">,
) => void;

export interface WasmAssetUrls {
  js: string | URL;
  wasm: string | URL;
  /** Package/build identity recorded in the compilation report. */
  version?: string;
}

export interface ChitinCompilerOptions {
  wasm: WasmAssetUrls;
  workerUrl?: string | URL;
  /** Test/custom-runtime hook matching DecomposeWorker. */
  workerFactory?: () => WorkerLike;
  /** Maximum simultaneous CoACD workers. Default 2, capped at 4. */
  maxWorkers?: number;
}

export interface CompileGlbOptions {
  /** Only the interactive profile is implemented by the browser compiler. */
  profile?: ProfileName;
  decompose?: DecomposeConfig;
  signal?: AbortSignal;
  /** Check every connected part before CoACD. Default true; opt out only for known-good meshes. */
  checkManifold?: boolean;
  /** Deterministic, scene-aware policy used by the interactive GLB compiler. */
  componentPolicy?: InteractiveComponentPolicy;
  /** Opt-in sampled artifact-fit measurements. Disabled by default because they add verification work. */
  quality?: boolean | ColliderQualityOptions;
  onProgress?: (progress: CompilationProgress) => void;
}

export interface CompileGlbResult {
  phys: ArrayBuffer;
  hulls: ConvexHull[];
  report: CompilationReport;
  /** Work reused by this call when a persistent ChitinCompiler compiles the same Blob/File. */
  reuse: {
    prepared_geometry: boolean;
    component_results: number;
    total_components: number;
  };
  source: {
    mesh_count: number;
    primitive_count: number;
    node_count: number;
    vertex_count: number;
    triangle_count: number;
  };
}

export interface OneShotCompileGlbOptions extends CompileGlbOptions, ChitinCompilerOptions {}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function elapsed(start: number): number {
  return Math.max(0, now() - start);
}

interface PreparedGlb {
  source: string | null;
  summary: CompileGlbResult["source"];
  processed: CanonicalizedMesh;
  components: TriangleMesh[];
  componentCache: ComponentResultCache;
}

async function sha256(buffer: ArrayBuffer): Promise<string | null> {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function writePhysArtifact(hulls: ConvexHull[]): ArrayBuffer {
  try {
    return writePhys(hulls);
  } catch (cause) {
    if (cause instanceof ChitinError && cause.stage === null) {
      throw new ChitinError(cause.code, cause.message, {
        stage: "writing-phys",
        suggestion: "Inspect the generated hull data and retry the compilation.",
        context: cause.context,
        cause,
      });
    }
    throw cause;
  }
}

/**
 * Reusable, worker-backed browser compiler for self-contained GLB 2.0 files.
 * One compilation may run at a time. Reuse keeps the CoACD WASM module warm.
 */
export class ChitinCompiler {
  private readonly workers: DecomposeWorker[];
  private active = false;
  private activeAbort: AbortController | null = null;
  private preparedBlob: { input: Blob; value: PreparedGlb } | null = null;

  constructor(private readonly options: ChitinCompilerOptions) {
    const maxWorkers = options.maxWorkers ?? 2;
    if (!Number.isInteger(maxWorkers) || maxWorkers < 1 || maxWorkers > 4) {
      throw new ChitinError("INVALID_CONFIG", `maxWorkers must be an integer in [1, 4], got ${maxWorkers}`);
    }
    const workerOptions: DecomposeWorkerOptions = {
      workerUrl: options.workerUrl,
      workerFactory: options.workerFactory,
    };
    this.workers = Array.from(
      { length: maxWorkers },
      () => new DecomposeWorker(
        { js: String(options.wasm.js), wasm: String(options.wasm.wasm) },
        workerOptions,
      ),
    );
  }

  async compileGlb(input: GlbInput, options: CompileGlbOptions = {}): Promise<CompileGlbResult> {
    if (this.active) {
      throw new ChitinError("COMPILER_BUSY", "this compiler already has a compilation in progress", {
        suggestion: "Await the active call or create another ChitinCompiler for parallel work.",
        retryable: true,
      });
    }
    this.active = true;
    const lifecycle = new AbortController();
    this.activeAbort = lifecycle;
    const merged = mergeSignals(options.signal, lifecycle.signal);
    try {
      return await this.compileGlbInternal(input, { ...options, signal: merged.signal });
    } finally {
      merged.cleanup();
      if (this.activeAbort === lifecycle) this.activeAbort = null;
      this.active = false;
    }
  }

  private async readAndParseGlb(
    input: GlbInput,
    options: CompileGlbOptions,
    emit: EmitFn,
    timings: Record<string, number>,
  ): Promise<{ prepared: PreparedGlb; cachedBlob: boolean; readStarted: number }> {
    const readStarted = now();
    let prepared: PreparedGlb;
    const cachedBlob = typeof Blob !== "undefined" && input instanceof Blob && this.preparedBlob?.input === input;
    if (cachedBlob) {
      emit("reading-input", "Reusing the selected GLB", readStarted);
      prepared = this.preparedBlob!.value;
      timings.read_input = elapsed(readStarted);
      timings.parse_input = 0;
      emit("parsing-input", "Reusing prepared triangle geometry", readStarted);
    } else {
      emit("reading-input", "Reading GLB input", readStarted);
      const loaded = await readInput(input, options.signal);
      timings.read_input = elapsed(readStarted);

      const parseStarted = now();
      emit("parsing-input", "Parsing active GLB scene", parseStarted);
      const mesh = parseGlb(loaded.buffer);
      // Render-oriented GLBs duplicate positions at UV/normal seams. Restore
      // geometry topology before handing the mesh to CoACD.
      const summary = sourceSummary(mesh);
      const processed = canonicalizeMesh(mesh.vertices, mesh.faces);
      const components = splitMeshComponents(processed.vertices, processed.faces);
      prepared = {
        source: loaded.source,
        summary,
        processed,
        components,
        componentCache: new ComponentResultCache(),
      };
      if (typeof Blob !== "undefined" && input instanceof Blob) {
        this.preparedBlob = { input, value: prepared };
      }
      timings.parse_input = elapsed(parseStarted);
    }
    return { prepared, cachedBlob, readStarted };
  }

  private async runComponentDecomposition(
    prepared: PreparedGlb,
    policy: ResolvedComponentPolicy,
    plans: ComponentPlan[],
    options: CompileGlbOptions,
    emit: EmitFn,
    componentsLength: number,
    localAbort: AbortController,
  ): Promise<{ hulls: ConvexHull[]; hullsByComponent: ConvexHull[][]; cachedComponentCount: number; decomposeMs: number }> {
    const decomposeStarted = now();
    const hullsByComponent: ConvexHull[][] = Array.from({ length: componentsLength }, () => []);
    let cursor = 0;
    let completed = 0;
    let executed = 0;
    let cached = 0;
    let firstError: unknown = null;
    let firstErrorIndex = Infinity;
    const shouldCheckManifold = options.checkManifold ?? true;

    const componentError = (cause: unknown, plan: ComponentPlan): unknown => {
      if (cause instanceof ChitinError && cause.stage === null) {
        const componentContext = {
          ...cause.context,
          component_index: plan.originalIndex,
          component_number: plan.originalIndex + 1,
          component_count: componentsLength,
          component_vertices: plan.mesh.vertices.length / 3,
          component_triangles: plan.mesh.faces.length / 3,
        };
        if (cause.code === "NON_MANIFOLD") {
          const part = `Connected part ${plan.originalIndex + 1} of ${componentsLength}`;
          const issueCounts = [
            [cause.context.boundary_edges, "boundary edge"],
            [cause.context.non_manifold_edges, "non-manifold edge"],
            [cause.context.degenerate_triangles, "degenerate triangle"],
          ]
            .filter(([count]) => typeof count === "number" && count > 0)
            .map(([count, label]) => `${count} ${label}${count === 1 ? "" : "s"}`);
          const details = issueCounts.length > 0
            ? ` Found ${issueCounts.join(", ")}.`
            : ` ${cause.message}`;
          return new ChitinError(
            cause.code,
            `${part} is not a closed solid.${details}`,
            {
              stage: "validating-input",
              suggestion:
                "Use the full Chitin compiler to repair this geometry, or close the mesh in your modelling tool and upload it again.",
              retryable: false,
              context: componentContext,
              cause,
            },
          );
        }
        return new ChitinError(cause.code, cause.message, {
          stage: "decomposing",
          suggestion:
            cause.code === "CANCELLED"
              ? "Start a new compilation when ready."
              : "Check the mesh and decomposition settings, then retry.",
          retryable: cause.code === "CANCELLED" || cause.code === "WORKER_ERROR",
          context: componentContext,
          cause,
        });
      }
      return cause;
    };

    // With parallel workers, validate topology once in source order before
    // scheduling. Otherwise whichever invalid part aborts first would make the
    // reported component nondeterministic. A single-worker compiler keeps the
    // check inside that worker to avoid an extra copy of the same O(faces) pass.
    if (shouldCheckManifold && this.workers.length > 1) {
      for (const plan of [...plans].sort((left, right) => left.originalIndex - right.originalIndex)) {
        try {
          validateManifold(plan.mesh.vertices, plan.mesh.faces);
        } catch (cause) {
          throw componentError(cause, plan);
        }
      }
    }

    const reportProgress = (current?: ComponentPlan) => {
      const stageElapsed = elapsed(decomposeStarted);
      const remaining = componentsLength - completed;
      const parallelism = Math.min(this.workers.length, plans.length);
      const eta = executed >= parallelism
        ? (stageElapsed / executed) * (remaining / parallelism)
        : undefined;
      const currentCopy = current
        ? ` · working on part ${current.originalIndex + 1}`
        : "";
      const cacheCopy = cached > 0 ? ` · ${cached} reused` : "";
      emit(
        "decomposing",
        `Built ${completed} of ${componentsLength} connected parts${currentCopy}${cacheCopy}`,
        decomposeStarted,
        { completed, total: componentsLength, eta_ms: eta },
      );
    };

    emit(
      "loading-wasm",
      `Preparing ${Math.min(this.workers.length, plans.length)} compiler ${Math.min(this.workers.length, plans.length) === 1 ? "worker" : "workers"}`,
      decomposeStarted,
    );
    reportProgress();
    const runQueue = async (worker: DecomposeWorker) => {
      while (!localAbort.signal.aborted) {
        const queueIndex = cursor++;
        if (queueIndex >= plans.length) return;
        const plan = plans[queueIndex];
        const config = componentConfig(plan, options.decompose, policy);
        const cacheKey = configCacheKey(plan.originalIndex, config, shouldCheckManifold);
        const cachedHulls = prepared.componentCache.get(plan.originalIndex, cacheKey);
        if (cachedHulls) {
          hullsByComponent[plan.originalIndex] = cachedHulls;
          cached++;
          completed++;
          reportProgress();
          // Cache hits resolve synchronously (no `await` is crossed above), so
          // without a yield here the same runQueue turn would fall straight
          // through and claim the next cursor slot before any sibling
          // worker's turn is ever scheduled. Yield once via a microtask so
          // siblings get a chance to run and claim slots too.
          await Promise.resolve();
          continue;
        }
        try {
          reportProgress(plan);
          const result = await worker.decompose(
            plan.mesh.vertices.slice(),
            plan.mesh.faces.slice(),
            config,
            {
              signal: localAbort.signal,
              checkManifold: shouldCheckManifold && this.workers.length === 1,
              onState: () => {},
            },
          );
          prepared.componentCache.set(plan.originalIndex, cacheKey, result.hulls);
          hullsByComponent[plan.originalIndex] = result.hulls;
          executed++;
          completed++;
          reportProgress();
        } catch (cause) {
          if (plan.originalIndex < firstErrorIndex) {
            firstError = componentError(cause, plan);
            firstErrorIndex = plan.originalIndex;
          }
          localAbort.abort();
          return;
        }
      }
    };

    await Promise.all(this.workers.slice(0, Math.min(this.workers.length, plans.length)).map(runQueue));
    if (firstError !== null) throw firstError;
    throwIfAborted(options.signal, "decomposing");
    const hulls = hullsByComponent.flat();
    const decomposeMs = elapsed(decomposeStarted);
    return { hulls, hullsByComponent, cachedComponentCount: cached, decomposeMs };
  }

  private async compileGlbInternal(
    input: GlbInput,
    options: CompileGlbOptions,
  ): Promise<CompileGlbResult> {
    const profile = options.profile ?? "interactive";
    if (profile !== "interactive") {
      throw new ChitinError("INVALID_CONFIG", `browser profile ${String(profile)} is not implemented`, {
        stage: "validating-input",
        suggestion: "Use profile: \"interactive\" until outcome-gated browser profiles ship.",
        context: { profile: String(profile) },
      });
    }
    const started = now();
    const timings: Record<string, number> = {};
    const emit = (
      stage: CompilationStage,
      message: string,
      stageStarted = started,
      detail: Pick<CompilationProgress, "completed" | "total" | "eta_ms"> = {},
    ) => {
      options.onProgress?.({ stage, message, elapsed_ms: elapsed(stageStarted), ...detail });
    };

    const localAbort = new AbortController();
    const abortLocal = () => localAbort.abort();
    options.signal?.addEventListener("abort", abortLocal, { once: true });
    try {
      const { prepared, cachedBlob, readStarted } = await this.readAndParseGlb(input, options, emit, timings);
      const { summary, processed, components } = prepared;
      throwIfAborted(options.signal, "validating-input");
      const {
        policy,
        plans,
        requestedThreshold,
        simplifiedCount,
        hollowShellCount,
        requestedMaxHullVertices,
        effectiveHullVertexCaps,
        hullVertexCapByComponent,
        adaptedHullVertexCount,
        thresholdClamped,
        detailedThreshold,
        guardedHollowShellCount,
        importanceGuardedPlans,
        effectiveDetailedThresholds,
      } = planGlbComponents(processed, components, options);
      emit(
        "validating-input",
        simplifiedCount > 0
          ? `Validated ${components.length} connected parts · ${simplifiedCount} scene-small parts use one hull`
          : `Validated ${components.length} connected ${components.length === 1 ? "part" : "parts"}`,
        readStarted,
      );

      const { hulls, hullsByComponent, cachedComponentCount, decomposeMs } = await this.runComponentDecomposition(
        prepared,
        policy,
        plans,
        options,
        emit,
        components.length,
        localAbort,
      );
      timings.decompose = decomposeMs;

      let qualityMetrics: Record<string, CompilationMetric> | undefined;
      if (options.quality) {
        const verifyStarted = now();
        emit("verifying", "Measuring sampled collider fit", verifyStarted);
        qualityMetrics = evaluateQualityMetrics(processed, hulls, hullsByComponent, options.quality, plans);
        timings.verify = elapsed(verifyStarted);
      }

      const writeStarted = now();
      emit("writing-phys", "Writing .phys sidecar", writeStarted);
      const phys = writePhysArtifact(hulls);
      timings.write_phys = elapsed(writeStarted);
      const artifactHash = await sha256(phys);
      throwIfAborted(options.signal, "writing-phys");
      timings.total = elapsed(started);
      const { result } = assembleCompilationResult({
        profile,
        compilerOptions: this.options,
        decomposeConfig: options.decompose,
        summary,
        processed,
        hulls,
        hullsByComponent,
        phys,
        timings,
        artifactHash,
        policy,
        plans,
        requestedThreshold,
        simplifiedCount,
        hollowShellCount,
        requestedMaxHullVertices,
        effectiveHullVertexCaps,
        hullVertexCapByComponent,
        adaptedHullVertexCount,
        thresholdClamped,
        detailedThreshold,
        guardedHollowShellCount,
        importanceGuardedPlans,
        effectiveDetailedThresholds,
        componentCount: components.length,
        workerCount: this.workers.length,
        qualityMetrics,
        source: prepared.source,
        cachedBlob,
        cachedComponentCount,
      });
      emit("done", "Compilation complete", started);
      return result;
    } finally {
      options.signal?.removeEventListener("abort", abortLocal);
    }
  }

  /** Release the worker and cancel any in-flight compilation. */
  terminate(): void {
    this.activeAbort?.abort();
    for (const worker of this.workers) worker.terminate();
  }
}

/** Compile one GLB and release its worker after completion or failure. */
export async function compileGlb(input: GlbInput, options: OneShotCompileGlbOptions): Promise<CompileGlbResult> {
  const compiler = new ChitinCompiler(options);
  try {
    return await compiler.compileGlb(input, options);
  } finally {
    compiler.terminate();
  }
}
