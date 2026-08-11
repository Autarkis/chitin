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
  type ComponentPlanningResult,
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
  preprocessGaussianField,
  proximityFilterMesh,
  extrudeThinShell,
  type GaussianFieldInput,
  type GaussianFieldReconstructionOptions,
} from "./splat-preprocess.js";
import { autoShellThickness } from "./mesh-glb.js";
import {
  type CompilationMetric,
  type CompilationProgress,
  type CompilationReport,
  type CompilationStage,
} from "./report.js";
import type { ColliderQualityOptions } from "./quality.js";
import { BROWSER_PROFILE_NAMES, type BrowserProfileName } from "./shared-constants.js";
import type { ConvexHull, DecomposeConfig } from "./types.js";
import {
  ChitinWorkerClient,
  type ChitinWorkerClientOptions,
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
  /** Poisson WASM module URL. When set, compileGaussianField uses Poisson reconstruction. */
  poissonJs?: string | URL;
  /** Poisson WASM binary URL. Required when poissonJs is set. */
  poissonWasm?: string | URL;
}

export interface ChitinCompilerOptions {
  wasm: WasmAssetUrls;
  workerUrl?: string | URL;
  /** Test/custom-runtime hook matching ChitinWorkerClient. */
  workerFactory?: () => WorkerLike;
  /** Maximum simultaneous CoACD workers. Default 2, capped at 4. */
  maxWorkers?: number;
}

export interface CompileGlbOptions {
  /** Only the interactive profile is implemented by the browser compiler. */
  profile?: BrowserProfileName;
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

export interface CompileGaussianFieldOptions extends CompileGlbOptions {
  reconstruction?: GaussianFieldReconstructionOptions;
  /** Poisson octree depth. Defaults to 7. */
  poissonDepth?: number;
  /** Low-density vertex trim threshold. Defaults to 0.1. */
  densityQuantile?: number;
  /** Splat surface densification ratio. Defaults to 0.5. */
  surfaceRatio?: number;
  /** Max distance ratio for proximity filtering. 0 = disabled. Defaults to 3. */
  surfaceProximityFilter?: number;
  /** Thin shell extrusion thickness. 0 = disabled. Defaults to auto. */
  thinShellThickness?: number;
}

export interface OneShotCompileGaussianFieldOptions
  extends CompileGaussianFieldOptions, ChitinCompilerOptions {}

export interface OneShotCompileGlbOptions extends CompileGlbOptions, ChitinCompilerOptions {}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function elapsed(start: number): number {
  return Math.max(0, now() - start);
}

interface PreparedGeometry {
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

function enrichComponentError(cause: unknown, plan: ComponentPlan, componentCount: number): unknown {
  if (cause instanceof ChitinError && cause.stage === null) {
    const componentContext = {
      ...cause.context,
      component_index: plan.originalIndex,
      component_number: plan.originalIndex + 1,
      component_count: componentCount,
      component_vertices: plan.mesh.vertices.length / 3,
      component_triangles: plan.mesh.faces.length / 3,
    };
    if (cause.code === "NON_MANIFOLD") {
      const part = `Connected part ${plan.originalIndex + 1} of ${componentCount}`;
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
}

/**
 * Reusable, worker-backed browser compiler for self-contained GLB 2.0 files.
 * One compilation may run at a time. Reuse keeps the CoACD WASM module warm.
 */
export class ChitinCompiler {
  private readonly workers: ChitinWorkerClient[];
  private active = false;
  private activeAbort: AbortController | null = null;
  private preparedBlob: { input: Blob; value: PreparedGeometry } | null = null;

  constructor(private readonly options: ChitinCompilerOptions) {
    const maxWorkers = options.maxWorkers ?? 2;
    if (!Number.isInteger(maxWorkers) || maxWorkers < 1 || maxWorkers > 4) {
      throw new ChitinError("INVALID_CONFIG", `maxWorkers must be an integer in [1, 4], got ${maxWorkers}`);
    }
    const workerOptions: ChitinWorkerClientOptions = {
      workerUrl: options.workerUrl,
      workerFactory: options.workerFactory,
    };
    this.workers = Array.from(
      { length: maxWorkers },
      () => new ChitinWorkerClient(
        { js: String(options.wasm.js), wasm: String(options.wasm.wasm) },
        workerOptions,
        options.wasm.poissonJs && options.wasm.poissonWasm
          ? { js: String(options.wasm.poissonJs), wasm: String(options.wasm.poissonWasm) }
          : undefined,
      ),
    );
  }

  async compileGlb(input: GlbInput, options: CompileGlbOptions = {}): Promise<CompileGlbResult> {
    return this.compileGlbInternal(input, options);
  }

  async compileGaussianField(
    input: GaussianFieldInput,
    options: CompileGaussianFieldOptions = {},
  ): Promise<CompileGlbResult> {
    if (!this.options.wasm.poissonJs || !this.options.wasm.poissonWasm) {
      throw new ChitinError(
        "INVALID_CONFIG",
        "compileGaussianField requires poissonJs and poissonWasm URLs in wasm options",
      );
    }

    const {
      reconstruction,
      poissonDepth = 7,
      densityQuantile = 0.1,
      surfaceRatio = 0.5,
      surfaceProximityFilter = 3,
      thinShellThickness = 0,
      ...compileOptions
    } = options;

    const started = now();
    const timings: Record<string, number> = {};
    const emit: EmitFn = (stage, message, stageStarted = started, detail = {}) => {
      options.onProgress?.({ stage, message, elapsed_ms: elapsed(stageStarted), ...detail });
    };

    return this.runCompilation(options.signal, async (mergedSignal, localAbort) => {
      const mergedCompileOptions = { ...compileOptions, signal: mergedSignal };
      emit("reading-input", "Preprocessing Gaussian field");
      const logScale = false;
      const preprocessed = preprocessGaussianField(input, {
        surfaceRatio,
        minOpacity: reconstruction?.minOpacity ?? 0.2,
        logScale,
      });
      const inputPositions = new Float64Array(preprocessed.positions);

      emit("parsing-input", "Reconstructing surface via Poisson");
      const worker = this.workers[0];
      const mesh = await worker.poissonReconstruct(
        preprocessed.positions,
        preprocessed.normals,
        {
          depth: poissonDepth,
          densityQuantile,
          signal: mergedSignal,
        },
      );
      timings.poisson = elapsed(started);

      let filtered: TriangleMesh = {
        vertices: new Float64Array(mesh.vertices),
        faces: new Int32Array(mesh.faces),
      };

      if (surfaceProximityFilter > 0) {
        filtered = proximityFilterMesh(
          filtered.vertices as Float64Array,
          filtered.faces as Int32Array,
          inputPositions,
          surfaceProximityFilter,
        );
      }

      if (thinShellThickness !== 0) {
        filtered = extrudeThinShell(
          filtered.vertices as Float64Array,
          filtered.faces as Int32Array,
          thinShellThickness > 0
            ? thinShellThickness
            : autoShellThickness(filtered.vertices as Float64Array),
        );
      }

      const processed = canonicalizeMesh(
        filtered.vertices as Float64Array,
        filtered.faces as Int32Array,
      );
      const components = splitMeshComponents(
        processed.vertices as Float64Array,
        processed.faces as Int32Array,
      );
      const splatCount = input.centers.length / 3;
      const prepared: PreparedGeometry = {
        source: "gaussian-field",
        summary: {
          mesh_count: 1,
          primitive_count: 1,
          node_count: 1,
          vertex_count: splatCount,
          triangle_count: processed.faces.length / 3,
        },
        processed,
        components,
        componentCache: new ComponentResultCache(),
      };
      timings.prepare = elapsed(started);

      return this.compileMeshInternal(prepared, mergedCompileOptions, emit, timings, started, localAbort);
    });
  }

  private async readAndParseGlb(
    input: GlbInput,
    options: CompileGlbOptions,
    emit: EmitFn,
    timings: Record<string, number>,
  ): Promise<{ prepared: PreparedGeometry; cachedBlob: boolean; readStarted: number }> {
    const readStarted = now();
    let prepared: PreparedGeometry;
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

  private validateComponentManifolds(
    plans: ComponentPlan[],
    componentCount: number,
  ): void {
    for (const plan of [...plans].sort((left, right) => left.originalIndex - right.originalIndex)) {
      try {
        validateManifold(plan.mesh.vertices, plan.mesh.faces);
      } catch (cause) {
        throw enrichComponentError(cause, plan, componentCount);
      }
    }
  }

  private async decomposeComponents(
    prepared: PreparedGeometry,
    planning: ComponentPlanningResult,
    options: CompileGlbOptions,
    emit: EmitFn,
    componentCount: number,
    localAbort: AbortController,
  ): Promise<{ hulls: ConvexHull[]; hullsByComponent: ConvexHull[][]; cachedComponentCount: number; decomposeMs: number }> {
    const { policy, plans } = planning;
    const decomposeStarted = now();
    const hullsByComponent: ConvexHull[][] = Array.from({ length: componentCount }, () => []);
    let cursor = 0;
    let completed = 0;
    let executed = 0;
    let cached = 0;
    let firstError: unknown = null;
    let firstErrorIndex = Infinity;
    const shouldCheckManifold = options.checkManifold ?? true;

    const reportProgress = (current?: ComponentPlan) => {
      const stageElapsed = elapsed(decomposeStarted);
      const remaining = componentCount - completed;
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
        `Built ${completed} of ${componentCount} connected parts${currentCopy}${cacheCopy}`,
        decomposeStarted,
        { completed, total: componentCount, eta_ms: eta },
      );
    };

    emit(
      "loading-wasm",
      `Preparing ${Math.min(this.workers.length, plans.length)} compiler ${Math.min(this.workers.length, plans.length) === 1 ? "worker" : "workers"}`,
      decomposeStarted,
    );
    reportProgress();
    const runQueue = async (worker: ChitinWorkerClient) => {
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
            firstError = enrichComponentError(cause, plan, componentCount);
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

  private async runCompilation<T>(
    signal: AbortSignal | undefined,
    work: (mergedSignal: AbortSignal, localAbort: AbortController) => Promise<T>,
  ): Promise<T> {
    if (this.active) {
      throw new ChitinError("COMPILER_BUSY", "A compilation is already in progress.", {
        stage: "reading-input",
        suggestion: "Await the active call or create another ChitinCompiler for parallel work.",
        retryable: true,
      });
    }
    this.active = true;
    const lifecycle = new AbortController();
    this.activeAbort = lifecycle;
    const merged = mergeSignals(signal, lifecycle.signal);
    const localAbort = new AbortController();
    const abortLocal = () => localAbort.abort();
    merged.signal.addEventListener("abort", abortLocal, { once: true });
    try {
      return await work(merged.signal, localAbort);
    } finally {
      merged.signal.removeEventListener("abort", abortLocal);
      merged.cleanup();
      this.active = false;
      this.activeAbort = null;
    }
  }

  private async compileMeshInternal(
    prepared: PreparedGeometry,
    options: CompileGlbOptions,
    emit: EmitFn,
    timings: Record<string, number>,
    started: number,
    localAbort: AbortController,
    cachedBlob = false,
  ): Promise<CompileGlbResult> {
    const profile: BrowserProfileName = options.profile ?? "interactive";
    const { summary, processed, components } = prepared;
    throwIfAborted(options.signal, "validating-input");
    const planning = planGlbComponents(processed, components, options);
    emit(
      "validating-input",
      planning.simplifiedCount > 0
        ? `Validated ${components.length} connected parts · ${planning.simplifiedCount} scene-small parts use one hull`
        : `Validated ${components.length} connected ${components.length === 1 ? "part" : "parts"}`,
      started,
    );

    if ((options.checkManifold ?? true) && this.workers.length > 1) {
      this.validateComponentManifolds(planning.plans, components.length);
    }

    const { hulls, hullsByComponent, cachedComponentCount, decomposeMs } = await this.decomposeComponents(
      prepared,
      planning,
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
      qualityMetrics = evaluateQualityMetrics(processed, hulls, hullsByComponent, options.quality, planning.plans);
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
      ...planning,
      componentCount: components.length,
      workerCount: this.workers.length,
      qualityMetrics,
      source: prepared.source,
      cachedBlob,
      cachedComponentCount,
    });
    emit("done", "Compilation complete", started);
    return result;
  }

  private async compileGlbInternal(
    input: GlbInput,
    options: CompileGlbOptions,
  ): Promise<CompileGlbResult> {
    const profile: BrowserProfileName = options.profile ?? "interactive";
    if (!(BROWSER_PROFILE_NAMES as readonly string[]).includes(profile)) {
      throw new ChitinError(
        "INVALID_CONFIG",
        `Browser profile "${String(profile)}" is not supported. Use one of: ${BROWSER_PROFILE_NAMES.join(", ")}.`,
        {
          stage: "validating-input",
          suggestion: `Pass profile: "${BROWSER_PROFILE_NAMES[0]}" or omit the profile option.`,
          context: { profile: String(profile) },
        },
      );
    }
    const started = now();
    const timings: Record<string, number> = {};
    const emit: EmitFn = (stage, message, stageStarted = started, detail = {}) => {
      options.onProgress?.({ stage, message, elapsed_ms: elapsed(stageStarted), ...detail });
    };

    return this.runCompilation(options.signal, async (mergedSignal, localAbort) => {
      const mergedOptions = { ...options, signal: mergedSignal };
      const { prepared, cachedBlob } = await this.readAndParseGlb(input, mergedOptions, emit, timings);
      return this.compileMeshInternal(prepared, mergedOptions, emit, timings, started, localAbort, cachedBlob);
    });
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

export async function compileGaussianField(
  input: GaussianFieldInput,
  options: OneShotCompileGaussianFieldOptions,
): Promise<CompileGlbResult> {
  const compiler = new ChitinCompiler(options);
  try {
    return await compiler.compileGaussianField(input, options);
  } finally {
    compiler.terminate();
  }
}
