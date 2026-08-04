import { ChitinError } from "./errors.js";
import { parseGlb, type ParsedGlbMesh } from "./glb.js";
import { checkManifold as validateManifold } from "./manifold.js";
import {
  canonicalizeMesh,
  splitMeshComponents,
  type CanonicalizedMesh,
  type TriangleMesh,
} from "./mesh.js";
import { writePhys } from "./phys-writer.js";
import {
  createCompilationReport,
  type CompilationMetric,
  type CompilationProgress,
  type CompilationReport,
  type CompilationRuntime,
  type CompilationStage,
} from "./report.js";
import { evaluateColliderQuality, type ColliderQualityOptions } from "./quality.js";
import { meshBounds, boundsDiagonal, boundsVolume, componentArea } from "./geometry.js";
import type { ConvexHull, DecomposeConfig } from "./types.js";
import {
  DecomposeWorker,
  type DecomposeWorkerOptions,
  type WorkerLike,
} from "./worker-client.js";
import { CHITIN_LITE_VERSION } from "./version.js";

export type GlbInput = ArrayBuffer | ArrayBufferView | Blob | URL | string;

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

export interface InteractiveComponentPolicy {
  /** Disable scene-aware simplification and preserve the legacy per-part behavior. */
  enabled?: boolean;
  /** Fine-detail hull-budget ceiling across all connected components. Default 128. */
  maxHulls?: number;
  /** A component must be below this fraction of the scene diagonal to be simplified. Default 0.2. */
  smallComponentMaxDiagonalRatio?: number;
  /** A component must be below this fraction of the scene AABB volume to be simplified. Default 0.005. */
  smallComponentMaxVolumeRatio?: number;
  /** CoACD threshold used for scene-small components. Default 1.0 (one convex approximation). */
  smallComponentThreshold?: number;
  /** Minimum threshold for detailed parts in the interactive profile. Default 0.10; use 0 to disable. */
  detailedComponentMinThreshold?: number;
  /** Coarsest threshold for a scene-dominant component; smaller parts may use up to twice this value. Default 0.14. */
  importantComponentMaxThreshold?: number;
  /** Maximum enclosed-volume/AABB ratio eligible for importance-weighted threshold protection. Default 0.50. */
  importantComponentMaxOccupancyRatio?: number;
  /** Maximum enclosed-volume/AABB ratio treated as a low-occupancy shell. Default 0.05. */
  hollowShellMaxOccupancyRatio?: number;
  /** Coarsest threshold allowed for low-occupancy shells. Default 0.05. */
  hollowShellMaxThreshold?: number;
  /** Hull capacity reserved for each low-occupancy shell. Default 8. */
  hollowShellMinHulls?: number;
  /** Lower bound for adaptive vertices per emitted hull. Default 8. */
  minHullVertices?: number;
  /** Upper bound for adaptive vertices per emitted hull. Default 96. */
  maxHullVertices?: number;
}

export interface CompileGlbOptions {
  /** Only interactive is accepted until artifact-level profile checks ship. */
  profile?: "interactive";
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

// These are hand-picked defaults for browser interactivity, not values derived
// from a benchmark sweep. `examples/quickstart/benchmarks/sample-quality.json`
// locks the resulting hull counts and coverage on three sample meshes, so
// changing any of these will fail that benchmark; it validates the outcome
// of the current values, not the choice of them. Re-check after a change:
// `npm run benchmark:samples` (run from `examples/quickstart`).
const DEFAULT_MAX_HULLS = 128; // hull ceiling across all components when maxHulls is unset; higher allows more detail.
const DEFAULT_SMALL_DIAGONAL_RATIO = 0.2; // bbox-diagonal fraction under which a component counts as small; higher marks more.
const DEFAULT_SMALL_VOLUME_RATIO = 0.005; // bbox-volume fraction under which a component counts as small; higher marks more.
const DEFAULT_SMALL_THRESHOLD = 1.0; // CoACD threshold for small components; 1.0 accepts the coarsest, cheapest hull.
const DEFAULT_DETAILED_MIN_THRESHOLD = 0.1; // floor on CoACD threshold for non-small components; lower forces tighter, slower fits.
const DEFAULT_IMPORTANT_COMPONENT_MAX_THRESHOLD = 0.14; // ceiling on CoACD threshold for important components; lower keeps them tighter.
const DEFAULT_IMPORTANT_COMPONENT_MAX_OCCUPANCY_RATIO = 0.5; // occupancy ratio at/below which a component qualifies for the importance cap; higher includes more.
const DEFAULT_HOLLOW_SHELL_MAX_OCCUPANCY_RATIO = 0.05; // occupancy ratio at/below which a component is a hollow shell; higher classifies more as hollow.
const DEFAULT_HOLLOW_SHELL_MAX_THRESHOLD = 0.05; // ceiling on CoACD threshold for hollow shells; lower preserves cavities more, costs more.
const DEFAULT_HOLLOW_SHELL_MIN_HULLS = 8; // min hulls reserved for a hollow shell so its cavity survives; higher preserves more, uses more budget.
const DEFAULT_MIN_HULL_VERTICES = 8; // floor on per-hull vertex count for low-fidelity components; lower gives coarser, cheaper hulls.
const DEFAULT_MAX_HULL_VERTICES = 96; // ceiling on per-hull vertex count for high-fidelity components; higher gives smoother, costlier hulls.
const INTERACTIVE_FINE_THRESHOLD = 0.1; // threshold at/below which the detail budget ratio saturates to max; lower widens "fine".
const INTERACTIVE_COARSE_THRESHOLD = 0.6; // threshold at/above which the detail budget ratio saturates to min; higher widens "coarse".
const INTERACTIVE_COARSE_BUDGET_RATIO = 0.7; // min fraction of extra hull budget granted at the coarse end; lower starves coarse requests more.
const INTERACTIVE_MCTS_NODES = 8; // CoACD MCTS nodes in interactive mode; higher searches more branches, slower.
const INTERACTIVE_MCTS_ITERATIONS = 40; // CoACD MCTS iterations in interactive mode; higher searches longer per node, slower.
const INTERACTIVE_MCTS_MAX_DEPTH = 2; // CoACD MCTS depth in interactive mode; higher allows deeper lookahead, slower.
const MAX_CACHED_CONFIGS_PER_COMPONENT = 6; // prior decompose configs cached per component; higher raises memory use, avoids recompute.

interface ResolvedComponentPolicy {
  enabled: boolean;
  maxHulls: number;
  maxHullsCeiling: number;
  maxHullsExplicit: boolean;
  detailBudgetRatio: number;
  smallComponentMaxDiagonalRatio: number;
  smallComponentMaxVolumeRatio: number;
  smallComponentThreshold: number;
  detailedComponentMinThreshold: number;
  importantComponentMaxThreshold: number;
  importantComponentMaxOccupancyRatio: number;
  hollowShellMaxOccupancyRatio: number;
  hollowShellMaxThreshold: number;
  hollowShellMinHulls: number;
  minHullVertices: number;
  maxHullVertices: number;
}

interface ComponentPlan {
  originalIndex: number;
  mesh: TriangleMesh;
  triangleCount: number;
  diagonalRatio: number;
  volumeRatio: number;
  importance: number;
  allocationWeight: number;
  simplified: boolean;
  occupancyRatio: number;
  hollowShell: boolean;
  roundness: number;
  maxHullVertices: number;
  maxThreshold: number;
  maxHulls: number;
}

interface PreparedGlb {
  source: string | null;
  summary: CompileGlbResult["source"];
  processed: CanonicalizedMesh;
  components: TriangleMesh[];
  componentResults: Map<string, ConvexHull[]>;
  componentCacheKeys: Map<number, string[]>;
}

function cacheComponentResult(
  prepared: PreparedGlb,
  componentIndex: number,
  cacheKey: string,
  hulls: ConvexHull[],
): void {
  prepared.componentResults.set(cacheKey, hulls);
  const keys = prepared.componentCacheKeys.get(componentIndex) ?? [];
  const previous = keys.indexOf(cacheKey);
  if (previous >= 0) keys.splice(previous, 1);
  keys.push(cacheKey);
  while (keys.length > MAX_CACHED_CONFIGS_PER_COMPONENT) {
    const expired = keys.shift();
    if (expired) prepared.componentResults.delete(expired);
  }
  prepared.componentCacheKeys.set(componentIndex, keys);
}

function touchComponentResult(prepared: PreparedGlb, componentIndex: number, cacheKey: string): void {
  const keys = prepared.componentCacheKeys.get(componentIndex);
  if (!keys) return;
  const previous = keys.indexOf(cacheKey);
  if (previous < 0 || previous === keys.length - 1) return;
  keys.splice(previous, 1);
  keys.push(cacheKey);
}

function finiteRatio(value: number | undefined, fallback: number, label: string): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < 0 || resolved > 1) {
    throw new ChitinError("INVALID_CONFIG", `${label} must be in [0, 1], got ${resolved}`, {
      stage: "validating-input",
      context: { [label]: resolved },
    });
  }
  return resolved;
}

function interactiveDetailBudgetRatio(threshold: number): number {
  const normalized = Math.max(
    0,
    Math.min(1, (INTERACTIVE_COARSE_THRESHOLD - threshold) /
      (INTERACTIVE_COARSE_THRESHOLD - INTERACTIVE_FINE_THRESHOLD)),
  );
  return INTERACTIVE_COARSE_BUDGET_RATIO +
    (1 - INTERACTIVE_COARSE_BUDGET_RATIO) * normalized;
}

function hullVertexLimit(value: number | undefined, fallback: number, label: string): number {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved < 4) {
    throw new ChitinError("INVALID_CONFIG", `${label} must be an integer of at least 4, got ${resolved}`, {
      stage: "validating-input",
      context: { [label]: resolved },
    });
  }
  return resolved;
}

function resolveComponentPolicy(
  policy: InteractiveComponentPolicy | undefined,
  decompose: DecomposeConfig | undefined,
  componentCount: number,
): ResolvedComponentPolicy {
  if (
    policy?.maxHulls !== undefined &&
    decompose?.maxConvexHull !== undefined &&
    policy.maxHulls !== decompose.maxConvexHull
  ) {
    throw new ChitinError(
      "INVALID_CONFIG",
      "componentPolicy.maxHulls and decompose.maxConvexHull must match when both are set",
      { stage: "validating-input" },
    );
  }
  const explicitlyRequested = policy?.maxHulls ?? decompose?.maxConvexHull;
  let maxHulls = explicitlyRequested ?? Math.max(DEFAULT_MAX_HULLS, componentCount);
  if (!Number.isInteger(maxHulls) || maxHulls === 0 || maxHulls < -1) {
    throw new ChitinError("INVALID_CONFIG", `maxHulls must be -1 or a positive integer, got ${maxHulls}`, {
      stage: "validating-input",
      context: { max_hulls: maxHulls },
    });
  }
  if (maxHulls !== -1 && maxHulls < componentCount) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `maxHulls ${maxHulls} cannot represent ${componentCount} disconnected components`,
      {
        stage: "validating-input",
        suggestion: `Set maxHulls to at least ${componentCount} or use -1 for unlimited hulls.`,
        context: {
          component_count: componentCount,
          max_hulls: maxHulls,
          max_convex_hull: maxHulls,
        },
      },
    );
  }
  if (policy?.enabled === false && explicitlyRequested === undefined) maxHulls = -1;
  const smallThreshold = policy?.smallComponentThreshold ?? DEFAULT_SMALL_THRESHOLD;
  if (!Number.isFinite(smallThreshold) || smallThreshold <= 0 || smallThreshold > 1) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `smallComponentThreshold must be in (0, 1], got ${smallThreshold}`,
      { stage: "validating-input" },
    );
  }
  const minHullVertices = hullVertexLimit(
    policy?.minHullVertices,
    DEFAULT_MIN_HULL_VERTICES,
    "minHullVertices",
  );
  const maxHullVertices = hullVertexLimit(
    policy?.maxHullVertices,
    DEFAULT_MAX_HULL_VERTICES,
    "maxHullVertices",
  );
  if (maxHullVertices < minHullVertices) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `maxHullVertices ${maxHullVertices} must be at least minHullVertices ${minHullVertices}`,
      { stage: "validating-input" },
    );
  }
  return {
    enabled: policy?.enabled ?? true,
    maxHulls,
    maxHullsCeiling: maxHulls,
    maxHullsExplicit: explicitlyRequested !== undefined,
    detailBudgetRatio: 1,
    smallComponentMaxDiagonalRatio: finiteRatio(
      policy?.smallComponentMaxDiagonalRatio,
      DEFAULT_SMALL_DIAGONAL_RATIO,
      "smallComponentMaxDiagonalRatio",
    ),
    smallComponentMaxVolumeRatio: finiteRatio(
      policy?.smallComponentMaxVolumeRatio,
      DEFAULT_SMALL_VOLUME_RATIO,
      "smallComponentMaxVolumeRatio",
    ),
    smallComponentThreshold: smallThreshold,
    detailedComponentMinThreshold: finiteRatio(
      policy?.detailedComponentMinThreshold,
      DEFAULT_DETAILED_MIN_THRESHOLD,
      "detailedComponentMinThreshold",
    ),
    importantComponentMaxThreshold: (() => {
      const value = finiteRatio(
        policy?.importantComponentMaxThreshold,
        DEFAULT_IMPORTANT_COMPONENT_MAX_THRESHOLD,
        "importantComponentMaxThreshold",
      );
      if (value === 0) {
        throw new ChitinError(
          "INVALID_CONFIG",
          "importantComponentMaxThreshold must be greater than 0",
          { stage: "validating-input" },
        );
      }
      return value;
    })(),
    importantComponentMaxOccupancyRatio: finiteRatio(
      policy?.importantComponentMaxOccupancyRatio,
      DEFAULT_IMPORTANT_COMPONENT_MAX_OCCUPANCY_RATIO,
      "importantComponentMaxOccupancyRatio",
    ),
    hollowShellMaxOccupancyRatio: finiteRatio(
      policy?.hollowShellMaxOccupancyRatio,
      DEFAULT_HOLLOW_SHELL_MAX_OCCUPANCY_RATIO,
      "hollowShellMaxOccupancyRatio",
    ),
    hollowShellMaxThreshold: (() => {
      const value = finiteRatio(
        policy?.hollowShellMaxThreshold,
        DEFAULT_HOLLOW_SHELL_MAX_THRESHOLD,
        "hollowShellMaxThreshold",
      );
      if (value === 0) {
        throw new ChitinError(
          "INVALID_CONFIG",
          "hollowShellMaxThreshold must be greater than 0",
          { stage: "validating-input" },
        );
      }
      return value;
    })(),
    hollowShellMinHulls: (() => {
      const value = policy?.hollowShellMinHulls ?? DEFAULT_HOLLOW_SHELL_MIN_HULLS;
      if (!Number.isInteger(value) || value < 1) {
        throw new ChitinError(
          "INVALID_CONFIG",
          `hollowShellMinHulls must be a positive integer, got ${value}`,
          { stage: "validating-input" },
        );
      }
      return value;
    })(),
    minHullVertices,
    maxHullVertices,
  };
}

function bounds(mesh: TriangleMesh): { diagonal: number; volume: number } {
  const meshBoundsResult = meshBounds(mesh.vertices);
  return { diagonal: boundsDiagonal(meshBoundsResult), volume: boundsVolume(meshBoundsResult) };
}

function enclosedVolume(mesh: TriangleMesh): number {
  let signedVolumeTimesSix = 0;
  const originX = mesh.vertices[0] ?? 0;
  const originY = mesh.vertices[1] ?? 0;
  const originZ = mesh.vertices[2] ?? 0;
  for (let offset = 0; offset < mesh.faces.length; offset += 3) {
    const a = mesh.faces[offset] * 3;
    const b = mesh.faces[offset + 1] * 3;
    const c = mesh.faces[offset + 2] * 3;
    // Translate close to the component before summing tetrahedra. Closed-mesh
    // volume is origin independent, and this avoids cancellation for scenes
    // authored far from world zero.
    const ax = mesh.vertices[a] - originX;
    const ay = mesh.vertices[a + 1] - originY;
    const az = mesh.vertices[a + 2] - originZ;
    const bx = mesh.vertices[b] - originX;
    const by = mesh.vertices[b + 1] - originY;
    const bz = mesh.vertices[b + 2] - originZ;
    const cx = mesh.vertices[c] - originX;
    const cy = mesh.vertices[c + 1] - originY;
    const cz = mesh.vertices[c + 2] - originZ;
    signedVolumeTimesSix +=
      ax * (by * cz - bz * cy) +
      ay * (bz * cx - bx * cz) +
      az * (bx * cy - by * cx);
  }
  return Math.abs(signedVolumeTimesSix / 6);
}

function planComponents(
  processed: CanonicalizedMesh,
  components: TriangleMesh[],
  policy: ResolvedComponentPolicy,
  requestedThreshold: number,
): ComponentPlan[] {
  policy.detailBudgetRatio = policy.enabled
    ? interactiveDetailBudgetRatio(requestedThreshold)
    : 1;
  const sceneBounds = bounds(processed);
  const measured = components.map((mesh, originalIndex) => {
    const componentBounds = bounds(mesh);
    const diagonalRatio = sceneBounds.diagonal > 0 ? componentBounds.diagonal / sceneBounds.diagonal : 1;
    const volumeRatio = sceneBounds.volume > 0 ? componentBounds.volume / sceneBounds.volume : 1;
    const triangleCount = mesh.faces.length / 3;
    const simplified =
      policy.enabled &&
      components.length > 1 &&
      triangleCount >= 4 &&
      diagonalRatio <= policy.smallComponentMaxDiagonalRatio &&
      volumeRatio <= policy.smallComponentMaxVolumeRatio;
    const volume = enclosedVolume(mesh);
    const occupancyRatio = componentBounds.volume > 0
      ? volume / componentBounds.volume
      : 1;
    const area = componentArea(mesh);
    // The isoperimetric quotient is stable across tessellation density and
    // scale: 1 for a sphere, lower for flatter or more angular shapes. It is a
    // bounded roundness/curvature proxy, not an artifact-level fit metric.
    const roundness = area > 0
      ? Math.min(1, Math.max(0, (36 * Math.PI * volume ** 2) / area ** 3))
      : 0;
    const importance = Math.max(
      volumeRatio,
      diagonalRatio ** 3,
      triangleCount / (processed.faces.length / 3),
    );
    // Relative size (diagonalRatio, clamped to 1) sets the primary budget;
    // roundness then scales that allocation by 0.5-1x, so flat fins stay cheap
    // while large curved bodies keep enough vertices to describe their silhouette.
    const fidelity = Math.min(1, diagonalRatio) * (0.5 + 0.5 * roundness);
    const adaptiveHullVertices = Math.round(
      policy.minHullVertices +
      (policy.maxHullVertices - policy.minHullVertices) * fidelity,
    );
    const maxHullVertices = Math.max(
      4,
      Math.min(
        mesh.vertices.length / 3,
        simplified ? policy.minHullVertices : adaptiveHullVertices,
      ),
    );
    const hollowShell =
      policy.enabled &&
      !simplified &&
      occupancyRatio <= policy.hollowShellMaxOccupancyRatio;
    return {
      originalIndex,
      mesh,
      triangleCount,
      diagonalRatio,
      volumeRatio,
      importance,
      allocationWeight: 0,
      simplified,
      occupancyRatio,
      hollowShell,
      roundness,
      maxHullVertices,
      maxThreshold: policy.enabled && occupancyRatio <= policy.importantComponentMaxOccupancyRatio
        ? Math.max(
            policy.detailedComponentMinThreshold,
            Math.min(1, policy.importantComponentMaxThreshold * (2 - Math.min(1, importance))),
          )
        : 1,
      maxHulls: simplified ? 1 : hollowShell ? policy.hollowShellMinHulls : -1,
    };
  });

  if (policy.maxHulls !== -1) {
    const requiredMinimum = measured.reduce(
      (sum, component) => sum + (component.hollowShell ? policy.hollowShellMinHulls : 1),
      0,
    );
    if (policy.maxHulls < requiredMinimum) {
      if (policy.maxHullsExplicit) {
        throw new ChitinError(
          "INVALID_CONFIG",
          `maxHulls ${policy.maxHulls} cannot retain interior detail for ${measured.filter((item) => item.hollowShell).length} low-occupancy shell components`,
          {
            stage: "validating-input",
            suggestion: `Set maxHulls to at least ${requiredMinimum}, reduce hollowShellMinHulls, or use -1 for unlimited hulls.`,
            context: {
              component_count: components.length,
              hollow_shell_component_count: measured.filter((item) => item.hollowShell).length,
              max_hulls: policy.maxHulls,
              required_minimum_hulls: requiredMinimum,
            },
          },
        );
      }
      policy.maxHulls = requiredMinimum;
    }
    policy.maxHullsCeiling = policy.maxHulls;
    if (!policy.maxHullsExplicit) {
      policy.maxHulls = requiredMinimum + Math.round(
        (policy.maxHullsCeiling - requiredMinimum) * policy.detailBudgetRatio,
      );
    }
    const extraBudget = policy.maxHulls - requiredMinimum;
    let assigned = 0;
    const detailed = measured.filter((component) => !component.simplified);
    const detailedTriangles = detailed.reduce((sum, component) => sum + component.triangleCount, 0);
    for (const component of detailed) {
      // Importance captures scene scale; the normalized complexity term stops
      // a highly articulated major shell from losing most of its budget merely
      // because a simpler sibling occupies a larger AABB.
      component.allocationWeight = component.importance +
        (detailedTriangles > 0 ? component.triangleCount / detailedTriangles : 0);
    }
    const totalWeight = detailed.reduce((sum, component) => sum + component.allocationWeight, 0);
    const fractions: Array<{ component: ComponentPlan; fraction: number }> = [];
    for (const component of detailed) {
      const exact = totalWeight > 0
        ? (extraBudget * component.allocationWeight) / totalWeight
        : extraBudget / detailed.length;
      const extra = Math.floor(exact);
      const base = component.hollowShell ? policy.hollowShellMinHulls : 1;
      component.maxHulls = base + extra;
      assigned += extra;
      fractions.push({ component, fraction: exact - extra });
    }
    fractions.sort((left, right) =>
      right.fraction - left.fraction || left.component.originalIndex - right.component.originalIndex,
    );
    for (let index = 0; index < extraBudget - assigned && fractions.length > 0; index++) {
      fractions[index % fractions.length].component.maxHulls++;
    }
  }

  return measured.sort((left, right) =>
    Number(left.simplified) - Number(right.simplified) ||
    right.importance - left.importance ||
    left.originalIndex - right.originalIndex,
  );
}

function componentConfig(
  plan: ComponentPlan,
  requested: DecomposeConfig | undefined,
  policy: ResolvedComponentPolicy,
): DecomposeConfig {
  const config: DecomposeConfig = { ...(requested ?? {}) };
  if (policy.enabled) {
    config.mctsNodes ??= INTERACTIVE_MCTS_NODES;
    config.mctsIteration ??= INTERACTIVE_MCTS_ITERATIONS;
    config.mctsMaxDepth ??= INTERACTIVE_MCTS_MAX_DEPTH;
  }
  if (plan.simplified) config.threshold = policy.smallComponentThreshold;
  else if (policy.enabled) {
    config.threshold = Math.max(config.threshold ?? 0.05, policy.detailedComponentMinThreshold);
    config.threshold = Math.min(config.threshold, plan.maxThreshold);
    if (plan.hollowShell) {
      config.threshold = Math.min(config.threshold, policy.hollowShellMaxThreshold);
    }
  }
  if (policy.enabled) {
    config.maxChVertex = Math.min(
      config.maxChVertex ?? policy.maxHullVertices,
      plan.maxHullVertices,
    );
  }
  if (plan.maxHulls !== -1) config.maxConvexHull = plan.maxHulls;
  else if (config.maxConvexHull !== undefined) config.maxConvexHull = -1;
  return config;
}

function configCacheKey(index: number, config: DecomposeConfig, checkManifold: boolean): string {
  return `${index}:${checkManifold}:${JSON.stringify({
    threshold: config.threshold ?? 0.05,
    maxConvexHull: config.maxConvexHull ?? -1,
    prepResolution: config.prepResolution ?? 50,
    sampleResolution: config.sampleResolution ?? 2000,
    mctsNodes: config.mctsNodes ?? 20,
    mctsIteration: config.mctsIteration ?? 150,
    mctsMaxDepth: config.mctsMaxDepth ?? 3,
    maxChVertex: config.maxChVertex ?? 256,
    merge: config.merge ?? true,
  })}`;
}

interface ComponentPlanningResult {
  policy: ResolvedComponentPolicy;
  plans: ComponentPlan[];
  requestedThreshold: number;
  simplifiedCount: number;
  hollowShellCount: number;
  requestedMaxHullVertices: number;
  effectiveHullVertexCaps: number[];
  hullVertexCapByComponent: Map<number, number>;
  adaptedHullVertexCount: number;
  thresholdClamped: boolean;
  detailedThreshold: number;
  guardedHollowShellCount: number;
  importanceGuardedPlans: ComponentPlan[];
  effectiveDetailedThresholds: number[];
}

function planGlbComponents(
  processed: CanonicalizedMesh,
  components: TriangleMesh[],
  options: CompileGlbOptions,
): ComponentPlanningResult {
  const policy = resolveComponentPolicy(options.componentPolicy, options.decompose, components.length);
  const requestedThreshold = options.decompose?.threshold ?? 0.05;
  const plans = planComponents(processed, components, policy, requestedThreshold);
  const simplifiedCount = plans.filter((plan) => plan.simplified).length;
  const hollowShellCount = plans.filter((plan) => plan.hollowShell).length;
  const requestedMaxHullVertices = options.decompose?.maxChVertex ?? 256;
  const effectiveHullVertexCaps = plans.map((plan) => Math.min(
    requestedMaxHullVertices,
    policy.maxHullVertices,
    plan.maxHullVertices,
  ));
  const hullVertexCapByComponent = new Map<number, number>();
  plans.forEach((plan, i) => hullVertexCapByComponent.set(plan.originalIndex, effectiveHullVertexCaps[i]));
  const adaptedHullVertexCount = policy.enabled
    ? effectiveHullVertexCaps.filter((cap) => cap < requestedMaxHullVertices).length
    : 0;
  const thresholdClamped = policy.enabled && requestedThreshold < policy.detailedComponentMinThreshold;
  const detailedThreshold = policy.enabled
    ? Math.max(requestedThreshold, policy.detailedComponentMinThreshold)
    : requestedThreshold;
  const guardedHollowShellCount = detailedThreshold > policy.hollowShellMaxThreshold
    ? hollowShellCount
    : 0;
  const importanceGuardedPlans = policy.enabled
    ? plans.filter((plan) =>
        !plan.simplified &&
        !plan.hollowShell &&
        detailedThreshold > plan.maxThreshold
      )
    : [];
  const effectiveDetailedThresholds = plans
    .filter((plan) => !plan.simplified)
    .map((plan) => Math.min(
      detailedThreshold,
      plan.maxThreshold,
      plan.hollowShell ? policy.hollowShellMaxThreshold : 1,
    ));
  return {
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
  };
}

function evaluateQualityMetrics(
  processed: CanonicalizedMesh,
  hulls: ConvexHull[],
  hullsByComponent: ConvexHull[][],
  quality: true | ColliderQualityOptions,
  plans: ComponentPlan[],
): Record<string, CompilationMetric> {
  const evaluated = evaluateColliderQuality(
    processed,
    hulls,
    quality === true ? {} : quality,
    hullsByComponent,
  );
  const measured = (value: string | number, unit: string): CompilationMetric => ({
    value,
    unit,
    status: "measured",
  });
  const optional = (value: number | null, unit: string): CompilationMetric => ({
    value,
    unit,
    status: value === null ? "not_measured" : "measured",
  });
  const planByIndex = new Map(plans.map((plan) => [plan.originalIndex, plan]));
  const detailedQuality = evaluated.components.filter(
    (component) => !planByIndex.get(component.component_index)?.simplified,
  );
  const detailedSampleCount = detailedQuality.reduce(
    (sum, component) => sum + component.surface_samples,
    0,
  );
  const detailedSurfaceCoverage = detailedSampleCount > 0
    ? detailedQuality.reduce(
        (sum, component) => sum + component.surface_coverage * component.surface_samples,
        0,
      ) / detailedSampleCount
    : null;
  const worstDetailedCoverage = detailedQuality.length > 0
    ? Math.min(...detailedQuality.map((component) => component.surface_coverage))
    : null;
  const qualityMetrics: Record<string, CompilationMetric> = {
    source_surface_coverage: measured(evaluated.source_surface_coverage, "ratio"),
    worst_component_surface_coverage: measured(
      evaluated.worst_component_surface_coverage,
      "ratio",
    ),
    detailed_source_surface_coverage: optional(detailedSurfaceCoverage, "ratio"),
    worst_detailed_component_surface_coverage: optional(
      worstDetailedCoverage,
      "ratio",
    ),
    collider_volume_precision: optional(evaluated.collider_volume_precision, "ratio"),
    false_fill_fraction: optional(evaluated.false_fill_fraction, "ratio"),
    deep_false_fill_fraction: optional(evaluated.deep_false_fill_fraction, "ratio"),
    quality_method: measured(evaluated.method, "method"),
    quality_surface_samples: measured(evaluated.surface_samples, "count"),
    quality_volume_samples: measured(evaluated.volume_samples, "count"),
    quality_collider_volume_samples: measured(evaluated.collider_volume_samples, "count"),
    quality_component_count: measured(evaluated.component_count, "count"),
    quality_volume_tolerance: measured(evaluated.volume_tolerance, "source_unit"),
    quality_surface_tolerance_ratio: measured(evaluated.surface_tolerance_ratio, "ratio"),
    quality_deep_fill_clearance_ratio: measured(
      evaluated.deep_fill_clearance_ratio,
      "ratio",
    ),
  };
  for (const component of evaluated.components) {
    const prefix = `quality_component_${component.component_index}`;
    qualityMetrics[`${prefix}_surface_coverage`] = measured(
      component.surface_coverage,
      "ratio",
    );
    qualityMetrics[`${prefix}_surface_area_fraction`] = measured(
      component.surface_area_fraction,
      "ratio",
    );
    qualityMetrics[`${prefix}_diagonal_ratio`] = measured(
      component.diagonal_ratio,
      "ratio",
    );
    qualityMetrics[`${prefix}_vertex_count`] = measured(component.vertex_count, "count");
    qualityMetrics[`${prefix}_triangle_count`] = measured(component.triangle_count, "count");
    qualityMetrics[`${prefix}_surface_samples`] = measured(component.surface_samples, "count");
    if (component.hull_count !== null) {
      qualityMetrics[`${prefix}_hull_count`] = measured(component.hull_count, "count");
    }
    if (component.collider_triangle_count !== null) {
      qualityMetrics[`${prefix}_collider_triangle_count`] = measured(
        component.collider_triangle_count,
        "count",
      );
    }
    qualityMetrics[`${prefix}_collider_volume_precision`] = optional(
      component.collider_volume_precision,
      "ratio",
    );
    qualityMetrics[`${prefix}_false_fill_fraction`] = optional(
      component.false_fill_fraction,
      "ratio",
    );
    qualityMetrics[`${prefix}_deep_false_fill_fraction`] = optional(
      component.deep_false_fill_fraction,
      "ratio",
    );
    qualityMetrics[`${prefix}_collider_volume_samples`] = optional(
      component.collider_volume_samples,
      "count",
    );
  }
  return qualityMetrics;
}

function cancelled(stage: CompilationStage, message: string): ChitinError {
  return new ChitinError("CANCELLED", message, {
    stage,
    suggestion: "Start a new compilation when ready.",
    retryable: true,
  });
}

function throwIfAborted(signal: AbortSignal | undefined, stage: CompilationStage): void {
  if (signal?.aborted) throw cancelled(stage, "compilation aborted by caller");
}

function mergeSignals(
  caller: AbortSignal | undefined,
  lifecycle: AbortSignal,
): { signal: AbortSignal; cleanup: () => void } {
  if (!caller) return { signal: lifecycle, cleanup: () => {} };
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (caller.aborted || lifecycle.aborted) abort();
  else {
    caller.addEventListener("abort", abort, { once: true });
    lifecycle.addEventListener("abort", abort, { once: true });
  }
  return {
    signal: controller.signal,
    cleanup: () => {
      caller.removeEventListener("abort", abort);
      lifecycle.removeEventListener("abort", abort);
    },
  };
}

function copyView(input: ArrayBufferView): ArrayBuffer {
  const bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  return bytes.slice().buffer;
}

async function readInput(input: GlbInput, signal?: AbortSignal): Promise<{ buffer: ArrayBuffer; source: string | null }> {
  throwIfAborted(signal, "reading-input");
  if (input instanceof ArrayBuffer) return { buffer: input.slice(0), source: null };
  if (ArrayBuffer.isView(input)) return { buffer: copyView(input), source: null };
  if (typeof Blob !== "undefined" && input instanceof Blob) {
    try {
      const buffer = await input.arrayBuffer();
      throwIfAborted(signal, "reading-input");
      return {
        buffer,
        source: typeof File !== "undefined" && input instanceof File ? input.name : null,
      };
    } catch (cause) {
      if (signal?.aborted) throw cancelled("reading-input", "input read aborted by caller");
      throw new ChitinError("LOAD_ERROR", "could not read GLB blob", {
        stage: "reading-input",
        retryable: true,
        cause,
      });
    }
  }
  const url = input instanceof URL ? input.href : String(input);
  let response: Response;
  try {
    response = await fetch(url, { signal });
  } catch (cause) {
    if (signal?.aborted) throw cancelled("reading-input", "GLB fetch aborted by caller");
    throw new ChitinError("LOAD_ERROR", `could not fetch GLB from ${url}`, {
      stage: "reading-input",
      suggestion: "Check the URL, network connection, and server CORS response.",
      retryable: true,
      context: { url },
      cause,
    });
  }
  if (!response.ok) {
    throw new ChitinError("LOAD_ERROR", `GLB request failed with HTTP ${response.status}`, {
      stage: "reading-input",
      suggestion: "Check that the URL exists and is accessible to this browser.",
      retryable: response.status >= 500,
      context: { url, http_status: response.status },
    });
  }
  let buffer: ArrayBuffer;
  try {
    buffer = await response.arrayBuffer();
  } catch (cause) {
    if (signal?.aborted) throw cancelled("reading-input", "GLB fetch aborted by caller");
    throw new ChitinError("LOAD_ERROR", `could not read the GLB response from ${url}`, {
      stage: "reading-input",
      retryable: true,
      context: { url },
      cause,
    });
  }
  throwIfAborted(signal, "reading-input");
  return { buffer, source: url };
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

function runtime(options: ChitinCompilerOptions): CompilationRuntime {
  const wasmVersion = options.wasm.version ?? "unknown";
  return {
    kind: "browser_wasm",
    implementation: "@autarkis/chitin-lite",
    version: CHITIN_LITE_VERSION,
    compiler_version: `${CHITIN_LITE_VERSION}+coacd-wasm${wasmVersion}`,
    dependencies: { "@autarkis/chitin-coacd-wasm": wasmVersion },
  };
}

function sourceSummary(mesh: ParsedGlbMesh): CompileGlbResult["source"] {
  return {
    mesh_count: mesh.mesh_count,
    primitive_count: mesh.primitive_count,
    node_count: mesh.node_count,
    vertex_count: mesh.vertices.length / 3,
    triangle_count: mesh.faces.length / 3,
  };
}

interface ReportAssemblyContext {
  profile: "interactive";
  compilerOptions: ChitinCompilerOptions;
  decomposeConfig: DecomposeConfig | undefined;
  summary: CompileGlbResult["source"];
  processed: CanonicalizedMesh;
  hulls: ConvexHull[];
  hullsByComponent: ConvexHull[][];
  phys: ArrayBuffer;
  timings: Record<string, number>;
  artifactHash: string | null;
  policy: ResolvedComponentPolicy;
  plans: ComponentPlan[];
  requestedThreshold: number;
  simplifiedCount: number;
  hollowShellCount: number;
  requestedMaxHullVertices: number;
  effectiveHullVertexCaps: number[];
  hullVertexCapByComponent: Map<number, number>;
  adaptedHullVertexCount: number;
  thresholdClamped: boolean;
  detailedThreshold: number;
  guardedHollowShellCount: number;
  importanceGuardedPlans: ComponentPlan[];
  effectiveDetailedThresholds: number[];
  componentCount: number;
  workerCount: number;
  qualityMetrics: Record<string, CompilationMetric> | undefined;
  source: string | null;
  cachedBlob: boolean;
  cachedComponentCount: number;
}

function assembleCompilationResult(
  ctx: ReportAssemblyContext,
): { report: CompilationReport; result: CompileGlbResult } {
  const {
    profile,
    compilerOptions,
    decomposeConfig,
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
    componentCount,
    workerCount,
    qualityMetrics,
    source,
    cachedBlob,
    cachedComponentCount,
  } = ctx;

  const report = createCompilationReport({
    profile,
    input: {
      kind: "glb",
      source_vertices: summary.vertex_count,
      processed_vertices: processed.welded_vertex_count,
      mesh_vertices: processed.welded_vertex_count,
    },
    hulls,
    phys_bytes: phys.byteLength,
    timings_ms: timings,
    runtime: runtime(compilerOptions),
    deterministic: null,
    artifact_sha256: artifactHash,
    requested_config: { ...(decomposeConfig ?? {}) },
    effective_config: {
      max_hulls: policy.maxHulls,
      max_hulls_ceiling: policy.maxHullsCeiling,
      detail_budget_ratio: policy.detailBudgetRatio,
      detail_budget_fine_threshold: INTERACTIVE_FINE_THRESHOLD,
      detail_budget_coarse_threshold: INTERACTIVE_COARSE_THRESHOLD,
      detail_budget_coarse_ratio: INTERACTIVE_COARSE_BUDGET_RATIO,
      max_workers: Math.min(workerCount, plans.length),
      component_count: componentCount,
      simplified_component_count: simplifiedCount,
      small_component_max_diagonal_ratio: policy.smallComponentMaxDiagonalRatio,
      small_component_max_volume_ratio: policy.smallComponentMaxVolumeRatio,
      small_component_threshold: policy.smallComponentThreshold,
      detailed_component_min_threshold: policy.detailedComponentMinThreshold,
      detailed_component_threshold: detailedThreshold,
      important_component_max_threshold: policy.importantComponentMaxThreshold,
      important_component_max_occupancy_ratio: policy.importantComponentMaxOccupancyRatio,
      importance_guarded_component_count: importanceGuardedPlans.length,
      effective_component_threshold_min: effectiveDetailedThresholds.length > 0
        ? Math.min(...effectiveDetailedThresholds)
        : null,
      effective_component_threshold_max: effectiveDetailedThresholds.length > 0
        ? Math.max(...effectiveDetailedThresholds)
        : null,
      hollow_shell_component_count: hollowShellCount,
      guarded_hollow_shell_component_count: guardedHollowShellCount,
      hollow_shell_max_occupancy_ratio: policy.hollowShellMaxOccupancyRatio,
      hollow_shell_max_threshold: policy.hollowShellMaxThreshold,
      hollow_shell_min_hulls: policy.hollowShellMinHulls,
      hollow_shell_threshold: hollowShellCount > 0
        ? Math.min(detailedThreshold, policy.hollowShellMaxThreshold)
        : null,
      adaptive_hull_vertices: policy.enabled,
      hull_vertex_roundness_metric: "isoperimetric_quotient",
      min_hull_vertices: policy.minHullVertices,
      max_hull_vertices: policy.maxHullVertices,
      requested_max_hull_vertices: requestedMaxHullVertices,
      effective_component_hull_vertices_min: policy.enabled
        ? Math.min(...effectiveHullVertexCaps)
        : requestedMaxHullVertices,
      effective_component_hull_vertices_max: policy.enabled
        ? Math.max(...effectiveHullVertexCaps)
        : requestedMaxHullVertices,
      effective_component_hull_vertices_mean: policy.enabled
        ? effectiveHullVertexCaps.reduce((sum, cap) => sum + cap, 0) / effectiveHullVertexCaps.length
        : requestedMaxHullVertices,
      component_plans: [...plans]
        .sort((left, right) => left.originalIndex - right.originalIndex)
        .map((plan) => ({
          component_index: plan.originalIndex,
          triangle_count: plan.triangleCount,
          diagonal_ratio: plan.diagonalRatio,
          volume_ratio: plan.volumeRatio,
          occupancy_ratio: plan.occupancyRatio,
          importance: plan.importance,
          allocation_weight: plan.allocationWeight,
          roundness: plan.roundness,
          simplified: plan.simplified,
          hollow_shell: plan.hollowShell,
          max_hulls: plan.maxHulls,
          max_hull_vertices: hullVertexCapByComponent.get(plan.originalIndex)!,
          threshold: componentConfig(plan, decomposeConfig, policy).threshold ?? 0.05,
          output_hulls: hullsByComponent[plan.originalIndex].length,
          output_triangles: hullsByComponent[plan.originalIndex].reduce(
            (sum, hull) => sum + hull.indices.length / 3,
            0,
          ),
        })),
      mcts_nodes: decomposeConfig?.mctsNodes ?? (policy.enabled ? INTERACTIVE_MCTS_NODES : 20),
      mcts_iterations: decomposeConfig?.mctsIteration ?? (policy.enabled ? INTERACTIVE_MCTS_ITERATIONS : 150),
      mcts_max_depth: decomposeConfig?.mctsMaxDepth ?? (policy.enabled ? INTERACTIVE_MCTS_MAX_DEPTH : 3),
    },
    warnings: [
      ...(simplifiedCount > 0 ? [{
          code: "INTERACTIVE_SMALL_COMPONENTS_SIMPLIFIED",
          severity: "info" as const,
          message: `${simplifiedCount} scene-small connected parts use one convex approximation each`,
          context: {
            component_count: componentCount,
            simplified_component_count: simplifiedCount,
            small_component_threshold: policy.smallComponentThreshold,
          },
        }] : []),
      ...(thresholdClamped ? [{
        code: "INTERACTIVE_THRESHOLD_CLAMPED",
        severity: "info" as const,
        message: `Interactive detail threshold ${requestedThreshold} was raised to ${policy.detailedComponentMinThreshold}`,
        context: {
          requested_threshold: requestedThreshold,
          effective_threshold: policy.detailedComponentMinThreshold,
        },
      }] : []),
      ...(guardedHollowShellCount > 0 ? [{
        code: "INTERACTIVE_HOLLOW_SHELL_GUARD",
        severity: "info" as const,
        message: `Limited coarsening on ${guardedHollowShellCount} low-occupancy shell ${guardedHollowShellCount === 1 ? "component" : "components"} because a coarser convex approximation can fill free interior space`,
        context: {
          requested_threshold: requestedThreshold,
          effective_hollow_shell_threshold: policy.hollowShellMaxThreshold,
          hollow_shell_component_count: hollowShellCount,
          hollow_shell_max_occupancy_ratio: policy.hollowShellMaxOccupancyRatio,
        },
      }] : []),
      ...(importanceGuardedPlans.length > 0 ? [{
        code: "INTERACTIVE_IMPORTANCE_GUARD",
        severity: "info" as const,
        message: `Limited coarsening on ${importanceGuardedPlans.length} scene-dominant connected ${importanceGuardedPlans.length === 1 ? "part" : "parts"} so large bodies retain useful silhouette detail`,
        context: {
          requested_threshold: requestedThreshold,
          guarded_component_count: importanceGuardedPlans.length,
          important_component_max_threshold: policy.importantComponentMaxThreshold,
          important_component_max_occupancy_ratio: policy.importantComponentMaxOccupancyRatio,
          effective_threshold_min: Math.min(...importanceGuardedPlans.map((plan) => plan.maxThreshold)),
          effective_threshold_max: Math.max(...importanceGuardedPlans.map((plan) => plan.maxThreshold)),
        },
      }] : []),
      ...(adaptedHullVertexCount > 0 ? [{
        code: "INTERACTIVE_HULL_VERTICES_ADAPTED",
        severity: "info" as const,
        message: `Adapted per-hull vertex limits for ${adaptedHullVertexCount} connected ${adaptedHullVertexCount === 1 ? "component" : "components"} using scene-relative size and geometric roundness`,
        context: {
          adapted_component_count: adaptedHullVertexCount,
          component_count: componentCount,
          requested_max_hull_vertices: requestedMaxHullVertices,
          effective_min_hull_vertices: Math.min(...effectiveHullVertexCaps),
          effective_max_hull_vertices: Math.max(...effectiveHullVertexCaps),
        },
      }] : []),
    ],
    metrics: qualityMetrics,
    artifacts: source ? { source } : {},
  });

  return {
    report,
    result: {
      phys,
      hulls,
      report,
      reuse: {
        prepared_geometry: cachedBlob,
        component_results: cachedComponentCount,
        total_components: componentCount,
      },
      source: summary,
    },
  };
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
        componentResults: new Map(),
        componentCacheKeys: new Map(),
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
        const cachedHulls = prepared.componentResults.get(cacheKey);
        if (cachedHulls) {
          touchComponentResult(prepared, plan.originalIndex, cacheKey);
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
          cacheComponentResult(prepared, plan.originalIndex, cacheKey, result.hulls);
          hullsByComponent[plan.originalIndex] = result.hulls;
          executed++;
          completed++;
          reportProgress();
        } catch (cause) {
          if (firstError === null) firstError = componentError(cause, plan);
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
