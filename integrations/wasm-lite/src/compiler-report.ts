import type { ChitinCompilerOptions, CompileGlbResult } from "./compiler.js";
import { DECOMPOSE_DEFAULTS } from "./defaults.js";
import type { ParsedGlbMesh } from "./glb.js";
import {
  componentConfig,
  INTERACTIVE_COARSE_BUDGET_RATIO,
  INTERACTIVE_COARSE_THRESHOLD,
  INTERACTIVE_FINE_THRESHOLD,
  INTERACTIVE_MCTS_ITERATIONS,
  INTERACTIVE_MCTS_MAX_DEPTH,
  INTERACTIVE_MCTS_NODES,
  type ComponentPlan,
  type ResolvedComponentPolicy,
} from "./interactive-policy.js";
import type { CanonicalizedMesh } from "./mesh.js";
import {
  createCompilationReport,
  type CompilationMetric,
  type CompilationReport,
  type CompilationRuntime,
  type CompilationWarning,
} from "./report.js";
import type { ConvexHull, DecomposeConfig } from "./types.js";
import { CHITIN_LITE_VERSION } from "./version.js";

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

export function sourceSummary(mesh: ParsedGlbMesh): CompileGlbResult["source"] {
  return {
    mesh_count: mesh.mesh_count,
    primitive_count: mesh.primitive_count,
    node_count: mesh.node_count,
    vertex_count: mesh.vertices.length / 3,
    triangle_count: mesh.faces.length / 3,
  };
}

export interface ReportAssemblyContext {
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

function reportedComponentPlans(ctx: ReportAssemblyContext): Record<string, unknown>[] {
  return [...ctx.plans]
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
      max_hull_vertices: ctx.hullVertexCapByComponent.get(plan.originalIndex)!,
      threshold:
        componentConfig(plan, ctx.decomposeConfig, ctx.policy).threshold ??
        DECOMPOSE_DEFAULTS.threshold,
      output_hulls: ctx.hullsByComponent[plan.originalIndex].length,
      output_triangles: ctx.hullsByComponent[plan.originalIndex].reduce(
        (sum, hull) => sum + hull.indices.length / 3,
        0,
      ),
    }));
}

function effectiveConfig(ctx: ReportAssemblyContext): Record<string, unknown> {
  const {
    policy,
    plans,
    workerCount,
    componentCount,
    simplifiedCount,
    detailedThreshold,
    importanceGuardedPlans,
    effectiveDetailedThresholds,
    hollowShellCount,
    guardedHollowShellCount,
    requestedMaxHullVertices,
    effectiveHullVertexCaps,
    decomposeConfig,
  } = ctx;
  return {
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
    effective_component_threshold_min:
      effectiveDetailedThresholds.length > 0 ? Math.min(...effectiveDetailedThresholds) : null,
    effective_component_threshold_max:
      effectiveDetailedThresholds.length > 0 ? Math.max(...effectiveDetailedThresholds) : null,
    hollow_shell_component_count: hollowShellCount,
    guarded_hollow_shell_component_count: guardedHollowShellCount,
    hollow_shell_max_occupancy_ratio: policy.hollowShellMaxOccupancyRatio,
    hollow_shell_max_threshold: policy.hollowShellMaxThreshold,
    hollow_shell_min_hulls: policy.hollowShellMinHulls,
    hollow_shell_threshold:
      hollowShellCount > 0 ? Math.min(detailedThreshold, policy.hollowShellMaxThreshold) : null,
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
      ? effectiveHullVertexCaps.reduce((sum, cap) => sum + cap, 0) /
        effectiveHullVertexCaps.length
      : requestedMaxHullVertices,
    component_plans: reportedComponentPlans(ctx),
    mcts_nodes:
      decomposeConfig?.mctsNodes ??
      (policy.enabled ? INTERACTIVE_MCTS_NODES : DECOMPOSE_DEFAULTS.mctsNodes),
    mcts_iterations:
      decomposeConfig?.mctsIteration ??
      (policy.enabled ? INTERACTIVE_MCTS_ITERATIONS : DECOMPOSE_DEFAULTS.mctsIteration),
    mcts_max_depth:
      decomposeConfig?.mctsMaxDepth ??
      (policy.enabled ? INTERACTIVE_MCTS_MAX_DEPTH : DECOMPOSE_DEFAULTS.mctsMaxDepth),
  };
}

function interactiveWarnings(ctx: ReportAssemblyContext): CompilationWarning[] {
  const warnings: CompilationWarning[] = [];
  const {
    simplifiedCount,
    componentCount,
    policy,
    thresholdClamped,
    requestedThreshold,
    guardedHollowShellCount,
    hollowShellCount,
    importanceGuardedPlans,
    adaptedHullVertexCount,
    requestedMaxHullVertices,
    effectiveHullVertexCaps,
  } = ctx;

  if (simplifiedCount > 0) {
    warnings.push({
      code: "INTERACTIVE_SMALL_COMPONENTS_SIMPLIFIED",
      severity: "info",
      message: `${simplifiedCount} scene-small connected parts use one convex approximation each`,
      context: {
        component_count: componentCount,
        simplified_component_count: simplifiedCount,
        small_component_threshold: policy.smallComponentThreshold,
      },
    });
  }
  if (thresholdClamped) {
    warnings.push({
      code: "INTERACTIVE_THRESHOLD_CLAMPED",
      severity: "info",
      message: `Interactive detail threshold ${requestedThreshold} was raised to ${policy.detailedComponentMinThreshold}`,
      context: {
        requested_threshold: requestedThreshold,
        effective_threshold: policy.detailedComponentMinThreshold,
      },
    });
  }
  if (guardedHollowShellCount > 0) {
    warnings.push({
      code: "INTERACTIVE_HOLLOW_SHELL_GUARD",
      severity: "info",
      message: `Limited coarsening on ${guardedHollowShellCount} low-occupancy shell ${guardedHollowShellCount === 1 ? "component" : "components"} because a coarser convex approximation can fill free interior space`,
      context: {
        requested_threshold: requestedThreshold,
        effective_hollow_shell_threshold: policy.hollowShellMaxThreshold,
        hollow_shell_component_count: hollowShellCount,
        hollow_shell_max_occupancy_ratio: policy.hollowShellMaxOccupancyRatio,
      },
    });
  }
  if (importanceGuardedPlans.length > 0) {
    warnings.push({
      code: "INTERACTIVE_IMPORTANCE_GUARD",
      severity: "info",
      message: `Limited coarsening on ${importanceGuardedPlans.length} low-occupancy connected ${importanceGuardedPlans.length === 1 ? "part" : "parts"} so large bodies retain useful silhouette detail`,
      context: {
        requested_threshold: requestedThreshold,
        guarded_component_count: importanceGuardedPlans.length,
        important_component_max_threshold: policy.importantComponentMaxThreshold,
        important_component_max_occupancy_ratio: policy.importantComponentMaxOccupancyRatio,
        effective_threshold_min: Math.min(...importanceGuardedPlans.map((plan) => plan.maxThreshold)),
        effective_threshold_max: Math.max(...importanceGuardedPlans.map((plan) => plan.maxThreshold)),
      },
    });
  }
  if (adaptedHullVertexCount > 0) {
    warnings.push({
      code: "INTERACTIVE_HULL_VERTICES_ADAPTED",
      severity: "info",
      message: `Adapted per-hull vertex limits for ${adaptedHullVertexCount} connected ${adaptedHullVertexCount === 1 ? "component" : "components"} using scene-relative size and geometric roundness`,
      context: {
        adapted_component_count: adaptedHullVertexCount,
        component_count: componentCount,
        requested_max_hull_vertices: requestedMaxHullVertices,
        effective_min_hull_vertices: Math.min(...effectiveHullVertexCaps),
        effective_max_hull_vertices: Math.max(...effectiveHullVertexCaps),
      },
    });
  }
  return warnings;
}

export function assembleCompilationResult(
  ctx: ReportAssemblyContext,
): { report: CompilationReport; result: CompileGlbResult } {
  const {
    profile,
    compilerOptions,
    decomposeConfig,
    summary,
    processed,
    hulls,
    phys,
    timings,
    artifactHash,
    componentCount,
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
    effective_config: effectiveConfig(ctx),
    warnings: interactiveWarnings(ctx),
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
