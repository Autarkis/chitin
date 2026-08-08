import { DECOMPOSE_DEFAULTS, resolveDecomposeConfig } from "./defaults.js";
import { ChitinError } from "./errors.js";
import { boundsDiagonal, boundsVolume, componentArea, meshBounds } from "./geometry.js";
import type { CanonicalizedMesh, TriangleMesh } from "./mesh.js";
import type { DecomposeConfig } from "./types.js";

export interface InteractiveComponentPolicy {
  /** Disable scene-aware simplification and use uniform per-component settings. */
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

export interface InteractiveCompileOptions {
  decompose?: DecomposeConfig;
  componentPolicy?: InteractiveComponentPolicy;
}

export interface ResolvedComponentPolicy {
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

export interface ComponentPlan {
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

export interface ComponentPlanningResult {
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

export const INTERACTIVE_FINE_THRESHOLD = 0.1;
export const INTERACTIVE_COARSE_THRESHOLD = 0.6;
export const INTERACTIVE_COARSE_BUDGET_RATIO = 0.7;
export const INTERACTIVE_MCTS_NODES = 8;
export const INTERACTIVE_MCTS_ITERATIONS = 40;
export const INTERACTIVE_MCTS_MAX_DEPTH = 2;

const DEFAULT_MAX_HULLS = 128;
const DEFAULT_SMALL_DIAGONAL_RATIO = 0.2;
const DEFAULT_SMALL_VOLUME_RATIO = 0.005;
const DEFAULT_SMALL_THRESHOLD = 1.0;
const DEFAULT_DETAILED_MIN_THRESHOLD = 0.1;
const DEFAULT_IMPORTANT_COMPONENT_MAX_THRESHOLD = 0.14;
const DEFAULT_IMPORTANT_COMPONENT_MAX_OCCUPANCY_RATIO = 0.5;
const DEFAULT_HOLLOW_SHELL_MAX_OCCUPANCY_RATIO = 0.05;
const DEFAULT_HOLLOW_SHELL_MAX_THRESHOLD = 0.05;
const DEFAULT_HOLLOW_SHELL_MIN_HULLS = 8;
const DEFAULT_MIN_HULL_VERTICES = 8;
const DEFAULT_MAX_HULL_VERTICES = 96;

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

function detailBudgetRatio(threshold: number): number {
  const normalized = Math.max(
    0,
    Math.min(
      1,
      (INTERACTIVE_COARSE_THRESHOLD - threshold) /
        (INTERACTIVE_COARSE_THRESHOLD - INTERACTIVE_FINE_THRESHOLD),
    ),
  );
  return INTERACTIVE_COARSE_BUDGET_RATIO +
    (1 - INTERACTIVE_COARSE_BUDGET_RATIO) * normalized;
}

function hullVertexLimit(value: number | undefined, fallback: number, label: string): number {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved < 4) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `${label} must be an integer of at least 4, got ${resolved}`,
      { stage: "validating-input", context: { [label]: resolved } },
    );
  }
  return resolved;
}

function resolveComponentPolicy(
  policy: InteractiveComponentPolicy | undefined,
  componentCount: number,
): ResolvedComponentPolicy {
  const explicitlyRequested = policy?.maxHulls;
  let maxHulls = explicitlyRequested ?? Math.max(DEFAULT_MAX_HULLS, componentCount);
  if (!Number.isInteger(maxHulls) || maxHulls === 0 || maxHulls < -1) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `maxHulls must be -1 or a positive integer, got ${maxHulls}`,
      { stage: "validating-input", context: { max_hulls: maxHulls } },
    );
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
  const importantComponentMaxThreshold = finiteRatio(
    policy?.importantComponentMaxThreshold,
    DEFAULT_IMPORTANT_COMPONENT_MAX_THRESHOLD,
    "importantComponentMaxThreshold",
  );
  if (importantComponentMaxThreshold === 0) {
    throw new ChitinError(
      "INVALID_CONFIG",
      "importantComponentMaxThreshold must be greater than 0",
      { stage: "validating-input" },
    );
  }
  const hollowShellMaxThreshold = finiteRatio(
    policy?.hollowShellMaxThreshold,
    DEFAULT_HOLLOW_SHELL_MAX_THRESHOLD,
    "hollowShellMaxThreshold",
  );
  if (hollowShellMaxThreshold === 0) {
    throw new ChitinError(
      "INVALID_CONFIG",
      "hollowShellMaxThreshold must be greater than 0",
      { stage: "validating-input" },
    );
  }
  const hollowShellMinHulls =
    policy?.hollowShellMinHulls ?? DEFAULT_HOLLOW_SHELL_MIN_HULLS;
  if (!Number.isInteger(hollowShellMinHulls) || hollowShellMinHulls < 1) {
    throw new ChitinError(
      "INVALID_CONFIG",
      `hollowShellMinHulls must be a positive integer, got ${hollowShellMinHulls}`,
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
    importantComponentMaxThreshold,
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
    hollowShellMaxThreshold,
    hollowShellMinHulls,
    minHullVertices,
    maxHullVertices,
  };
}

function bounds(mesh: TriangleMesh): { diagonal: number; volume: number } {
  const result = meshBounds(mesh.vertices);
  return { diagonal: boundsDiagonal(result), volume: boundsVolume(result) };
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

function measureComponents(
  processed: CanonicalizedMesh,
  components: TriangleMesh[],
  policy: ResolvedComponentPolicy,
): ComponentPlan[] {
  const sceneBounds = bounds(processed);
  return components.map((mesh, originalIndex) => {
    const componentBounds = bounds(mesh);
    const diagonalRatio =
      sceneBounds.diagonal > 0 ? componentBounds.diagonal / sceneBounds.diagonal : 1;
    const volumeRatio =
      sceneBounds.volume > 0 ? componentBounds.volume / sceneBounds.volume : 1;
    const triangleCount = mesh.faces.length / 3;
    const simplified =
      policy.enabled &&
      components.length > 1 &&
      triangleCount >= 4 &&
      diagonalRatio <= policy.smallComponentMaxDiagonalRatio &&
      volumeRatio <= policy.smallComponentMaxVolumeRatio;
    const volume = enclosedVolume(mesh);
    const occupancyRatio = componentBounds.volume > 0 ? volume / componentBounds.volume : 1;
    const area = componentArea(mesh);
    const roundness =
      area > 0
        ? Math.min(1, Math.max(0, (36 * Math.PI * volume ** 2) / area ** 3))
        : 0;
    const importance = Math.max(
      volumeRatio,
      diagonalRatio ** 3,
      triangleCount / (processed.faces.length / 3),
    );
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
      maxThreshold:
        policy.enabled && occupancyRatio <= policy.importantComponentMaxOccupancyRatio
          ? Math.max(
              policy.detailedComponentMinThreshold,
              Math.min(
                1,
                policy.importantComponentMaxThreshold * (2 - Math.min(1, importance)),
              ),
            )
          : 1,
      maxHulls: simplified ? 1 : hollowShell ? policy.hollowShellMinHulls : -1,
    };
  });
}

function allocateHullBudget(
  measured: ComponentPlan[],
  policy: ResolvedComponentPolicy,
): void {
  if (policy.maxHulls === -1) return;
  const hollowShellCount = measured.filter((component) => component.hollowShell).length;
  const requiredMinimum = measured.reduce(
    (sum, component) => sum + (component.hollowShell ? policy.hollowShellMinHulls : 1),
    0,
  );
  if (policy.maxHulls < requiredMinimum) {
    if (policy.maxHullsExplicit) {
      throw new ChitinError(
        "INVALID_CONFIG",
        `maxHulls ${policy.maxHulls} cannot retain interior detail for ${hollowShellCount} low-occupancy shell components`,
        {
          stage: "validating-input",
          suggestion: `Set maxHulls to at least ${requiredMinimum}, reduce hollowShellMinHulls, or use -1 for unlimited hulls.`,
          context: {
            component_count: measured.length,
            hollow_shell_component_count: hollowShellCount,
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
    policy.maxHulls =
      requiredMinimum +
      Math.round((policy.maxHullsCeiling - requiredMinimum) * policy.detailBudgetRatio);
  }

  const extraBudget = policy.maxHulls - requiredMinimum;
  let assigned = 0;
  const detailed = measured.filter((component) => !component.simplified);
  const detailedTriangles = detailed.reduce(
    (sum, component) => sum + component.triangleCount,
    0,
  );
  for (const component of detailed) {
    component.allocationWeight =
      component.importance +
      (detailedTriangles > 0 ? component.triangleCount / detailedTriangles : 0);
  }
  const totalWeight = detailed.reduce(
    (sum, component) => sum + component.allocationWeight,
    0,
  );
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
  fractions.sort(
    (left, right) =>
      right.fraction - left.fraction ||
      left.component.originalIndex - right.component.originalIndex,
  );
  for (let index = 0; index < extraBudget - assigned && fractions.length > 0; index++) {
    fractions[index % fractions.length].component.maxHulls++;
  }
}

function planComponents(
  processed: CanonicalizedMesh,
  components: TriangleMesh[],
  policy: ResolvedComponentPolicy,
  requestedThreshold: number,
): ComponentPlan[] {
  policy.detailBudgetRatio = policy.enabled ? detailBudgetRatio(requestedThreshold) : 1;
  const measured = measureComponents(processed, components, policy);
  allocateHullBudget(measured, policy);

  return measured.sort(
    (left, right) =>
      Number(left.simplified) - Number(right.simplified) ||
      right.importance - left.importance ||
      left.originalIndex - right.originalIndex,
  );
}

export function componentConfig(
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
    config.threshold = Math.max(
      config.threshold ?? DECOMPOSE_DEFAULTS.threshold,
      policy.detailedComponentMinThreshold,
    );
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
  if (plan.maxHulls !== -1) {
    const requestedLimit = config.maxConvexHull;
    config.maxConvexHull = requestedLimit === undefined || requestedLimit === -1
      ? plan.maxHulls
      : Math.min(requestedLimit, plan.maxHulls);
  }
  return config;
}

export function configCacheKey(
  index: number,
  config: DecomposeConfig,
  checkManifold: boolean,
): string {
  return `${index}:${checkManifold}:${JSON.stringify(resolveDecomposeConfig(config))}`;
}

export function planGlbComponents(
  processed: CanonicalizedMesh,
  components: TriangleMesh[],
  options: InteractiveCompileOptions,
): ComponentPlanningResult {
  const policy = resolveComponentPolicy(
    options.componentPolicy,
    components.length,
  );
  const requestedThreshold = options.decompose?.threshold ?? DECOMPOSE_DEFAULTS.threshold;
  const plans = planComponents(processed, components, policy, requestedThreshold);
  const simplifiedCount = plans.filter((plan) => plan.simplified).length;
  const hollowShellCount = plans.filter((plan) => plan.hollowShell).length;
  const requestedMaxHullVertices =
    options.decompose?.maxChVertex ?? DECOMPOSE_DEFAULTS.maxChVertex;
  const effectiveHullVertexCaps = plans.map((plan) =>
    Math.min(requestedMaxHullVertices, policy.maxHullVertices, plan.maxHullVertices),
  );
  const hullVertexCapByComponent = new Map<number, number>();
  plans.forEach((plan, index) =>
    hullVertexCapByComponent.set(plan.originalIndex, effectiveHullVertexCaps[index]),
  );
  const adaptedHullVertexCount = policy.enabled
    ? effectiveHullVertexCaps.filter((cap) => cap < policy.maxHullVertices).length
    : 0;
  const thresholdClamped =
    policy.enabled && requestedThreshold < policy.detailedComponentMinThreshold;
  const detailedThreshold = policy.enabled
    ? Math.max(requestedThreshold, policy.detailedComponentMinThreshold)
    : requestedThreshold;
  const guardedHollowShellCount =
    requestedThreshold > policy.hollowShellMaxThreshold ? hollowShellCount : 0;
  const importanceGuardedPlans = policy.enabled
    ? plans.filter(
        (plan) =>
          !plan.simplified &&
          !plan.hollowShell &&
          detailedThreshold > plan.maxThreshold,
      )
    : [];
  const effectiveDetailedThresholds = plans
    .filter((plan) => !plan.simplified)
    .map((plan) =>
      Math.min(
        detailedThreshold,
        plan.maxThreshold,
        plan.hollowShell ? policy.hollowShellMaxThreshold : 1,
      ),
    );
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
