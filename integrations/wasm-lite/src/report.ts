import type { ConvexHull } from "./types.js";

export const COMPILATION_REPORT_VERSION = 1 as const;

export type CompilationStage =
  | "reading-input"
  | "parsing-input"
  | "validating-input"
  | "loading-wasm"
  | "decomposing"
  | "verifying"
  | "writing-phys"
  | "done";

export interface CompilationProgress {
  stage: CompilationStage;
  message?: string;
  completed?: number;
  total?: number;
  elapsed_ms?: number;
  /** Estimated milliseconds remaining in the current stage. */
  eta_ms?: number;
}

export interface CompilationErrorInfo {
  code: string;
  message: string;
  stage: CompilationStage | null;
  suggestion: string | null;
  retryable: boolean;
  context: {
    mesh_name?: string;
    mesh_index?: number;
    primitive_index?: number;
    [key: string]: string | number | boolean | null | undefined;
  };
}

export interface CompilationWarning {
  code: string;
  severity: "info" | "warning" | "error";
  message: string;
  context: Record<string, string | number | boolean | null>;
}

export interface CompilationMetric {
  value: string | number | boolean | null;
  unit: string;
  status: "measured" | "not_measured" | "not_applicable";
}

export interface CompilationCheck {
  code: string;
  status: "pass" | "fail" | "not_evaluated";
  message: string;
}

export interface CompilationVerdict {
  profile: string | null;
  status: "pass" | "fail" | "not_evaluated";
  reasons: string[];
  checks: CompilationCheck[];
}

export interface CompilationRuntime {
  kind: string;
  implementation: string;
  version: string;
  compiler_version: string;
  dependencies: Record<string, string>;
}

export interface CompilationReport {
  report_version: typeof COMPILATION_REPORT_VERSION;
  status: "complete" | "rejected";
  profile: string | null;
  verdict: CompilationVerdict;
  input: {
    kind: string;
    source_vertices: number;
    processed_vertices: number;
    mesh_vertices: number;
  };
  output: {
    collider_kind: string;
    hull_count: number;
    vertex_count: number;
    triangle_count: number;
    lod_tier_count: number;
    byte_length: number | null;
  };
  timings_ms: Record<string, number>;
  warnings: CompilationWarning[];
  metrics: Record<string, CompilationMetric>;
  processing: {
    pipeline: string[];
    fallbacks: {
      decomposition_failure_hulls: number;
      planar_substitute_hulls: number | null;
    };
    refinements: {
      snug_fit: {
        status: "applied" | "skipped" | "not_requested" | "unknown";
        refined_hulls: number | null;
        rejected_hulls: number | null;
        skipped_hulls: number | null;
      };
    };
  };
  runtime: CompilationRuntime;
  reproducibility: {
    scope: "same_runtime_toolchain";
    deterministic: boolean | null;
    artifact_sha256: string | null;
  };
  config: {
    requested: Record<string, unknown> | null;
    effective: Record<string, unknown> | null;
  };
  artifacts: Record<string, string>;
}

export interface CreateCompilationReportOptions {
  profile?: string | null;
  verdict?: CompilationVerdict;
  input: CompilationReport["input"];
  collider_kind?: string;
  hulls: ConvexHull[];
  phys_bytes?: number | null;
  timings_ms?: Record<string, number>;
  warnings?: CompilationWarning[];
  metrics?: Record<string, CompilationMetric>;
  runtime: CompilationRuntime;
  deterministic?: boolean | null;
  artifact_sha256?: string | null;
  requested_config?: Record<string, unknown> | null;
  effective_config?: Record<string, unknown> | null;
  artifacts?: Record<string, string>;
}

function metric(
  value: string | number | boolean | null,
  unit: string,
  absent: CompilationMetric["status"] = "not_measured",
): CompilationMetric {
  return { value, unit, status: value === null ? absent : "measured" };
}

/**
 * Build the v1 cross-runtime report for a browser/WASM compilation.
 *
 * The current low-level WASM API does not run profile acceptance probes, so a
 * caller that does not provide a verdict receives `not_evaluated`, never an
 * implied pass. The high-level GLB compiler will add its measured stages to
 * this same contract.
 */
export function createCompilationReport(
  options: CreateCompilationReportOptions,
): CompilationReport {
  const profile = options.verdict?.profile ?? options.profile ?? null;
  const verdict: CompilationVerdict = options.verdict ?? {
    profile,
    status: "not_evaluated",
    reasons: [],
    checks: [],
  };
  const vertexCount = options.hulls.reduce(
    (total, hull) => total + hull.vertices.length / 3,
    0,
  );
  const triangleCount = options.hulls.reduce(
    (total, hull) => total + hull.indices.length / 3,
    0,
  );
  const deterministic = options.deterministic ?? null;
  const report: CompilationReport = {
    report_version: COMPILATION_REPORT_VERSION,
    status: verdict.status === "fail" ? "rejected" : "complete",
    profile,
    verdict,
    input: { ...options.input },
    output: {
      collider_kind: options.collider_kind ?? "static",
      hull_count: options.hulls.length,
      vertex_count: vertexCount,
      triangle_count: triangleCount,
      lod_tier_count: 0,
      byte_length: options.phys_bytes ?? null,
    },
    timings_ms: { ...options.timings_ms },
    warnings: [...(options.warnings ?? [])],
    metrics: {
      hull_count: metric(options.hulls.length, "count"),
      covered_fraction: metric(null, "ratio"),
      worst_cell_fraction: metric(null, "ratio"),
      worst_decile_fraction: metric(null, "ratio"),
      fallback_hulls: metric(0, "count"),
      coacd_timeouts: metric(0, "count"),
      coacd_deterministic: metric(
        deterministic,
        "boolean",
        "not_applicable",
      ),
      ...(options.metrics ?? {}),
    },
    processing: {
      pipeline: ["decompose", "write-phys"],
      fallbacks: {
        decomposition_failure_hulls: 0,
        planar_substitute_hulls: null,
      },
      refinements: {
        snug_fit: {
          status: "not_requested",
          refined_hulls: null,
          rejected_hulls: null,
          skipped_hulls: null,
        },
      },
    },
    runtime: { ...options.runtime },
    reproducibility: {
      scope: "same_runtime_toolchain",
      deterministic,
      artifact_sha256: options.artifact_sha256 ?? null,
    },
    config: {
      requested: options.requested_config ?? null,
      effective: options.effective_config ?? null,
    },
    artifacts: { ...options.artifacts },
  };
  const problems = validateCompilationReport(report);
  if (problems.length > 0) {
    throw new Error(`invalid compilation report: ${problems.join("; ")}`);
  }
  return report;
}

/** Lightweight structural validation for reports received across a boundary. */
export function validateCompilationReport(report: unknown): string[] {
  if (!report || typeof report !== "object") return ["report must be an object"];
  const value = report as Partial<CompilationReport>;
  const problems: string[] = [];
  const required = [
    "report_version",
    "status",
    "profile",
    "verdict",
    "input",
    "output",
    "timings_ms",
    "warnings",
    "metrics",
    "processing",
    "runtime",
    "reproducibility",
    "config",
    "artifacts",
  ] as const;
  for (const field of required) {
    if (!(field in value)) problems.push(`missing field: ${field}`);
  }
  if (value.report_version !== COMPILATION_REPORT_VERSION) {
    problems.push(
      `unsupported report_version ${String(value.report_version)}; expected ${COMPILATION_REPORT_VERSION}`,
    );
  }
  if (
    !value.verdict ||
    !["pass", "fail", "not_evaluated"].includes(value.verdict.status)
  ) {
    problems.push("verdict.status must be pass, fail, or not_evaluated");
  }
  if (!Array.isArray(value.warnings)) problems.push("warnings must be an array");
  if (!value.metrics || typeof value.metrics !== "object") {
    problems.push("metrics must be an object");
  }
  if (!value.timings_ms || typeof value.timings_ms !== "object") {
    problems.push("timings_ms must be an object");
  } else {
    for (const [name, timing] of Object.entries(value.timings_ms)) {
      if (!Number.isFinite(timing) || timing < 0) {
        problems.push(`timing ${name} must be a finite non-negative number`);
      }
    }
  }
  if (value.reproducibility?.scope !== "same_runtime_toolchain") {
    problems.push("reproducibility.scope must be same_runtime_toolchain");
  }
  return problems;
}
