import type { ConvexHull } from "./types.js";

import {
  TOP_LEVEL_REQUIRED,
  VERDICT_REQUIRED,
  CHECK_REQUIRED,
  INPUT_REQUIRED,
  OUTPUT_REQUIRED,
  WARNING_REQUIRED,
  METRIC_REQUIRED,
  PROCESSING_REQUIRED,
  FALLBACKS_REQUIRED,
  REFINEMENTS_REQUIRED,
  SNUG_FIT_REQUIRED,
  RUNTIME_REQUIRED,
  REPRODUCIBILITY_REQUIRED,
  CONFIG_REQUIRED,
  BUILD_IDENTITY_REQUIRED,
  REPORT_STATUS_VALUES,
  VERDICT_STATUS_VALUES,
  CHECK_STATUS_VALUES,
  WARNING_SEVERITY_VALUES,
  METRIC_STATUS_VALUES,
  SNUG_FIT_STATUS_VALUES,
} from "./report-schema-constants.js";

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
  suggestion?: string | null;
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

export interface BuildIdentity {
  effective_input_digest: string | null;
  algorithm_version: string;
  numerical_policy_version: string;
  config_digest: string | null;
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
  build_identity: BuildIdentity;
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
  algorithm_version?: string;
  numerical_policy_version?: string;
}

export function metric(
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
      ...(options.metrics ?? {}),
    },
    processing: {
      pipeline: ["decompose", "write-phys"],
      fallbacks: {
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
    build_identity: {
      effective_input_digest: null,
      algorithm_version: options.algorithm_version ?? "1.0.0",
      numerical_policy_version: options.numerical_policy_version ?? "1.0.0",
      config_digest: null,
    },
  };
  const problems = validateCompilationReport(report);
  if (problems.length > 0) {
    throw new Error(`invalid compilation report: ${problems.join("; ")}`);
  }
  return report;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Pushes `missing field: <label>.<key>` for every required key absent from `value`. */
function requireFields(
  value: unknown,
  keys: readonly string[],
  label: string,
  problems: string[],
): value is Record<string, unknown> {
  if (!isPlainObject(value)) {
    problems.push(`${label} must be an object`);
    return false;
  }
  for (const key of keys) {
    if (!(key in value)) problems.push(`missing field: ${label}.${key}`);
  }
  return true;
}

function isNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/** Lightweight structural validation for reports received across a boundary. */
export function validateCompilationReport(report: unknown): string[] {
  if (!isPlainObject(report)) return ["report must be an object"];
  const problems: string[] = [];
  for (const field of TOP_LEVEL_REQUIRED) {
    if (!(field in report)) problems.push(`missing field: ${field}`);
  }

  if (report.report_version !== COMPILATION_REPORT_VERSION) {
    problems.push(
      `unsupported report_version ${String(report.report_version)}; expected ${COMPILATION_REPORT_VERSION}`,
    );
  }
  if (
    report.status !== undefined &&
    !(REPORT_STATUS_VALUES as readonly string[]).includes(report.status as string)
  ) {
    problems.push("status must be complete or rejected");
  }

  if (requireFields(report.verdict, VERDICT_REQUIRED, "verdict", problems)) {
    const verdict = report.verdict as Record<string, unknown>;
    if (
      "status" in verdict &&
      !(VERDICT_STATUS_VALUES as readonly string[]).includes(verdict.status as string)
    ) {
      problems.push("verdict.status must be pass, fail, or not_evaluated");
    }
    if ("reasons" in verdict && !Array.isArray(verdict.reasons)) {
      problems.push("verdict.reasons must be an array");
    }
    if ("checks" in verdict) {
      if (!Array.isArray(verdict.checks)) {
        problems.push("verdict.checks must be an array");
      } else {
        verdict.checks.forEach((check, index) => {
          if (
            requireFields(check, CHECK_REQUIRED, `verdict.checks[${index}]`, problems) &&
            !(CHECK_STATUS_VALUES as readonly string[]).includes(
              (check as Record<string, unknown>).status as string,
            )
          ) {
            problems.push(`verdict.checks[${index}].status must be pass, fail, or not_evaluated`);
          }
        });
      }
    }
  }

  requireFields(report.input, INPUT_REQUIRED, "input", problems);
  requireFields(report.output, OUTPUT_REQUIRED, "output", problems);

  if (!Array.isArray(report.warnings)) {
    problems.push("warnings must be an array");
  } else {
    report.warnings.forEach((warning, index) => {
      const label = `warnings[${index}]`;
      if (
        requireFields(warning, WARNING_REQUIRED, label, problems) &&
        !(WARNING_SEVERITY_VALUES as readonly string[]).includes(
          (warning as Record<string, unknown>).severity as string,
        )
      ) {
        problems.push(`${label}.severity must be info, warning, or error`);
      }
    });
  }

  if (!isPlainObject(report.metrics)) {
    problems.push("metrics must be an object");
  } else {
    for (const [name, metric] of Object.entries(report.metrics)) {
      const label = `metrics.${name}`;
      if (
        requireFields(metric, METRIC_REQUIRED, label, problems) &&
        !(METRIC_STATUS_VALUES as readonly string[]).includes(
          (metric as Record<string, unknown>).status as string,
        )
      ) {
        problems.push(`${label}.status must be measured, not_measured, or not_applicable`);
      }
    }
  }

  if (!isPlainObject(report.timings_ms)) {
    problems.push("timings_ms must be an object");
  } else {
    for (const [name, timing] of Object.entries(report.timings_ms)) {
      if (!isNonNegativeNumber(timing)) {
        problems.push(`timing ${name} must be a finite non-negative number`);
      }
    }
  }

  if (requireFields(report.processing, PROCESSING_REQUIRED, "processing", problems)) {
    const processing = report.processing as Record<string, unknown>;
    if ("pipeline" in processing && !Array.isArray(processing.pipeline)) {
      problems.push("processing.pipeline must be an array");
    }
    requireFields(processing.fallbacks, FALLBACKS_REQUIRED, "processing.fallbacks", problems);
    if (requireFields(processing.refinements, REFINEMENTS_REQUIRED, "processing.refinements", problems)) {
      const refinements = processing.refinements as Record<string, unknown>;
      if (
        requireFields(refinements.snug_fit, SNUG_FIT_REQUIRED, "processing.refinements.snug_fit", problems)
      ) {
        const snugFit = refinements.snug_fit as Record<string, unknown>;
        if (
          "status" in snugFit &&
          !(SNUG_FIT_STATUS_VALUES as readonly string[]).includes(snugFit.status as string)
        ) {
          problems.push(
            "processing.refinements.snug_fit.status must be applied, skipped, not_requested, or unknown",
          );
        }
      }
    }
  }

  requireFields(report.runtime, RUNTIME_REQUIRED, "runtime", problems);

  if (requireFields(report.reproducibility, REPRODUCIBILITY_REQUIRED, "reproducibility", problems)) {
    const reproducibility = report.reproducibility as Record<string, unknown>;
    if (reproducibility.scope !== "same_runtime_toolchain") {
      problems.push("reproducibility.scope must be same_runtime_toolchain");
    }
    const sha = reproducibility.artifact_sha256;
    if (sha !== null && sha !== undefined && !/^[0-9a-f]{64}$/.test(sha as string)) {
      problems.push("reproducibility.artifact_sha256 must be a 64-char lowercase hex string or null");
    }
  }

  requireFields(report.config, CONFIG_REQUIRED, "config", problems);
  requireFields(report.build_identity, BUILD_IDENTITY_REQUIRED, "build_identity", problems);

  if (!isPlainObject(report.artifacts)) {
    problems.push("artifacts must be an object");
  }

  return problems;
}
