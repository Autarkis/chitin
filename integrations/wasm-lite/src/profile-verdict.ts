import { ACCEPTANCE_THRESHOLDS, type BrowserProfileName } from "./shared-constants.js";
import type { CompilationCheck, CompilationMetric, CompilationVerdict } from "./report.js";
import type { ConvexHull } from "./types.js";

function value(metrics: Record<string, CompilationMetric> | undefined, name: string): number | null {
  const candidate = metrics?.[name]?.value;
  return typeof candidate === "number" ? candidate : null;
}

function thresholdCheck(
  code: string,
  measured: number | null,
  threshold: number,
  direction: "min" | "max",
): CompilationCheck {
  const passed = measured !== null && (direction === "min" ? measured >= threshold : measured <= threshold);
  return {
    code,
    status: measured === null ? "not_evaluated" : passed ? "pass" : "fail",
    message: measured === null
      ? `${code} was not measured`
      : `${code} ${measured.toFixed(4)} ${direction === "min" ? ">=" : "<="} ${threshold}`,
    suggestion: passed ? null : "Adjust collider detail and rerun profile diagnostics.",
  };
}

export function browserProfileVerdict(
  profile: BrowserProfileName,
  hulls: ConvexHull[],
  metrics: Record<string, CompilationMetric> | undefined,
): CompilationVerdict | undefined {
  if (profile === "interactive") return undefined;
  const checks: CompilationCheck[] = [{
    code: "has_hulls",
    status: hulls.length > 0 ? "pass" : "fail",
    message: `${hulls.length} hull(s) generated`,
  }];
  if (profile === "walkable") {
    const policy = ACCEPTANCE_THRESHOLDS.walkable;
    checks.push(
      thresholdCheck("source_surface_coverage", value(metrics, "source_surface_coverage"), policy.min_covered_fraction, "min"),
      thresholdCheck("false_fill_fraction", value(metrics, "false_fill_fraction"), policy.max_false_fill_fraction, "max"),
      { code: "walkable_probe", status: "not_evaluated", message: "Native walkable probe is unavailable in WASM" },
      { code: "capsule_sweep", status: "not_evaluated", message: "Native capsule sweep is unavailable in WASM" },
    );
  } else {
    const policy = ACCEPTANCE_THRESHOLDS.robotics;
    checks.push(
      thresholdCheck("source_surface_coverage", value(metrics, "source_surface_coverage"), policy.min_covered_fraction, "min"),
      thresholdCheck("worst_component_surface_coverage", value(metrics, "worst_component_surface_coverage"), policy.min_worst_cell_fraction, "min"),
      thresholdCheck("false_fill_fraction", value(metrics, "false_fill_fraction"), policy.max_false_fill_fraction, "max"),
      thresholdCheck("deep_false_fill_fraction", value(metrics, "deep_false_fill_fraction"), policy.max_deep_false_fill_fraction, "max"),
      { code: "snug_fit", status: "not_evaluated", message: "Native snug-fit refinement is unavailable in WASM" },
    );
  }
  const status = checks.some((check) => check.status === "fail")
    ? "fail"
    : checks.some((check) => check.status === "not_evaluated") ? "not_evaluated" : "pass";
  return {
    profile,
    status,
    reasons: checks.filter((check) => check.status === "fail").map((check) => check.message),
    checks,
  };
}
