import type { CompileGlbResult } from "@autarkis/chitin-lite";

function hasWarning(result: CompileGlbResult, code: string): boolean {
  return result.report.warnings.some((warning) => warning.code === code);
}

function effectiveNumber(result: CompileGlbResult, key: string): number | null {
  const value = result.report.config.effective?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function measuredMetricNumber(
  result: CompileGlbResult,
  key: string,
): number | null {
  const metric = result.report.metrics[key];
  return metric?.status === "measured" &&
    typeof metric.value === "number" &&
    Number.isFinite(metric.value)
    ? metric.value
    : null;
}

export function metricPercentCopy(result: CompileGlbResult, key: string): string {
  const value = measuredMetricNumber(result, key);
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function hasQualityDiagnostics(result: CompileGlbResult): boolean {
  return measuredMetricNumber(result, "source_surface_coverage") !== null;
}

function hullBudgetCopy(result: CompileGlbResult): string {
  const budget = effectiveNumber(result, "max_hulls");
  const ceiling = effectiveNumber(result, "max_hulls_ceiling");
  if (budget === null || ceiling === null || budget === -1) return "";
  return budget === ceiling ? ` · hull budget ${budget}` : ` · hull budget ${budget}/${ceiling}`;
}

export function resultSummaryCopy(result: CompileGlbResult, detail: number): string {
  const budget = hullBudgetCopy(result);
  const profile = result.report.profile ?? "interactive";
  const verdict = result.report.verdict.status.replace("_", " ");
  if (hasWarning(result, "INTERACTIVE_HOLLOW_SHELL_GUARD")) {
    return `Detail ${detail.toFixed(2)} requested${budget} · hollow-shell guard · ${profile} ${verdict}`;
  }
  if (hasWarning(result, "INTERACTIVE_IMPORTANCE_GUARD")) {
    return `Detail ${detail.toFixed(2)} requested${budget} · scale-aware detail · ${profile} ${verdict}`;
  }
  if (hasWarning(result, "INTERACTIVE_HULL_VERTICES_ADAPTED")) {
    return `Detail ${detail.toFixed(2)} applied${budget} · adaptive hull budget · ${profile} ${verdict}`;
  }
  return `Detail ${detail.toFixed(2)} applied${budget} · ${profile} profile · ${verdict}`;
}

export function appliedThresholdCopy(result: CompileGlbResult, detail: number): string {
  const hulls = `${result.hulls.length} ${result.hulls.length === 1 ? "hull" : "hulls"}`;
  const budget = hullBudgetCopy(result);
  if (hasWarning(result, "INTERACTIVE_HOLLOW_SHELL_GUARD")) {
    return `Requested ${detail.toFixed(2)} · ${hulls}${budget} · hollow-shell guard active`;
  }
  if (hasWarning(result, "INTERACTIVE_IMPORTANCE_GUARD")) {
    return `Requested ${detail.toFixed(2)} · ${hulls}${budget} · scale-aware body detail active`;
  }
  return `Applied ${detail.toFixed(2)} · ${hulls}${budget}`;
}
