import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

interface Metric {
  value: string | number | boolean | null;
  unit: string;
  status: "measured" | "not_measured" | "not_applicable";
}

interface BenchmarkReport {
  verdict: { status: string };
  output: { hull_count: number; triangle_count: number };
  timings_ms: Record<string, number>;
  metrics: Record<string, Metric>;
  config: {
    effective: {
      component_plans?: Array<Record<string, string | number | boolean | null>>;
      max_hulls?: number;
      max_hulls_ceiling?: number;
      detail_budget_ratio?: number;
    } | null;
  };
}

interface QualityThresholds {
  min_detailed_surface_coverage: number;
  min_worst_detailed_component_coverage: number;
  max_deep_false_fill: number;
}

interface RegressionThresholds extends QualityThresholds {
  min_hulls: number;
  max_hulls: number;
  max_triangles: number;
}

interface BenchmarkConfig {
  samples: Record<string, {
    detail: number;
    regression: RegressionThresholds;
    target: QualityThresholds;
  }>;
}

const config = JSON.parse(
  readFileSync(resolve("benchmarks/sample-quality.json"), "utf8"),
) as BenchmarkConfig;

const scenarios = [
  { name: "wicker", testId: "sample-wicker" },
  { name: "dish", testId: "sample-dish" },
  { name: "fish", testId: "sample-fish" },
] as const;

function targetGaps(result: {
  detailedSurfaceCoverage: number;
  worstDetailedComponentCoverage: number;
  deepFalseFill: number;
}, target: QualityThresholds): string[] {
  const gaps: string[] = [];
  if (result.detailedSurfaceCoverage < target.min_detailed_surface_coverage) {
    gaps.push(
      `detailed surface ${result.detailedSurfaceCoverage} < ${target.min_detailed_surface_coverage}`,
    );
  }
  if (
    result.worstDetailedComponentCoverage < target.min_worst_detailed_component_coverage
  ) {
    gaps.push(
      `worst detailed component ${result.worstDetailedComponentCoverage} < ${target.min_worst_detailed_component_coverage}`,
    );
  }
  if (result.deepFalseFill > target.max_deep_false_fill) {
    gaps.push(`deep false fill ${result.deepFalseFill} > ${target.max_deep_false_fill}`);
  }
  return gaps;
}

function measuredNumber(report: BenchmarkReport, name: string): number {
  const metric = report.metrics[name];
  expect(metric, `missing ${name}`).toBeDefined();
  expect(metric.status, `${name} was not measured`).toBe("measured");
  expect(typeof metric.value, `${name} is not numeric`).toBe("number");
  return metric.value as number;
}

function optionalNumber(report: BenchmarkReport, name: string): number | null {
  const metric = report.metrics[name];
  if (!metric || metric.status !== "measured" || typeof metric.value !== "number") return null;
  return metric.value;
}

function componentDiagnostics(report: BenchmarkReport) {
  const plans = new Map(
    (report.config.effective?.component_plans ?? []).map((plan) => [
      Number(plan.component_index),
      plan,
    ]),
  );
  return Object.keys(report.metrics)
    .map((name) => /^quality_component_(\d+)_surface_coverage$/.exec(name))
    .filter((match): match is RegExpExecArray => match !== null)
    .map((match) => {
      const component = Number(match[1]);
      const value = (suffix: string) =>
        measuredNumber(report, `quality_component_${component}_${suffix}`);
      return {
        component,
        coverage: value("surface_coverage"),
        areaFraction: value("surface_area_fraction"),
        diagonalRatio: value("diagonal_ratio"),
        triangles: value("triangle_count"),
        samples: value("surface_samples"),
        hulls: optionalNumber(report, `quality_component_${component}_hull_count`),
        colliderTriangles: optionalNumber(
          report,
          `quality_component_${component}_collider_triangle_count`,
        ),
        falseFill: optionalNumber(report, `quality_component_${component}_false_fill_fraction`),
        deepFalseFill: optionalNumber(
          report,
          `quality_component_${component}_deep_false_fill_fraction`,
        ),
        volumeSamples: optionalNumber(
          report,
          `quality_component_${component}_collider_volume_samples`,
        ),
        plan: plans.get(component) ?? null,
      };
    })
    .sort((left, right) => left.coverage - right.coverage || right.areaFraction - left.areaFraction);
}

test.describe("sample collider quality acceptance", () => {
  test.describe.configure({ mode: "serial" });

  for (const scenario of scenarios) {
    test(`${scenario.name} records deterministic artifact-fit metrics`, async ({ page }, testInfo) => {
      test.skip(
        testInfo.project.name !== "chromium",
        "Pinned Chromium is the benchmark runtime; functional behavior remains cross-browser tested.",
      );
      const acceptance = config.samples[scenario.name];
      await page.goto("/?qualityBenchmark=1");
      await page.waitForFunction(() => window.__chitinDemo?.ready);
      await page.locator("#threshold").evaluate((element, detail) => {
        const slider = element as HTMLInputElement;
        slider.value = detail.toFixed(2);
        slider.dispatchEvent(new Event("input", { bubbles: true }));
      }, acceptance.detail);
      await page.getByTestId(scenario.testId).click();
      await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 120_000 });

      const report = JSON.parse(
        (await page.locator("#report-output").textContent()) ?? "null",
      ) as BenchmarkReport;
      const result = {
        sample: scenario.name,
        detail: acceptance.detail,
        hulls: report.output.hull_count,
        triangles: report.output.triangle_count,
        surfaceCoverage: measuredNumber(report, "source_surface_coverage"),
        worstComponentCoverage: measuredNumber(report, "worst_component_surface_coverage"),
        detailedSurfaceCoverage: measuredNumber(report, "detailed_source_surface_coverage"),
        worstDetailedComponentCoverage: measuredNumber(
          report,
          "worst_detailed_component_surface_coverage",
        ),
        volumePrecision: measuredNumber(report, "collider_volume_precision"),
        falseFill: measuredNumber(report, "false_fill_fraction"),
        deepFalseFill: measuredNumber(report, "deep_false_fill_fraction"),
        decomposeMs: Math.round(report.timings_ms.decompose ?? 0),
        verifyMs: Math.round(report.timings_ms.verify ?? 0),
      };
      const diagnostics = componentDiagnostics(report);
      console.log(`QUALITY_BENCHMARK ${JSON.stringify({
        ...result,
        targetGaps: targetGaps(result, acceptance.target),
        worstCoverage: diagnostics.slice(0, 5),
        largestComponents: [...diagnostics]
          .sort((left, right) => right.areaFraction - left.areaFraction)
          .slice(0, 8),
        worstFalseFill: diagnostics
          .filter((component) => component.falseFill !== null)
          .sort((left, right) => right.falseFill! - left.falseFill!)
          .slice(0, 8),
      })}`);

      expect(report.verdict.status).toBe("not_evaluated");
      expect(report.metrics.quality_method).toEqual({
        value: "deterministic_halton_v1",
        unit: "method",
        status: "measured",
      });
      expect(result.surfaceCoverage).toBeGreaterThanOrEqual(0);
      expect(result.surfaceCoverage).toBeLessThanOrEqual(1);
      expect(result.worstComponentCoverage).toBeGreaterThanOrEqual(0);
      expect(result.worstComponentCoverage).toBeLessThanOrEqual(1);
      expect(result.deepFalseFill).toBeGreaterThanOrEqual(0);
      expect(result.deepFalseFill).toBeLessThanOrEqual(1);
      expect(result.detailedSurfaceCoverage).toBeGreaterThanOrEqual(
        acceptance.regression.min_detailed_surface_coverage,
      );
      expect(result.worstDetailedComponentCoverage).toBeGreaterThanOrEqual(
        acceptance.regression.min_worst_detailed_component_coverage,
      );
      expect(result.deepFalseFill).toBeLessThanOrEqual(
        acceptance.regression.max_deep_false_fill,
      );
      expect(result.hulls).toBeGreaterThanOrEqual(acceptance.regression.min_hulls);
      expect(result.hulls).toBeLessThanOrEqual(acceptance.regression.max_hulls);
      expect(result.triangles).toBeLessThanOrEqual(acceptance.regression.max_triangles);
      if (scenario.name === "dish") {
        expect(report.config.effective?.max_hulls).toBe(result.hulls);
        expect(report.config.effective?.max_hulls).toBeLessThan(
          report.config.effective?.max_hulls_ceiling ?? 0,
        );
        expect(report.config.effective?.detail_budget_ratio).toBeLessThan(1);
      }
    });
  }
});
