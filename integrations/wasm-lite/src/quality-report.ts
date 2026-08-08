import type { ComponentPlan } from "./interactive-policy.js";
import type { CanonicalizedMesh } from "./mesh.js";
import {
  COLLIDER_VOLUME_PRECISION,
  DEEP_FALSE_FILL_FRACTION,
  FALSE_FILL_FRACTION,
  QUALITY_METHOD,
  QUALITY_SURFACE_SAMPLES,
  QUALITY_VOLUME_SAMPLES,
  SOURCE_SURFACE_COVERAGE,
  WORST_COMPONENT_SURFACE_COVERAGE,
} from "./metric-names.js";
import {
  evaluateColliderQuality,
  type ColliderQualityOptions,
} from "./quality.js";
import { metric } from "./report.js";
import type { CompilationMetric } from "./report.js";
import type { ConvexHull } from "./types.js";

export function evaluateQualityMetrics(
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
    [SOURCE_SURFACE_COVERAGE]: metric(evaluated.source_surface_coverage, "ratio"),
    [WORST_COMPONENT_SURFACE_COVERAGE]: metric(
      evaluated.worst_component_surface_coverage,
      "ratio",
    ),
    detailed_source_surface_coverage: metric(detailedSurfaceCoverage, "ratio"),
    worst_detailed_component_surface_coverage: metric(worstDetailedCoverage, "ratio"),
    [COLLIDER_VOLUME_PRECISION]: metric(evaluated.collider_volume_precision, "ratio"),
    [FALSE_FILL_FRACTION]: metric(evaluated.false_fill_fraction, "ratio"),
    [DEEP_FALSE_FILL_FRACTION]: metric(evaluated.deep_false_fill_fraction, "ratio"),
    [QUALITY_METHOD]: metric(evaluated.method, "method"),
    [QUALITY_SURFACE_SAMPLES]: metric(evaluated.surface_samples, "count"),
    [QUALITY_VOLUME_SAMPLES]: metric(evaluated.volume_samples, "count"),
    quality_collider_volume_samples: metric(evaluated.collider_volume_samples, "count"),
    quality_component_count: metric(evaluated.component_count, "count"),
    quality_volume_tolerance: metric(evaluated.volume_tolerance, "source_unit"),
    quality_surface_tolerance_ratio: metric(evaluated.surface_tolerance_ratio, "ratio"),
    quality_deep_fill_clearance_ratio: metric(
      evaluated.deep_fill_clearance_ratio,
      "ratio",
    ),
  };
  for (const component of evaluated.components) {
    const prefix = `quality_component_${component.component_index}`;
    qualityMetrics[`${prefix}_surface_coverage`] = metric(
      component.surface_coverage,
      "ratio",
    );
    qualityMetrics[`${prefix}_surface_area_fraction`] = metric(
      component.surface_area_fraction,
      "ratio",
    );
    qualityMetrics[`${prefix}_diagonal_ratio`] = metric(component.diagonal_ratio, "ratio");
    qualityMetrics[`${prefix}_vertex_count`] = metric(component.vertex_count, "count");
    qualityMetrics[`${prefix}_triangle_count`] = metric(component.triangle_count, "count");
    qualityMetrics[`${prefix}_surface_samples`] = metric(component.surface_samples, "count");
    if (component.hull_count !== null) {
      qualityMetrics[`${prefix}_hull_count`] = metric(component.hull_count, "count");
    }
    if (component.collider_triangle_count !== null) {
      qualityMetrics[`${prefix}_collider_triangle_count`] = metric(
        component.collider_triangle_count,
        "count",
      );
    }
    qualityMetrics[`${prefix}_collider_volume_precision`] = metric(
      component.collider_volume_precision,
      "ratio",
    );
    qualityMetrics[`${prefix}_false_fill_fraction`] = metric(
      component.false_fill_fraction,
      "ratio",
    );
    qualityMetrics[`${prefix}_deep_false_fill_fraction`] = metric(
      component.deep_false_fill_fraction,
      "ratio",
    );
    qualityMetrics[`${prefix}_collider_volume_samples`] = metric(
      component.collider_volume_samples,
      "count",
    );
  }
  return qualityMetrics;
}
