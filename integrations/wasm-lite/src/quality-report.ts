import type { ComponentPlan } from "./interactive-policy.js";
import type { CanonicalizedMesh } from "./mesh.js";
import {
  evaluateColliderQuality,
  type ColliderQualityOptions,
} from "./quality.js";
import type { CompilationMetric } from "./report.js";
import type { ConvexHull } from "./types.js";

function measured(value: string | number, unit: string): CompilationMetric {
  return { value, unit, status: "measured" };
}

function optional(value: number | null, unit: string): CompilationMetric {
  return {
    value,
    unit,
    status: value === null ? "not_measured" : "measured",
  };
}

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
    source_surface_coverage: measured(evaluated.source_surface_coverage, "ratio"),
    worst_component_surface_coverage: measured(
      evaluated.worst_component_surface_coverage,
      "ratio",
    ),
    detailed_source_surface_coverage: optional(detailedSurfaceCoverage, "ratio"),
    worst_detailed_component_surface_coverage: optional(worstDetailedCoverage, "ratio"),
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
    qualityMetrics[`${prefix}_diagonal_ratio`] = measured(component.diagonal_ratio, "ratio");
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
