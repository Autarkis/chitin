// Canonical compilation-report metric names.
//
// Generated, do not edit by hand. Source of truth:
// docs/metric-vocabulary.json. Regenerate with:
//
//   node scripts/gen-metric-names.mjs

// --- coverage ---
export const SOURCE_SURFACE_COVERAGE = "source_surface_coverage";
export const WORST_COMPONENT_SURFACE_COVERAGE = "worst_component_surface_coverage";
export const WORST_DECILE_SURFACE_COVERAGE = "worst_decile_surface_coverage";

// --- volume ---
export const COLLIDER_VOLUME_PRECISION = "collider_volume_precision";
export const FALSE_FILL_FRACTION = "false_fill_fraction";
export const DEEP_FALSE_FILL_FRACTION = "deep_false_fill_fraction";

// --- decomposition ---
export const HULL_COUNT = "hull_count";

// --- quality_meta ---
export const QUALITY_METHOD = "quality_method";
export const QUALITY_SURFACE_SAMPLES = "quality_surface_samples";
export const QUALITY_VOLUME_SAMPLES = "quality_volume_samples";

// Maps a legacy/alias metric name to its canonical replacement.
export const ALIASES: Record<string, string> = {
  "covered_fraction": "source_surface_coverage",
  "worst_cell_fraction": "worst_component_surface_coverage",
  "worst_decile_fraction": "worst_decile_surface_coverage",
};
