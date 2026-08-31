"""Canonical compilation-report metric names.

Generated, do not edit by hand. Source of truth:
docs/metric-vocabulary.json. Regenerate with:

    node scripts/gen-metric-names.mjs
"""

from __future__ import annotations

# --- coverage ---
SOURCE_SURFACE_COVERAGE = "source_surface_coverage"
WORST_COMPONENT_SURFACE_COVERAGE = "worst_component_surface_coverage"
WORST_DECILE_SURFACE_COVERAGE = "worst_decile_surface_coverage"

# --- volume ---
COLLIDER_VOLUME_PRECISION = "collider_volume_precision"
FALSE_FILL_FRACTION = "false_fill_fraction"
DEEP_FALSE_FILL_FRACTION = "deep_false_fill_fraction"

# --- decomposition ---
HULL_COUNT = "hull_count"
HULL_VERTEX_COUNT = "hull_vertex_count"
HULL_TRIANGLE_COUNT = "hull_triangle_count"
FALLBACK_RATIO = "fallback_ratio"
PLANAR_SUBSTITUTE_HULLS = "planar_substitute_hulls"
SNUG_FIT_STATUS = "snug_fit_status"

# --- walkability ---
PROBE_COVERAGE = "probe_coverage"
PROBE_GAP_CLUSTERS = "probe_gap_clusters"
SWEEP_TRAVERSABILITY = "sweep_traversability"
STANDABLE_FRACTION = "standable_fraction"
CLEARANCE_BLOCKED_FRACTION = "clearance_blocked_fraction"
RADIUS_BLOCKED_FRACTION = "radius_blocked_fraction"
SEAM_SNAG_COUNT = "seam_snag_count"

# --- quality_meta ---
QUALITY_METHOD = "quality_method"
QUALITY_SURFACE_SAMPLES = "quality_surface_samples"
QUALITY_VOLUME_SAMPLES = "quality_volume_samples"

# Maps a legacy/alias metric name to its canonical replacement.
ALIASES: dict[str, str] = {
    "covered_fraction": "source_surface_coverage",
    "worst_cell_fraction": "worst_component_surface_coverage",
    "worst_decile_fraction": "worst_decile_surface_coverage",
}
