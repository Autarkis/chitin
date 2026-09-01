"""Generated from docs/shared-constants.json; do not edit."""

COACD_CONCAVITY_THRESHOLD = 0.05
COACD_PREPROCESS_RESOLUTION = 50
NATIVE_MIN_HULL_VERTICES = 4
INTERACTIVE_MIN_HULL_VERTICES = 8
PROFILE_NAMES = ("interactive", "walkable", "robotics")
ACCEPTANCE_THRESHOLDS = {
  "walkable": {
    "max_fallback_ratio": 0.25,
    "min_covered_fraction": 0.85,
    "max_false_fill_fraction": 0.5,
    "min_probe_coverage": 0.7,
    "max_probe_gap_clusters": 5,
    "min_sweep_traversability": 0.8,
    "min_standable_fraction": 0.7,
    "max_clearance_blocked_fraction": 0.2
  },
  "robotics": {
    "min_covered_fraction": 0.9,
    "min_worst_cell_fraction": 0.7,
    "max_false_fill_fraction": 0.3,
    "max_deep_false_fill_fraction": 0.2,
    "max_hull_count": 2048,
    "max_hull_vertices": 131072,
    "max_hull_triangles": 262144
  }
}
