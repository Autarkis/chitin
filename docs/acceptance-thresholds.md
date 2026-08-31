# Acceptance Thresholds

Profile verdicts gate compiled colliders against quality, coverage, and complexity
thresholds. A failed check rejects the build (strict profiles) and carries an
actionable suggestion.

## Threshold reference

| Profile | Check | Threshold | Unit | Rationale |
|---------|-------|-----------|------|-----------|
| robotics | min_covered_fraction | 0.90 | ratio | Sim contact accuracy requires ≥90% source surface covered |
| robotics | min_worst_cell_fraction | 0.70 | ratio | No single spatial cell below 70% coverage |
| robotics | allow_fallback_hulls | false | — | Phantom AABB from CoACD timeout breaks sim contact |
| robotics | require_deterministic | true | — | Sim-validated collider must be reproducible from manifest |
| robotics | max_false_fill_fraction | 0.30 | ratio | >30% phantom volume causes false collisions in grasping |
| robotics | max_deep_false_fill_fraction | 0.20 | ratio | >20% deep interior overfill is structurally wrong |
| walkable | min_covered_fraction | 0.85 | ratio | Floor coverage for navigation |
| walkable | allow_fallback_hulls | true | — | Bounding-box fallback acceptable for floor plates |
| walkable | max_false_fill_fraction | 0.50 | ratio | Coarser environments tolerate more overfill |
| walkable | min_probe_coverage | 0.70 | ratio | Capsule ray coverage ≥70% for traversable floor |
| walkable | max_probe_gap_clusters | 5 | count | ≤5 distinct gap regions in floor collider |
| interactive | (none) | — | — | Permissive: no gates, all builds accepted |

## Calibration status

These thresholds are **pre-calibration defaults**. They will be re-baselined
against the post-determinism benchmark corpus once it is available. The rationale
column documents the design intent; measured corpus statistics will follow.

## Optional gates (not enabled by default)

| Field | Unit | Purpose |
|-------|------|---------|
| max_compile_ms | ms | Flag slow compilations |
| max_output_bytes | bytes | Gate oversized .phys files |
| max_hull_vertices | count | Gate collider mesh complexity |

These are available on `AcceptancePolicy` for custom profiles. No built-in
profile enables them.

## Metric units

- **ratio**: [0, 1] fraction (e.g., 0.90 = 90%)
- **count**: non-negative integer
- **ms**: milliseconds
- **bytes**: byte count
