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
| robotics | require_snug_fit | true | — | Requested refinement must run rather than being silently skipped |
| robotics | max_false_fill_fraction | 0.30 | ratio | >30% phantom volume causes false collisions in grasping |
| robotics | max_deep_false_fill_fraction | 0.20 | ratio | >20% deep interior overfill is structurally wrong |
| robotics | max_hull_count | 2048 | count | Bound broad-phase collider complexity |
| robotics | max_hull_vertices | 131072 | count | Bound aggregate convex-mesh complexity |
| robotics | max_hull_triangles | 262144 | count | Bound physics-engine triangle processing cost |
| walkable | min_covered_fraction | 0.85 | ratio | Floor coverage for navigation |
| walkable | allow_fallback_hulls | true | — | A bounded number of floor-plate fallbacks is tolerated |
| walkable | max_fallback_ratio | 0.25 | ratio | More than 25% failure fallbacks makes the artifact untrustworthy |
| walkable | max_false_fill_fraction | 0.50 | ratio | Coarser environments tolerate more overfill |
| walkable | min_probe_coverage | 0.70 | ratio | Downward artifact-probe coverage ≥70% |
| walkable | max_probe_gap_clusters | 5 | count | ≤5 distinct gap regions in floor collider |
| walkable | min_sweep_traversability | 0.80 | ratio | At least 80% of standable cells belong to one capsule-reachable island |
| walkable | min_standable_fraction | 0.70 | ratio | The configured capsule must fit in at least 70% of grounded cells |
| walkable | max_clearance_blocked_fraction | 0.20 | ratio | At most 20% of grounded cells may be displaced by insufficient headroom |
| interactive | (none) | — | — | Permissive: no gates, all builds accepted |

## Calibration status

These thresholds are **pre-calibration defaults**. They will be re-baselined
against the post-determinism benchmark corpus once it is available. The rationale
column documents the design intent; measured corpus statistics will follow.

A configured strict-profile threshold makes its measurement required. Missing
volume, probe, sweep, latency, size, or complexity data fails the corresponding
check with an actionable suggestion; missing processing is never interpreted as
a pass. Capsule sweep reports also expose radius-blocked fraction and deduplicated
seam-snag count even though the built-in walkable profile does not gate them.

## Optional gates (not enabled by default)

| Field | Unit | Purpose |
|-------|------|---------|
| max_compile_ms | ms | Flag slow compilations |
| max_output_bytes | bytes | Gate oversized .phys files |
| max_hull_count | count | Gate the number of collider hulls |
| max_hull_vertices | count | Gate aggregate collider vertices |
| max_hull_triangles | count | Gate aggregate collider triangles |

These are available on `AcceptancePolicy` for custom profiles. The robotics
profile enables all three complexity budgets; compile latency and output size
remain opt-in.

## Metric units

- **ratio**: [0, 1] fraction (e.g., 0.90 = 90%)
- **count**: non-negative integer
- **ms**: milliseconds
- **bytes**: byte count
