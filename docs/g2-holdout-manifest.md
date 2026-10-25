# G2 Holdout Manifest

**Status: FROZEN — UNEVALUATED**
**Frozen: 2026-09-05T UTC**
**Gate: GPU Clipping Pipeline (Phase 10)**
**Contract: `docs/gpu-clipping-pipeline-contract.md`**

This manifest was frozen before any Phase 9 pipeline implementation or
tuning. These fixtures have never been evaluated against GPU clipping
code — CPU or GPU — and must not be until the single formal Phase 10
gate evaluation.

## Fixtures

| Name | Stratum | V | F | C | Structural property |
|------|---------|---|---|---|---------------------|
| `g2_oblique_dipyramid` | oblique geometry | 15 | 26 | 1 | 13-gonal dipyramid rotated 25° around (1,1,1)/√3; min face-normal-to-axis angle 17.75° |
| `g2_razor_star_prism` | near-degenerate edges | 30 | 56 | 1 | 7-point star (R=1.0, r=0.15) extruded 0.003; edge ratio 333:1 |
| `g2_separated_octahedra_trio` | multiple components | 18 | 24 | 3 | 3 unit octahedra at (0,0,0), (8,0,0), (0,8,0) |
| `g2_high_valence_bipyramid` | high-valence vertices | 26 | 48 | 1 | 24-gonal bipyramid; apex valence 24 |
| `g2_multiscale_lattice_box` | mixed-scale geometry | 16 | 28 | 1 | 2×2×2 box + 0.02³ micro-box sharing junction; edge ratio 141:1 |

## Source-geometry SHA-256 digests

Computed as `SHA-256(vertices.tobytes() + faces.tobytes())` over the
`float32` vertices and `int32` faces arrays.

```
g2_oblique_dipyramid:          4f8102ef027c4d1457bd00895f443625bd16f77c8e4ff8e41db7a96720631b18
g2_razor_star_prism:           67132b46c6e476b7fe8b01ba92c7ab8eee1576c16a38108a4525615f9069f000
g2_separated_octahedra_trio:   ad62f3c42f097c27ac757a2bd27e6601be701ca49c9a520fb4c8d0033ef92287
g2_high_valence_bipyramid:     0eaaf66593b8caf7c89f724db2513f8f2b15fbe01f5cafaced92f5d9b2c5d78f
g2_multiscale_lattice_box:     7f95ac764700d64ff9e55ea0bea7c261cd9b2949e8905a9d01d1fe6933091382
```

## Generator parameters

All generators live in `src/chitin/trace_fixtures.py` in the
`G2_HOLDOUT_FIXTURES` dict (separate from `HOLDOUT_FIXTURES`).

| Fixture | Key parameters |
|---------|---------------|
| oblique_dipyramid | n=13 sides, apex_height=1.5, rotation_axis=(1,1,1)/√3, rotation_angle=25° (0.436332 rad) |
| razor_star_prism | n=7 points, R_outer=1.0, R_inner=0.15, extrusion_depth=0.003 |
| separated_octahedra_trio | 3× unit octahedron, offsets (0,0,0), (8,0,0), (0,8,0) |
| high_valence_bipyramid | n=24 sides, apex_height=2.0, radius=1.0 |
| multiscale_lattice_box | large_side=2.0, small_side=0.02, junction at y=1 |

## Admission checks passed

All fixtures verified (source-only, no clipping):

- Finite coordinates (no NaN/Inf)
- Valid face indices (0 ≤ idx < V)
- Manifold: every edge shared by exactly 2 faces
- Watertight: no boundary edges
- Correct component count
- Correct vertex/face counts and dtypes (float32/int32)
- Deterministic: identical digests on repeated generation
- Structural properties match declared strata (see table above)

## Contamination status

These fixtures are **disjoint** from all spent corpora:

- Not in `HOLDOUT_FIXTURES` (Policy 0.2.0 gate: oblique_gear_prism, twisted_notched_column, skewed_rectangular_torus, multiscale_shard_cluster)
- Not in holdout-corpus-0.3.0.json (Policy 0.3.0 gate: barbed_helix_prism family, fluted_twist_column, ridged_torus, interlocked_frame)
- Not in calibration/development fixtures (box, l_shape, t_shape, thin_panel, etc.)
- No trace data captured, no clipping planes generated, no CPU or GPU clipping evaluated

## Rules

1. **Do not modify generators** after this freeze.
2. **Do not run GPU or CPU clipping** against these fixtures until Phase 10.
3. **Do not add G2 fixtures** to any parametrized test that exercises clipping code.
4. Source-only admission checks (geometry validity) are permitted.
5. The single Phase 10 evaluation will use the capture and evaluation protocol
   defined in `docs/gpu-clipping-pipeline-contract.md`.
