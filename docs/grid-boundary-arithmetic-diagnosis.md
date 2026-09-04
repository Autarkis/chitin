# Policy 0.2 Grid-Boundary Arithmetic Diagnosis

**Issue:** #122

**Evidence:** 14 clips promoted from the spent Policy 0.2 holdout

**Policy status:** remains FAIL; no holdout re-evaluation

## Result

The Policy 0.2 ambiguity band did not miss these cases.

| Observation | Count |
|-------------|------:|
| Diagnostic clips | 14 |
| Mismatching vertices | 42 |
| Mismatches entering the ambiguity path | 42 |
| Ambiguity-band misses | 0 |
| Policy disagreements with exact canonical-f32 inputs | 0 |
| Canonical-f32 clip/topology comparisons agreeing | 14/14 |

For every mismatching vertex, the grid dot product is zero and Policy 0.2
correctly takes its unquantized-f32 fallback. The original f64 inputs retain a
signed separation as small as `5.55e-17` and as large as `2.42e-8`. Casting the
vertex and plane point to f32 maps them to the same representable coordinate.
The fallback therefore returns zero.

Exact rational evaluation of the canonical f32 inputs also returns zero for all
42 vertices. Policy 0.2 agrees with that exact canonical-input result. It differs
only from the exact sign of the higher-precision source inputs.

## Consequence

No wider ambiguity band, ULP margin, or different f32 dot-product evaluation can
recover a sign after the distinguishing source bits have been discarded. These
are input-representation differences, not failures of the filtered predicate's
ambiguity detection or f32 arithmetic.

A Policy 0.3 contract must choose explicitly between:

1. **Canonical f32 inputs:** define the portable predicate over the values that
   can actually enter WGSL. Exact/f64 reference paths first canonicalize vertices,
   plane points, and normals to f32, then evaluate with higher precision. Under
   this contract all 14 diagnostic clips agree in classification and topology.
2. **Preserved source precision:** carry additional information into the GPU
   representation, such as paired-f32 values or integer plane relations. This is
   a representation and precision change, not a classifier-only adjustment.

The first option matches Chitin's stated WGSL-portable objective and existing f32
input boundary. The second has a wider performance and data-layout cost and must
not be adopted implicitly.

## Reproduction

```console
python scripts/diagnose_grid_boundary.py
```

The script records the original exact sign, canonical-f32 exact sign, Policy 0.2
sign, grid dot and bound, ambiguity-path decision, and world-frame f32/f64 dots
for each mismatching vertex.
