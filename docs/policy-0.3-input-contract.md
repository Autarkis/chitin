# Policy 0.3.0 — Canonical IEEE-f32 Input Contract

**Issue:** #122

## Summary

Policy 0.3.0 defines the classifier's portable input contract as canonical
IEEE-754 binary32 (f32) geometry. Vertices, plane points, and plane normals are
evaluated as their nearest f32 representable values. Any sign information present
only in higher-precision source inputs is explicitly outside the contract.

## Rationale

The Policy 0.2.0 holdout (FAIL, commit `0d56ee4`) produced 14 failing clips
with 42 mismatching vertices. Diagnosis (`grid-boundary-arithmetic-diagnosis.md`)
proved that:

1. All 42 vertices enter Policy 0.2's ambiguity path (no band misses).
2. Casting the source vertex and plane point to f32 maps them to the same
   representable coordinate — the signed separation (as small as 5.55e-17) is
   below f32 precision.
3. Exact rational arithmetic over the canonical f32 inputs returns zero for all
   42 cases. Policy 0.2 agrees with that result.
4. No wider epsilon, ULP margin, or alternative f32 dot-product evaluation can
   recover a sign whose distinguishing bits have been discarded.

## Contract

Under Policy 0.3.0:

- **Canonical f32 inputs** are the values obtained by casting each coordinate to
  IEEE-754 binary32 and back. Evaluation — whether by the grid quantization path,
  the unquantized-f32 fallback, or an exact rational oracle — operates on these
  canonical values.
- **Source precision loss** is the condition where the exact sign of the original
  source inputs differs from the exact sign of the canonical f32 inputs. It is
  reported as a distinct category but is not a classifier failure.
- **Genuine f32 arithmetic disagreement** is the condition where the policy's
  classification disagrees with exact rational arithmetic over the canonical f32
  inputs. This is a true classifier defect.
- **On-plane oracle excuse** (per #123) is the condition where the C++ oracle
  returns side=0 for a vertex plausibly on-plane within the f32 error bound. It
  is a tie-breaking convention difference, not a precision defect.

## Why source bits cannot be recovered in WGSL

Chitin's GPU path uploads vertex positions and plane parameters as f32 buffer
attributes. WGSL's `f32` type is IEEE-754 binary32. The cast from f64 source
geometry to f32 buffer values is a one-way precision loss — the extra mantissa
bits are rounded and discarded at buffer-upload time. The GPU shader receives
only the f32 values and cannot reconstruct the original sign.

No predicate operating on f32 inputs can distinguish two source values that map
to the same f32 representable number. This is not a limitation of the classifier
algorithm; it is a property of the input representation.

## What higher precision would require

If Chitin later needs to preserve source-precision signs at the GPU boundary,
the representation must change:

- **Paired-f32 (double-single):** carry each coordinate as two f32 values whose
  sum reconstructs more mantissa bits. Doubles buffer size and requires
  paired-f32 arithmetic in the WGSL shader.
- **Integer plane relations:** encode the side-of-plane relationship as a
  precomputed integer sign, bypassing the dot product entirely. Requires a
  different data layout and a precompute pass on the CPU.
- **f64 extension:** use `f64` in WGSL (not universally supported). Doubles
  buffer size and requires the `shader-f64` WebGPU feature.

Each option is a representation and data-layout change, not a classifier-only
adjustment. Policy 0.3.0 does not adopt any of them.

## Relationship to prior policies

| Policy | Input contract | Ambiguity fallback | Status |
|--------|---------------|--------------------|--------|
| 0.1.0  | Raw source precision (implicit) | No | DEFAULT_POLICY |
| 0.2.0  | Raw source precision (implicit) | Yes (unquantized-f32) | Holdout FAIL |
| 0.3.0  | Canonical f32 (explicit) | Yes (unquantized-f32) | Defined |

Policy 0.3.0 does not change the mechanical parameters (grid_bits, snap_bits,
ambiguity band). It changes what the reference comparison means: the f64 oracle
first canonicalizes its inputs to f32 before evaluating, so only genuine f32
arithmetic errors count as failures.
