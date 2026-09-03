# CoACD Instrumentation Spec — Clip Oracle Recording

**Issue**: #108 (trace infrastructure), blocking #101 (f32 gate)
**Date**: 2026-09-02
**Status**: Specification — not yet implemented

## Problem

The current trace hooks record the *output* of each clip (positive/negative
meshes) but not:

1. **The input mesh at `Clip()` entry** — the component mesh being clipped,
   with stable vertex IDs that survive through the operation.
2. **Per-triangle classification decisions** — the actual `Side`/`CutSide`
   enum values the C++ code assigns to each source triangle.

Without these, the Python replay reconstructs an approximate input by
concatenating the two outputs (`np.vstack([pos_v, neg_v])`), which:
- Duplicates cut vertices (intersection points appear in both halves)
- Loses original vertex/edge identity
- Introduces intersection vertices that are then classified as if they
  belonged to the pre-clip mesh

And the comparison is f64-NumPy vs f32-NumPy, not f32-Python vs C++-oracle.

## Required C++ trace additions

### 1. Record input mesh at Clip() entry

In the clip trace hook, before the clip operation executes, emit:

```jsonl
{"event": "clip", ..., "input_verts": "clip_NNN_input_verts.npy", "input_faces": "clip_NNN_input_faces.npy", ...}
```

Save the component mesh vertices and faces as `.npy` files, exactly as they
exist at the entry to `Clip()`. These are the vertices the plane classifies —
the ground truth input.

### 2. Record per-triangle Side decisions

After `ClassifyPoints` (or the equivalent per-vertex/per-triangle
classification), emit the classification array:

```jsonl
{"event": "clip_classify", "clip_index": NNN, "sides": "clip_NNN_sides.npy"}
```

Where `sides.npy` is an `int8` array of length `num_vertices`, containing
CoACD's `Side` enum values for each vertex:

| Value | Meaning |
|-------|---------|
| -1    | Negative (behind plane) |
|  0    | On plane |
| +1    | Positive (in front of plane) |

If CoACD uses a different enum (e.g., `NEGATIVE=0, ON=1, POSITIVE=2`), map
to signed convention in the trace hook to avoid ambiguity.

### 3. Record CutSide decisions (optional but valuable)

If CoACD makes per-*edge* or per-*face* intersection decisions separately
from per-vertex classification (e.g., deciding which edges to split and how),
record those too:

```jsonl
{"event": "clip_cut", "clip_index": NNN, "cut_edges": "clip_NNN_cut_edges.npy", "cut_points": "clip_NNN_cut_points.npy"}
```

Where:
- `cut_edges.npy`: `(K, 2) int32` — pairs of vertex indices for each split edge
- `cut_points.npy`: `(K, 3) float64` — the computed intersection points

This lets the replay verify intersection computation, not just classification.

## Python-side changes

### coacd_trace.py

Add fields to `TracedClip`:

```python
@dataclass
class TracedClip:
    # ... existing fields ...
    input_vertices: np.ndarray | None = None  # (V, 3) float64
    input_faces: np.ndarray | None = None  # (F, 3) int32
    oracle_sides: np.ndarray | None = None  # (V,) int8
    cut_edges: np.ndarray | None = None  # (K, 2) int32
    cut_points: np.ndarray | None = None  # (K, 3) float64
```

Update `_load_trace_file` to load these from the new `.npy` files when present.

### coacd_trace_replay.py

Replace the input-mesh reconstruction:

```python
# BEFORE (wrong — reconstructs from outputs):
vertices = np.vstack([pos_v, neg_v])

# AFTER (correct — uses recorded input):
if clip.input_vertices is not None:
    vertices = clip.input_vertices
    faces = clip.input_faces
else:
    # Legacy fallback for old traces without input recording
    vertices = np.vstack([pos_v, neg_v])
    ...
```

Add oracle comparison:

```python
def compare_against_oracle(
    clip: TracedClip, policy: QuantizationPolicy
) -> OracleComparison:
    """Compare f32 classification directly against C++ oracle decisions."""
    if clip.oracle_sides is None:
        return None  # trace doesn't have oracle data

    vertices = clip.input_vertices
    point = ...  # derive from plane
    normal = ...  # derive from plane

    f32_sides = classify_plane_f32(vertices, point, normal, policy)

    # Direct comparison: f32 Python vs C++ oracle
    agree = np.array_equal(f32_sides, clip.oracle_sides)
    if not agree:
        mismatches = np.where(f32_sides != clip.oracle_sides)[0]
        ...
```

## Trace format versioning

Add a `trace_version` field to the `begin` event:

```jsonl
{"event": "begin", "trace_version": 2, ...}
```

- Version 1: current format (pos/neg outputs only)
- Version 2: adds input mesh + oracle sides + optional cut data

The Python loader should handle both versions gracefully.

## Compact corpus format

The full trace corpus (29K files, 1.3 GiB) is too large for git. For CI:

1. **Minimal representative corpus**: One trace per fixture, capped at ~100
   clips each, stored as a single `.npz` archive per fixture (~10 MB total).
2. **Full corpus**: Stored as a release artifact or in CAS, referenced by
   SHA-256 digest in `tests/fixtures/traces/DIGEST`.
3. **Regeneration**: `python scripts/capture_trace_corpus.py` with traced DLL.

The `.npz` archive bundles all the `.npy` blobs for a trace into one file,
eliminating the 29K-file problem. The JSONL metadata goes in as a text entry.

## Validation

Once instrumented, the gate must verify:

1. `f32_classify(input_mesh, plane) == oracle_sides` — direct comparison
2. Where they disagree, measure the dot-product magnitude at divergent
   vertices (near-plane vertices are expected; far-from-plane divergence is a
   bug in the f32 path)
3. Full clip topology comparison using the real input mesh (not reconstruction)
4. Sweep across policies
5. Assert threshold in CI

## Pinned CoACD revision

The instrumentation patch MUST be versioned against a specific CoACD commit:

- **CoACD upstream**: https://github.com/SarahWeiii/CoACD
- **Pinned revision**: `v1.0.14` (tag) — the version used for the v1 trace corpus
- **Patch format**: A git-format-patch against the pinned revision, stored at
  `tools/coacd-v2-instrumentation.patch`
- **Build script**: `tools/build-traced-coacd.sh` — clones the pinned revision,
  applies the patch, builds the traced DLL, and verifies the trace output format

The patch and build script are the reproducibility contract. A prose specification
alone cannot reproduce #108. The build script must:

1. `git clone --branch v1.0.14 https://github.com/SarahWeiii/CoACD.git`
2. `git apply ../coacd-v2-instrumentation.patch`
3. Build with CMake (same flags as upstream)
4. Run a smoke test: trace a unit cube, verify the output contains
   `input_verts`, `oracle_sides`, and `cut_points` fields
5. Print the DLL path and SHA-256 digest

## Calibration and holdout protocol

The regression floors (90%/85%) are regression alarms, not acceptance gates.
Final acceptance thresholds MUST follow the calibration/holdout protocol:

### Protocol

1. **Capture calibration corpus** — run `capture_trace_corpus.py` with the v2
   traced DLL on the existing 10 fixtures. This corpus is for tuning only.

2. **Select policy** — sweep `grid_bits` and `classification_ulp_margin` on the
   calibration corpus. Choose the policy that preserves the best
   agreement/performance tradeoff.

3. **Freeze policy and acceptance criteria** — write the chosen policy and
   threshold into `src/chitin/f32_policy.py` as `GATE_POLICY`. Write the
   acceptance threshold into `tests/test_trace_replay.py` as `ACCEPTANCE_GATE`
   (distinct from the regression floor). Document the rationale.

4. **Capture holdout corpus** — generate NEW meshes not in the calibration set
   (different seeds, different scales, additional fixture types). Run
   `capture_trace_corpus.py` on these.

5. **Evaluate once** — run the frozen policy on the holdout corpus. Record
   PASS or FAIL. Do not tune after seeing holdout results.

6. **Record verdict** — update `docs/f32-gate-results.md` with the holdout
   result. Close #108. Close or reopen #101 according to the verdict.

### Why

Choosing thresholds after seeing the same corpus they're measured on is
overfitting to the calibration set. A policy that scores 93.7% on the
calibration corpus and is then gated at 90% has never been tested — the 90%
was chosen to pass. The holdout corpus is the actual measurement.

## CI corpus contract

Tests that silently skip in CI mean the gate does not exist outside the
development machine. CI must either:

1. **Download a digest-verified compact corpus** from a release artifact or CAS
   before running tests. The digest is stored in
   `tests/fixtures/traces/DIGEST` (SHA-256 of the `.tar.zst` archive).

2. **Or set `CHITIN_GATE_FINAL=1`** so that missing corpus = hard failure,
   not silent skip.

The compact corpus format is a single `.tar.zst` archive per fixture
containing the JSONL metadata and `.npy` blobs (~10 MB total for 6 fixtures
at 100 clips each). The archive is generated by
`scripts/package_ci_corpus.py` (to be written) and uploaded as a GitHub
release artifact.

CI workflow pseudo-code:
```yaml
- name: Download trace corpus
  run: |
    DIGEST=$(cat tests/fixtures/traces/DIGEST)
    gh release download v0.X.0 -p "trace-corpus-v2.tar.zst"
    echo "$DIGEST  trace-corpus-v2.tar.zst" | sha256sum -c
    tar xf trace-corpus-v2.tar.zst -C tests/fixtures/traces/

- name: Run f32 gate
  env:
    CHITIN_GATE_FINAL: "1"
  run: python -m pytest tests/test_trace_replay.py -v
```
