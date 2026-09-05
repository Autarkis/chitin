# GPU Clipping Pipeline Contract

Contract for the GPU-resident mesh/plane clipping pipeline. Frozen before
implementation; changes require a versioned amendment with rationale.

Refs: #92, #93, #103, #120.

## Scope

Given an indexed triangle mesh and a clipping plane, produce two closed,
consistently wound meshes (positive half and negative half) through a
GPU-resident pipeline with no intermediate host readback between the
classify-through-weld stages.

The pipeline is a stage within chitin's convex-decomposition compiler. It
does not own semantic input validation (manifold check, watertight check)
or final `.phys` serialization. It owns everything between receiving
GPU-resident buffers and returning two closed half-meshes.

### Validation boundary

The pipeline performs **defensive validation** — rejecting provably
invalid inputs (NaN/Inf coordinates, out-of-range indices, zero-length
plane normals). This is a safety net, not an admission gate.

The pipeline does **not** perform **semantic validation** — manifold
checks, watertight checks, or degeneracy analysis. Degenerate triangles
(zero area, collinear vertices) are processed, not rejected; they may
produce degenerate output (zero-area faces, collapsed edges). The caller
is responsible for providing a valid indexed mesh for its domain.

`INVALID_INPUT` fires only on defensive violations. A degenerate triangle
is not an invalid input.

## Input contract

All inputs are IEEE-754 binary32 (f32), following Policy 0.3.0
(`docs/policy-0.3-input-contract.md`). Source geometry in higher precision
is cast to nearest f32 before upload — sign information present only in
the higher-precision source is outside the contract.

### Plane representation

Policy 0.3.0 requires `plane_point` and `plane_normal` as separate
vectors because the grid-frame normalization computes centroid and scale
over `concatenate([vertices, plane_point])` — the point participates in
the centroid. The implicit form `(a, b, c, d)` loses the original point
and cannot reproduce Policy 0.3 classification.

The pipeline accepts planes as `(plane_point, plane_normal)` pairs using
the `ClipPlane` layout. The existing `PLANE` layout `(a, b, c, d)` is
retained for other uses (hull normals, general half-spaces) but is **not**
the clipping pipeline's input format.

### Buffers

| Buffer | Layout | Stride | Description |
|--------|--------|--------|-------------|
| `vertices` | `Point` (f32 x, y, z + pad) | 16B | Mesh vertex positions |
| `faces` | `Triangle` (u32 i0, i1, i2 + pad) | 16B | Triangle index triples, 0-based into `vertices` |
| `planes` | `ClipPlane` (f32 px, py, pz, pad, nx, ny, nz, pad) | 32B | Clipping plane point + normal |
| `mesh_headers` | `MeshHeader` (vertex_offset, vertex_count, face_offset, face_count) | 16B | Per-mesh ranges into shared vertex/face arrays |

All layouts are defined in `src/chitin/gpu/layouts.py` and are
authoritative. WGSL struct declarations are generated from Python and
drift-checked at test time.

#### New layout: `ClipPlane`

```
struct ClipPlane {          // 32 bytes, 16-byte aligned
    point_x:  f32,          //  0
    point_y:  f32,          //  4
    point_z:  f32,          //  8
    _pad0:    u32,          // 12
    normal_x: f32,          // 16
    normal_y: f32,          // 20
    normal_z: f32,          // 24
    _pad1:    u32,          // 28
}
```

#### New layout: `EdgeKey`

Canonical source-edge identifier. Two `u32` vertex indices in ascending
order: `v_lo = min(global_a, global_b)`, `v_hi = max(global_a, global_b)`.
Collision-free — no two distinct edges share a key.

```
struct EdgeKey {            // 8 bytes
    v_lo: u32,              // 0
    v_hi: u32,              // 4
}
```

#### New layout: `IntersectionRecord`

Extends the existing `IntersectionPoint` with the full edge key. Used
during emit and weld; the final output retains `IntersectionPoint` for
backward compatibility.

```
struct IntersectionRecord {  // 32 bytes, 16-byte aligned
    x:     f32,              //  0
    y:     f32,              //  4
    z:     f32,              //  8
    _pad0: u32,              // 12
    v_lo:  u32,              // 16  (EdgeKey.v_lo)
    v_hi:  u32,              // 20  (EdgeKey.v_hi)
    _pad1: u32,              // 24
    _pad2: u32,              // 28
}
```

### Dispatch

A single pipeline invocation processes `M` meshes against `P` planes,
producing `M × P` clip operations. Each operation is independent — no
shared state between (mesh_i, plane_j) and (mesh_k, plane_l).

### Preconditions (defensive)

The pipeline rejects — not silently ignores — inputs that violate these:

- All vertex indices in `faces` are in range `[0, vertex_count)` for
  their mesh.
- No NaN or Inf in vertex positions or plane coefficients.
- Plane normals are finite and non-zero (normalization is not required;
  classification uses the raw dot product sign, but a zero normal makes
  every distance zero).

## Pipeline stages

```
vertices, faces, planes, mesh_headers  (GPU-resident)
        │
        ▼
   ┌─────────┐
   │ Classify │  Per-vertex signed distance, three-state sign
   └────┬────┘   Policy 0.3 f32 semantics
        │
        ▼
   ┌─────────┐
   │  Count  │  Per-triangle: emitted geometry per side
   └────┬────┘   Full {-1,0,+1}³ truth table
        │
        ▼
   ┌─────────┐
   │  Scan   │  Multi-workgroup prefix sum → output offsets
   └────┬────┘   Logical counts within preallocated arenas
        │
        ▼
   ┌─────────┐
   │  Emit   │  Write positive/negative faces + intersections
   └────┬────┘   Source-edge keyed, source-face ancestry
        │
        ▼
   ┌─────────┐
   │ Compact │  Source-vertex compaction + local index rewrite
   └────┬────┘   Per-side 0-based output indices
        │
        ▼
   ┌─────────┐
   │  Weld   │  Deduplicate intersections by EdgeKey
   └────┬────┘   Rewrite face indices to welded vertices
        │
        ▼
   ┌─────────┐
   │  Loop   │  Extract closed boundary loops
   └────┬────┘
        │
        ▼
   ┌──────────┐
   │   Cap    │  Triangulate loops (ear-clipping), attach
   └────┬─────┘  Opposite winding per half
        │
        ▼
   ClipPipelineResult  (GPU-resident until explicit readback)
```

No intermediate host readback occurs between classify and weld. All
inter-stage data lives in GPU storage buffers within preallocated arenas.

Cap triangulation (loop extraction + ear-clipping) may readback boundary
loops to the host if the GPU ear-clipping kernel is not yet implemented.
This is an explicit, temporary concession documented in the stage
description. The classify → count → scan → emit → compact → weld chain
is strictly GPU-resident.

### Stage 1: Classify

Per vertex, compute signed distance to each plane in two frames.

#### Grid-frame normalization

For each (mesh, plane) pair:

1. Concatenate the mesh's vertices with the plane point:
   `combined = concat(vertices[mesh_offset:mesh_offset+vertex_count], plane_point)`.
2. Compute centroid: `centroid = mean(combined, axis=0)`.
3. Compute extent: `extent = max(abs(combined - centroid))`, floored at
   `1e-30` to prevent division by zero on degenerate input.
4. Compute scale: `scale_factor = 1.0 / (2.0 * extent)`.
5. Grid-scale: `grid_coords = round((combined - centroid) * scale_factor * 2^grid_bits)`.
6. Split: `grid_vertices = grid_coords[:-1]`, `grid_plane_point = grid_coords[-1]`.
7. Grid normal: `grid_normal = (plane_normal * scale_factor).as_f32()`.

This sequence is the exact port of `_to_grid_frame` +
`QuantizationPolicy.normalize_to_grid`. The plane point's participation in
the centroid is load-bearing — omitting it changes the grid frame and
produces different classifications.

#### Classification

Grid-frame dot product per vertex:

```
grid_dot[i] = dot(grid_vertices[i] - grid_plane_point, grid_normal)
```

Evaluated in f32 arithmetic. Three-state sign assignment:

| Condition | Sign |
|-----------|------|
| `abs(grid_dot) > classification_ulp_margin` and `grid_dot > 0` | `+1` |
| `abs(grid_dot) > classification_ulp_margin` and `grid_dot < 0` | `-1` |
| `abs(grid_dot) <= classification_ulp_margin` | Ambiguous |

Ambiguous vertices proceed to the world-frame fallback:

```
world_dot[i] = dot(f32(vertex[i]) - f32(plane_point), f32(plane_normal))
sign[i] = sign(world_dot[i])     // returns -1, 0, or +1
```

A vertex whose world-frame dot is exactly 0.0 remains **on-plane
(sign = 0)**. The pipeline does not force-resolve on-plane vertices.

Output: per-vertex `i32` sign buffer with values in `{-1, 0, +1}`.

Counters: `fast_path_count` (unambiguous grid classification),
`ambiguity_fallback_count`, `on_plane_count`.

### Stage 2: Count

Per triangle, given its three vertex signs `(s0, s1, s2)` where each is
in `{-1, 0, +1}`, determine emission geometry. The positive side keeps
vertices with `sign >= 0`; the negative side keeps vertices with
`sign <= 0`. On-plane vertices belong to **both** sides.

Complete truth table (27 cases, grouped by behavior):

#### All same side (no clipping)

| Signs | Positive emission | Negative emission | Cut edges | Boundary |
|-------|-------------------|-------------------|-----------|----------|
| (+,+,+) | 1 tri (3V, 1F) | — | 0 | — |
| (-,-,-) | — | 1 tri (3V, 1F) | 0 | — |

#### All non-negative (positive side takes whole triangle)

| Signs | Positive emission | Negative emission | Cut edges | Boundary |
|-------|-------------------|-------------------|-----------|----------|
| (+,+,0) | 1 tri (3V, 1F) | — | 0 | on-plane vertex is boundary |
| (+,0,+) | 1 tri (3V, 1F) | — | 0 | on-plane vertex is boundary |
| (0,+,+) | 1 tri (3V, 1F) | — | 0 | on-plane vertex is boundary |
| (+,0,0) | 1 tri (3V, 1F) | — | 0 | on-plane edge is boundary |
| (0,+,0) | 1 tri (3V, 1F) | — | 0 | on-plane edge is boundary |
| (0,0,+) | 1 tri (3V, 1F) | — | 0 | on-plane edge is boundary |

#### All coplanar

| Signs | Positive emission | Negative emission | Cut edges | Boundary |
|-------|-------------------|-------------------|-----------|----------|
| (0,0,0) | 1 tri (3V, 1F) | 1 tri (3V, 1F) | 0 | entire triangle is boundary |

Coplanar triangles are assigned to the positive side by default (matching
`sign >= 0`), **and** duplicated to the negative side. Status:
`COPLANAR_FACES` set in metrics. This dual-assignment ensures both halves
remain closed at coplanar regions.

#### All non-positive (negative side takes whole triangle)

| Signs | Positive emission | Negative emission | Cut edges | Boundary |
|-------|-------------------|-------------------|-----------|----------|
| (-,-,0) | — | 1 tri (3V, 1F) | 0 | on-plane vertex is boundary |
| (-,0,-) | — | 1 tri (3V, 1F) | 0 | on-plane vertex is boundary |
| (0,-,-) | — | 1 tri (3V, 1F) | 0 | on-plane vertex is boundary |
| (-,0,0) | — | 1 tri (3V, 1F) | 0 | on-plane edge is boundary |
| (0,-,0) | — | 1 tri (3V, 1F) | 0 | on-plane edge is boundary |
| (0,0,-) | — | 1 tri (3V, 1F) | 0 | on-plane edge is boundary |

#### Mixed: two positive, one negative (2 cut edges)

| Signs | Positive emission | Negative emission | Cut edges | Boundary edges |
|-------|-------------------|-------------------|-----------|----------------|
| (+,+,-) | quad → 2 tri (4V, 2F) | 1 tri (3V, 1F) | 2 | 1 (between 2 intersections) |
| (+,-,+) | quad → 2 tri (4V, 2F) | 1 tri (3V, 1F) | 2 | 1 |
| (-,+,+) | quad → 2 tri (4V, 2F) | 1 tri (3V, 1F) | 2 | 1 |

#### Mixed: one positive, two negative (2 cut edges)

| Signs | Positive emission | Negative emission | Cut edges | Boundary edges |
|-------|-------------------|-------------------|-----------|----------------|
| (+,-,-) | 1 tri (3V, 1F) | quad → 2 tri (4V, 2F) | 2 | 1 |
| (-,+,-) | 1 tri (3V, 1F) | quad → 2 tri (4V, 2F) | 2 | 1 |
| (-,-,+) | 1 tri (3V, 1F) | quad → 2 tri (4V, 2F) | 2 | 1 |

#### Mixed with on-plane vertex (1 cut edge)

| Signs | Positive emission | Negative emission | Cut edges | Boundary edges |
|-------|-------------------|-------------------|-----------|----------------|
| (0,+,-) | 1 tri (3V, 1F) | 1 tri (3V, 1F) | 1 | 1 (on-plane vertex to intersection) |
| (0,-,+) | 1 tri (3V, 1F) | 1 tri (3V, 1F) | 1 | 1 |
| (+,0,-) | 1 tri (3V, 1F) | 1 tri (3V, 1F) | 1 | 1 |
| (-,0,+) | 1 tri (3V, 1F) | 1 tri (3V, 1F) | 1 | 1 |
| (+,-,0) | 1 tri (3V, 1F) | 1 tri (3V, 1F) | 1 | 1 |
| (-,+,0) | 1 tri (3V, 1F) | 1 tri (3V, 1F) | 1 | 1 |

In these cases the boundary edge runs from the on-plane source vertex
to the single intersection vertex. The on-plane vertex is shared between
both halves (it exists in both output vertex arrays).

#### Boundary-edge accounting

A boundary edge is a segment lying in the clipping plane that separates
positive-side geometry from negative-side geometry. Boundary edges come
from two sources:

1. **Cut edges**: the segment between two intersection vertices created
   where two edges of the same triangle cross the plane (the 2-cut cases
   above).
2. **On-plane edges**: source edges where both endpoints have sign 0, or
   segments from an on-plane vertex to an intersection (the 1-cut cases
   above).

Both types participate in loop extraction. On-plane source vertices are
**boundary vertices** and their indices appear in boundary edges directly
(no intersection record needed).

### Stage 3: Scan

Multi-workgroup exclusive prefix sum over the per-triangle counts,
producing:

- Per-triangle write offsets into the output vertex and face arrays.
- Per-triangle write offsets into the intersection and boundary-edge
  arrays.
- Total logical counts (final element of the inclusive scan).

The scan operates across workgroups using a standard reduce-then-propagate
pattern. Not bounded to 256 elements.

Total logical counts are used to validate that output fits within
preallocated arenas (see Capacity). No host readback of these totals
occurs — the check is a GPU-side comparison against the arena's
preallocated capacity, writing the `CAPACITY_EXCEEDED` status code if
exceeded.

### Stage 4: Emit

Per triangle, using the scanned offsets:

**Kept triangles** (all same side): write the triangle's three vertex
indices into the appropriate side's face array.

**Cut triangles**: for each edge that crosses the plane (endpoints have
signs of opposite nonzero polarity, i.e., `sign_a * sign_b < 0`), compute
the intersection point:

```
d_a = world_dot[idx_a]
d_b = world_dot[idx_b]
t = d_a / (d_a - d_b)
intersection = vertex[idx_a] + t * (vertex[idx_b] - vertex[idx_a])
```

All in f32 arithmetic. `world_dot` is the signed-distance array computed
during classification (the world-frame f32 dot products).

The intersection is keyed by its **canonical source edge**:
`EdgeKey(min(global_a, global_b), max(global_a, global_b))`.
Two adjacent triangles sharing a source edge produce the same `EdgeKey`
and, after welding, share the same output vertex.

**On-plane vertex mixed triangles** (1 cut edge): the on-plane vertex
appears in both sides' face arrays. The single intersection vertex and
the on-plane vertex together form the boundary edge.

Fan-triangulate the resulting polygon for each side. Record boundary
edges.

Preserve **source-face ancestry**: each emitted face records the index of
the source triangle it came from (for topology parity testing).

At this stage, face indices reference **global** vertex indices (into the
shared input vertex array for source vertices, into the per-operation
intersection array for new vertices). Compaction to local indices happens
in the next stage.

### Stage 5: Compact

For each clip operation, for each side (positive/negative):

1. Scan the emitted face indices to identify the set of referenced source
   vertex indices.
2. Build a compaction map: `global_index → local_index` for referenced
   vertices only.
3. Copy referenced source vertex positions into a compact
   `positive_vertices` / `negative_vertices` array.
4. Append intersection vertices (post-weld) to the compact array.
5. Rewrite all face indices from global to local.

After compaction, each side's faces are 0-based into its own vertex array.
The output meshes are self-contained.

Compaction ordering is deterministic: source vertices appear in ascending
global-index order, followed by intersection vertices in ascending
`EdgeKey` order.

### Stage 6: Weld

Deduplicate intersection vertices by `EdgeKey`:

1. Collect all `(EdgeKey, intersection_position, pre_weld_index)` records.
2. Sort by `EdgeKey` (lexicographic on `(v_lo, v_hi)`).
3. For each unique `EdgeKey`, select the first record in sort order as
   canonical.
4. Rewrite all face indices referencing a duplicate to the canonical
   index.
5. Compact the intersection array to remove duplicates.

**Invariant**: after welding, every intersection on the same source edge
maps to exactly one output vertex. Two triangles sharing a source edge
reference the same welded intersection vertex index.

Output: welded intersection vertices, rewritten face indices, welded
boundary edges (pairs of welded vertex indices and/or on-plane source
vertex indices).

### Stage 7: Loop extraction

From the welded boundary edges, extract closed loops:

1. Build adjacency: each boundary vertex connects to its neighbors via
   boundary edges.
2. Walk the adjacency to extract closed loops. Walk order is
   deterministic: at each vertex, choose the neighbor with the smallest
   index not yet visited.
3. Every boundary vertex has valence exactly 2 in the boundary graph
   (for manifold input). Branching (valence > 2) or dangling (valence 1)
   vertices indicate a topology error — reported as `TOPOLOGY_ERROR`.
4. Multiple disconnected loops are extracted independently.

Each loop is a closed polygon whose vertices lie in (or near) the
clipping plane.

### Stage 8: Cap triangulation

For each closed boundary loop:

1. Project loop vertices onto the clipping plane's 2D coordinate system
   (choose the two axes with largest plane-normal component discarded).
2. Classify the projected polygon:
   - **Convex**: fan-triangulate from vertex 0.
   - **Simple non-convex**: ear-clipping triangulation (deterministic:
     always remove the first valid ear in index order).
   - **Self-intersecting or degenerate (< 3 vertices)**: report
     `TOPOLOGY_ERROR`.
3. Attach cap faces to the positive-side mesh with winding such that the
   cap normal points in the same direction as the clipping plane normal.
4. Attach cap faces to the negative-side mesh with opposite winding.
5. Multiple disconnected loops produce independent caps attached to the
   same output mesh.

Ear-clipping is O(n²) per loop but loops from convex-decomposition splits
are typically small (3–30 vertices). The pipeline may implement
ear-clipping on the GPU (one workgroup per loop) or read back boundary
loops to the host for CPU ear-clipping. If CPU fallback is used, this is
the **only** host readback in the pipeline, occurring after weld and
before final result assembly. The readback is explicitly permitted by this
contract as a temporary measure until GPU ear-clipping is implemented.

Nested loops (holes) are a future extension. In v1, each loop is
triangulated independently. A mesh section that produces nested loops
will generate overlapping caps — geometrically incorrect but bounded.
The status code `NESTED_LOOPS` is reserved for future detection.

## Output contract

### Per clip operation

| Field | Type | Description |
|-------|------|-------------|
| `positive_vertices` | `Point[]` | Vertex positions for positive half, 0-based |
| `positive_faces` | `Triangle[]` | Face indices for positive half, 0-based into `positive_vertices` |
| `negative_vertices` | `Point[]` | Vertex positions for negative half, 0-based |
| `negative_faces` | `Triangle[]` | Face indices for negative half, 0-based into `negative_vertices` |
| `intersection_records` | `IntersectionRecord[]` | Welded intersections with `EdgeKey` |
| `positive_face_ancestry` | `u32[]` | Source triangle index per positive face |
| `negative_face_ancestry` | `u32[]` | Source triangle index per negative face |
| `status` | `u32` | Operation status code |
| `metrics` | see below | Per-operation counters |

### Status codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `OK` | Both halves produced, closed and capped |
| 1 | `ALL_POSITIVE` | Entire mesh on positive side; negative half empty |
| 2 | `ALL_NEGATIVE` | Entire mesh on negative side; positive half empty |
| 3 | `ALL_COPLANAR` | Entire mesh coplanar with plane; assigned to both sides |
| 4 | `CAPACITY_EXCEEDED` | Output would exceed arena capacity. No partial result |
| 5 | `INVALID_INPUT` | Defensive precondition violated (NaN, out-of-range, zero normal) |
| 6 | `TOPOLOGY_ERROR` | Boundary graph not manifold (branching, dangling, self-intersecting loop) |
| 7 | `UNSUPPORTED_TOPOLOGY` | Reserved: nested loops, non-simple polygons |

`ALL_POSITIVE`, `ALL_NEGATIVE`, and `ALL_COPLANAR` are success states.

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `classify_fast_count` | `u32` | Vertices classified without ambiguity fallback |
| `classify_fallback_count` | `u32` | Vertices requiring ambiguity fallback |
| `classify_on_plane_count` | `u32` | Vertices with final sign = 0 |
| `input_face_count` | `u32` | Source triangle count |
| `positive_face_count` | `u32` | Output face count (positive, including caps) |
| `negative_face_count` | `u32` | Output face count (negative, including caps) |
| `coplanar_face_count` | `u32` | Source triangles assigned to both sides |
| `intersection_count_raw` | `u32` | Intersection vertices before welding |
| `intersection_count_welded` | `u32` | Intersection vertices after welding |
| `boundary_loop_count` | `u32` | Number of closed boundary loops |
| `cap_face_count` | `u32` | Total cap faces (both halves combined) |
| `stage_times_us` | `u32[8]` | Per-stage elapsed microseconds (optional) |

`stage_times_us` requires WebGPU timestamp-query support
(`timestamp-query` feature). When unavailable, all entries are 0 and the
`timestamps_available` flag in the result is `false`. Stage timing is
**informational, never mandatory** — no gate criterion depends on it.

## Invariants

These hold for every operation with status `OK`:

1. **Finite coordinates.** Every output vertex has finite f32 components.
   No NaN, no Inf.

2. **Valid indices.** Every face index is in range `[0, vertex_count)` for
   its output mesh (local, 0-based after compaction).

3. **Welded intersections.** Every intersection on the same source edge
   (same `EdgeKey`) maps to exactly one output vertex index. The welded
   count is strictly less than or equal to the raw count.

4. **Closed boundary.** Every boundary vertex has valence exactly 2 in
   the welded boundary-edge graph (excluding on-plane-only edges in
   coplanar regions).

5. **Closed caps.** Every boundary loop is closed. Cap faces triangulate
   the loop completely — no gaps, no overlaps.

6. **Consistent winding.** Positive-half cap normals point in the same
   direction as the clipping plane normal. Negative-half cap normals
   point opposite.

7. **Face conservation.** The total faces emitted from source triangles
   (excluding caps) on both sides accounts for every source face exactly
   once (coplanar faces counted on both sides). No face is lost.

8. **Determinism.** Identical inputs produce identical outputs across
   invocations on the same adapter. Cross-adapter determinism is not
   guaranteed (different rounding or workgroup scheduling), but
   classification parity with the CPU reference is required on identical
   f32 inputs.

9. **No intermediate readback (classify→weld).** All inter-stage data
   between classify and weld remains GPU-resident. The only permitted
   host transfer before final readback is boundary-loop data for CPU
   cap triangulation (temporary, see Stage 8).

10. **Self-contained output.** Each side's face indices are 0-based into
    its own compact vertex array. No references to the input vertex
    buffer.

## Capacity

### Preallocated arenas

The pipeline preallocates output arenas sized from input mesh dimensions,
not from GPU-computed totals. This eliminates any sizing readback.

Upper bounds per clip operation on mesh with `V` vertices and `F` faces:

| Arena | Worst-case elements | Rationale |
|-------|---------------------|-----------|
| Sign buffer | `V` | One sign per vertex |
| Per-triangle counts | `F` | One count record per face |
| Scan buffer | `F` | Prefix-sum workspace |
| Output vertices (per side) | `V + 2F` | All source vertices + max 2 intersections per face |
| Output faces (per side) | `2F` | Each face emits at most 2 triangles per side |
| Intersection records | `2F` | Max 2 per face (before welding) |
| Boundary edges | `F` | Max 1 per face |
| Cap faces (per side) | `2F` | Bounded by boundary vertex count minus 2 per loop |

Before allocation, the pipeline checks whether the arena sizes fit within
`GPUWorker.limits.max_buffer_size`. If not, the operation returns
`CAPACITY_EXCEEDED` before any GPU work.

After the scan, the exact logical counts are available GPU-side. A
GPU-side comparison confirms they fit within the preallocated arenas.
Logical overflow (totals exceed preallocated capacity — should be
impossible given the bounds, but checked defensively) writes
`CAPACITY_EXCEEDED` to the status buffer.

### Scratch management

Inter-stage temporaries are allocated from reusable scratch arenas:

- Allocated on first use, retained across operations within a session.
- Grown geometrically when needed (never shrunk within a session).
- Released on session teardown or device loss.

## Error handling

| Condition | Behavior |
|-----------|----------|
| Device lost during pipeline | `DeviceLostError`. Session invalidated. No partial result |
| Cancellation requested | `OperationCancelledError` at next command boundary. No partial result |
| Output exceeds preallocated arena | `CAPACITY_EXCEEDED` status. No partial result |
| Invalid input detected | `INVALID_INPUT` status. No partial result |
| Topology error in boundary | `TOPOLOGY_ERROR` status. No partial result |

**No partial results.** Every error or non-OK status (except
`ALL_POSITIVE`/`ALL_NEGATIVE`/`ALL_COPLANAR`) returns a complete failure.
The caller either gets valid closed meshes or a typed error.

## Relationship to existing infrastructure

### Policy 0.3.0

Classification uses Policy 0.3.0 f32 semantics exactly. The GPU
classifier is a port of the CPU classifier (`classify_plane_f32` +
`_classify_with_fallback` + `_to_grid_frame` in
`src/chitin/f32_predicates.py`), not a reimplementation. Zero
classification mismatches between GPU and CPU on identical f32 inputs is
a gate requirement.

### Layouts

All GPU struct layouts are defined in `src/chitin/gpu/layouts.py`.
The clipping pipeline adds `ClipPlane`, `EdgeKey`, and
`IntersectionRecord` to the authoritative layout set. The existing
`IntersectionPoint` layout is retained for final output; the richer
`IntersectionRecord` is internal to the pipeline.

### Worker and session

The pipeline runs within a `GPUWorker` session. It inherits device
lifecycle, pipeline caching, command-boundary cancellation, and
device-loss detection.

### Primitives

The existing one-workgroup primitives may be used as building blocks
within a single workgroup's local data. The multi-workgroup scan required
by stage 3 is a new primitive that supersedes the single-workgroup
`prefix_sum` for pipeline use.

### Trace infrastructure

Every clip operation is traceable via the existing stream-format v3
infrastructure. The pipeline emits trace records compatible with
`src/chitin/trace.py` for regression replay.

## Acceptance gate

The clipping pipeline ships under a formal gate, extending the f32 gate
protocol (`docs/f32-gate-protocol.md`).

### Evidence corpus

| Tier | Source | Purpose |
|------|--------|---------|
| **Regression** | Policy 0.1 corpus (114 clips) | Historical regression baseline |
| **Regression** | Policy 0.2 diagnostic corpus (14 clips) | Targeted boundary cases |
| **Calibration** | t_shape, curved_pipe, h_shape (~25K clips) | Tuning and development |
| **Holdout (G2)** | *Fresh manifest, frozen before tuning* | Gate holdout — see below |
| **Adversarial** | Generated boundary-directed cases | Stress testing |
| **Structural** | Large-coordinate, coplanar, multi-component, high-valence, nested-loop fixtures | Edge-case coverage |
| **Operational** | Capacity boundaries, cancellation, device loss | Resilience |

The existing t_shape, curved_pipe, and h_shape fixtures were holdout
data for the Policy 0.1/0.2 gates. They are **spent** — tuning against
them has already occurred. For the GPU clipping gate they serve as
**calibration** (develop and debug against) but are not holdout.

A fresh **G2 holdout manifest** must be frozen before any GPU clipping
tuning begins:

1. Select fixtures not previously used in any f32 gate evaluation.
2. Record the manifest (fixture names, clip counts, SHA-256) immutably.
3. Do not evaluate any GPU clipping code against G2 holdout fixtures
   until the gate evaluation.
4. Document the manifest in `docs/g2-holdout-manifest.md`.

### Gate criteria

All criteria must pass simultaneously:

1. **Zero classification mismatches.** GPU sign assignment matches CPU
   Policy 0.3 reference on identical f32 inputs, across the full corpus
   including G2 holdout.

2. **Topology parity.** For every clip in the corpus, the GPU output's
   topology matches the corrected CPU reference (#120). Parity is
   established through **canonical identities**, not raw index
   comparison:
   - Source vertices are identified by their global index in the input
     mesh.
   - Intersection vertices are identified by their `EdgeKey`.
   - Each output face is a triple of canonical vertex identities.
   - The face **sets** (unordered, but preserving multiplicity and
     per-face winding orientation relative to the plane normal) must
     match between GPU and CPU for both halves.
   - Source-face ancestry must match: each output face traces to the
     same source triangle.

3. **Zero non-finite output coordinates.** No NaN or Inf in any output
   vertex position.

4. **Zero invalid indices.** No out-of-range face index in any output.

5. **Every boundary loop closed.** Zero dangling or branching vertices
   in any boundary graph.

6. **Every cap consistently wound.** Verified by dot product of cap
   triangle normal with clipping plane normal.

7. **No classify→weld readback.** Verified by code audit — no
   `read_buffer` between classify and weld stages.

8. **Deterministic output.** Ten repeated runs of the full corpus on the
   same adapter produce bitwise-identical output.

### Procedure

1. Freeze G2 holdout manifest.
2. Fix CPU topology oracle (#120) — welded intersections, closed loops,
   on-plane boundary edges.
3. Run corrected CPU reference on the full corpus (regression +
   calibration + adversarial + structural). Record outputs as golden,
   keyed by canonical vertex identities.
4. Implement GPU pipeline.
5. Run GPU pipeline on the same corpus with identical f32 inputs.
6. Run GPU pipeline on G2 holdout (first and only evaluation).
7. Compare per gate criteria.
8. Record raw results immutably.
9. Issue PASS or FAIL verdict.
10. If PASS: close #92, #93, #103, #120 with sha link.
11. If FAIL: document failure class, file remediation issue.
