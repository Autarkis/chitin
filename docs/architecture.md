# Chitin Architecture

Chitin is a compiler, not a library.

It takes messy, heterogeneous 3D input -- gaussian splats, photogrammetry scans, rigged characters, raw meshes -- and produces portable, validated physics colliders. The interesting problem is not convex decomposition (CoACD does that). The interesting problem is the boundary: how geometry from dozens of authoring tools becomes trusted runtime physics data that engines can load without thinking.

## The contract

`.phys` is the contract. A binary sidecar that is:

- **Explicit**: every hull, bone assignment, and quantization parameter is inspectable. No opaque engine-specific blobs.
- **Versioned**: format version in the header, layout invariants documented, readers reject unknown versions.
- **Portable**: identical dequantization in Python, TypeScript, C#, and C++. Author once, ship everywhere.
- **Validated**: `chitin validate` checks structural integrity, offset consistency, index bounds, AABB sanity, and bind-pose block completeness before anything touches a physics engine.

The format is documented in [phys.md](phys.md).

## Components

```
                     ┌─────────────────────┐
                     │   3D input           │
                     │   PLY, OBJ, GLB,     │
                     │   splats, arrays      │
                     └─────────┬───────────┘
                               │
                     ┌─────────▼───────────┐
                     │   chitin compiler    │
                     │                      │
                     │   normalize          │
                     │   filter (opacity)   │
                     │   derive normals     │
                     │   octree partition   │
                     │   inflate (splats)   │
                     │   reconstruct        │
                     │   proximity filter   │
                     │   thin-shell (env)   │
                     │   dedup (cross-cell) │
                     │   decompose          │
                     │   segment (bones)    │
                     │   quantize           │
                     │   validate           │
                     └─────────┬───────────┘
                               │
                     ┌─────────▼───────────┐
                     │   .phys sidecar      │
                     │   (+ JSON, USD)      │
                     └─────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │  Web            │ │  Unity        │ │  Unreal      │
    │  Three.js       │ │  PhysReader   │ │  ChitinPhys  │
    │  Rapier         │ │  MeshCollider │ │  UBodySetup  │
    └────────────────┘ └──────────────┘ └──────────────┘
```

### Compiler (`chitin-core`)

The compiler kernel is a deterministic pipeline. Same input bytes + same config = same output bytes. This is what makes caching work.

Stages:
1. **Input normalization** -- load any supported format into a common mesh/point cloud representation
2. **Opacity filtering** -- for gaussian splats: discard points below threshold, keeping only physically present geometry
3. **Normal derivation** -- for 3DGS data with scale/rotation: derive oriented normals from the covariance ellipsoid (shortest axis = surface normal). Falls back to KD-tree estimation for plain point clouds
4. **Spatial partitioning** -- octree split for scenes exceeding 50K points. Ghost-zone padding (3x median gaussian scale) ensures boundary continuity across cells
5. **Splat inflation** -- optionally expand each gaussian center into disk samples along its two largest axes for better Poisson surface coverage
6. **Surface reconstruction** -- Poisson reconstruction via Open3D. Depth auto-selected per cell based on point count (4-7). Each cell runs in a subprocess so Open3D segfaults are isolated
7. **Surface filtering** -- configurable density quantile strips low-confidence Poisson vertices. Proximity filter removes closure surfaces far from input data. Thin-shell extrusion prevents volume-fill on environment scans
8. **Cross-cell deduplication** -- AABB IOU dedup (threshold 0.5) removes duplicate hulls at octree cell boundaries
9. **Convex decomposition** -- CoACD splits non-convex geometry into convex hulls
10. **Bone segmentation** -- for rigged assets: assign each vertex to its dominant bone, generate per-bone hulls in bone-local space
11. **Quantization** -- int16 per-axis quantization against per-hull AABBs (65536 levels, ~0.001% error for typical meshes)
12. **Validation** -- structural checks before writing. Reject oversized hulls, degenerate AABBs, index overflow

The compiler does not make runtime decisions. It produces an artifact. Consumers decide how to use it.

### Build service (`chitin-service`)

A local build server that wraps the compiler with:
- **Content-addressed caching**: `SHA-256(input_bytes + config_dict + compiler_version)`. Cache hit rate is high because iteration loops mostly change the asset, not the physics config.
- **Job tracking**: created, uploaded, running, complete, failed. Stored in SQLite.
- **Artifact storage**: `.phys`, `.json`, `.usda` written to local disk, downloadable by job ID.

Currently synchronous and single-process. Designed for local dev and CI, not production multi-tenant use. The job model exists so the transition to async workers is a schema change, not an architecture change.

### Consumers

Consumers are thin. They read `.phys`, dequantize vertices, and hand geometry to the engine's physics API. Each consumer is ~100-150 lines. The complexity lives in the compiler, not the readers.

| Consumer | Language | Physics API | Matrix convention |
|----------|----------|-------------|-------------------|
| `@autarkis/chitin-web` | TypeScript | Rapier WASM | Column-vector (implicit via `fromArray`) |
| `@autarkis/chitin-lite` | TypeScript | CoACD WASM | N/A (produces `.phys`, does not consume) |
| `com.chitin.physics` | C# | Unity MeshCollider | Column-vector (explicit transpose) |
| ChitinImporter | C++ | Unreal UBodySetup | Row-vector (native match) |
| `chitin.phys` | Python | Direct numpy | Row-vector (native match) |

### Validation and diagnostics

Validation is a first-class output, not a debugging afterthought.

`chitin validate` checks:
- Magic bytes, version, header size
- Offset consistency (hull table, vertex data, index data are contiguous and correctly sized)
- Index bounds (no index exceeds its hull's vertex count)
- AABB sanity (min <= max per component)
- Bind-pose block integrity (bone count, transform data, name lengths)
- Near-singular bind transforms (determinant check)

Golden fixtures with known transforms are tested in Python and TypeScript on every change. The fixtures include deliberately unaligned bind-pose blocks to catch typed-array alignment bugs in browser runtimes.

## Design decisions

**Row-vector convention for bind transforms.** GLTF stores column-major; numpy reshapes to row-major. Translation lives in row 3 of the 4x4. `world = local @ bind_transform`. Each consumer handles the conversion to its engine's convention. The spec documents one convention; consumers adapt.

**int16 quantization per hull, not global.** Each hull gets its own AABB. This means a scene with both a 100-meter terrain and a 1-centimeter button gets good precision on both, without needing float32 vertices everywhere.

**No padding in the binary format.** The bind-pose block starts immediately after index data, which may not be 4-byte aligned. Readers must use byte-level access (DataView, BinaryReader), not typed-array views. This keeps the format simple and avoids wasting bytes on alignment that only matters for one parse pattern.

**Compiler version in the cache key.** A CoACD update, quantization tweak, or reconstruction parameter change invalidates the cache automatically. No manual cache busting.

**Auto-selected Poisson depth, never 8.** Open3D Poisson segfaults non-deterministically on dense spatial clusters at depth 8. The formula `clamp(floor(log2(n_points) / 3), 4, 7)` gives ~1-10 points per leaf cell at each depth level. Per-cell auto-selection means small cells don't get over-resolved and dense cells get appropriate resolution.

**Subprocess isolation for Poisson.** Open3D uses internal threads that deadlock under `fork()` on macOS. The subprocess approach (`.npz` file exchange) avoids this entirely and also contains segfaults: if one cell crashes, the parent gets a nonzero return code, skips the cell, and continues. The I/O cost is negligible compared to CoACD.

**Environment scans are opt-in, not auto-detected.** Poisson creates watertight meshes, which fills concave environments with solid geometry. The proximity filter and thin-shell extrusion fix this, but require explicit configuration. Auto-detecting object vs. environment scans from normal orientation is feasible but noisy for mixed scenes (object on a table in a room). Better to let the user declare intent than guess wrong.

**CoACD compiles to WASM without OpenVDB.** CoACD uses OpenVDB only for manifold repair as a preprocessing step. Building with `-DWITH_3RD_PARTY_LIBS=OFF` drops OpenVDB, Boost, TBB, and spdlog -- leaving pure C++ with header-only deps (CDT, nanoflann, vendored Bullet quickhull). The result is a 558KB `.wasm` module that runs the full MCTS decomposition algorithm in the browser. The trade-off is that input meshes must be manifold; without OpenVDB there is no automatic repair.

**Two-tier dependency model.** The Python compiler (`pip install chitin`) handles the heavy path: point clouds, splats, Poisson reconstruction, environment scan filtering. The browser module (`@autarkis/chitin-lite`) handles the light path: mesh → convex hulls → `.phys`. Both produce the same `.phys` format. Open3D stays on the server; CoACD runs everywhere.

## What chitin is not

- Not a physics engine. It produces collider geometry; something else simulates it.
- Not a mesh optimizer. It does not simplify or retopologize visual meshes. Collision LOD tiers are fresh decompositions at different concavity thresholds, not decimations.
- Not a cloud service (yet). The build service is local-first. Cloud is a deployment decision, not an architecture one.
- Not a format converter. `.phys` is the primary output. JSON and USD are companions for ecosystems that need them.
