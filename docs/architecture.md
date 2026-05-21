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

## Positioning

Chitin is free MIT-licensed infrastructure, but the strategic distinction is not "open source vs. closed." Adjacent splat tooling can be open source too. The important distinction is the contract Chitin produces.

| Viewer collision pipeline | Chitin |
|---------------------------|--------|
| Starts from a splat viewer/editor workflow | Starts from an asset build pipeline |
| Voxelizes occupancy for walk mode, raycasts, or broad-phase queries | Reconstructs surfaces and decomposes them into convex hulls |
| Often depends on a seed point, fill/carve presets, and a specific runtime | Uses explicit compiler config, build diagnostics, and portable readers |
| Emits viewer-oriented voxel data or a generated collision mesh | Emits `.phys`: versioned, validated convex hull data with optional LOD and rig blocks |
| Usually lives inside one viewer's collision path | Loads beside any visual asset as a sidecar |
| Optimized for "can I walk around this splat now?" | Optimized for "can I ship this collision asset everywhere?" |

The approaches can coexist. A splat viewer can use voxel collision for immediate navigation while Chitin produces the final physics artifact for Unity, Unreal, Rapier, robotics simulators, or custom engines. In the browser, `@autarkis/chitin-web` exposes `parsePhys`, `createColliders`, `addToWorld`, and Three.js debug meshes so a viewer can attach Chitin collision with a few lines of runtime code.

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
                     │   flatness detect    │
                     │   decompose          │
                     │   seam repair        │
                     │   dedup (cross-cell) │
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
4. **Spatial partitioning** -- octree split for scenes exceeding 50K points. Per-cell ghost-zone padding (3x 95th-percentile splat radius) ensures boundary continuity without over-inflating cells that contain small splats
5. **Splat inflation** -- optionally expand each gaussian center into disk samples along its two largest axes for better Poisson surface coverage
6. **Surface reconstruction** -- Poisson reconstruction via Open3D. Depth auto-selected per cell based on point count (4-7). Each cell runs in a subprocess so Open3D segfaults are isolated
7. **Surface filtering** -- configurable density quantile strips low-confidence Poisson vertices. Proximity filter removes closure surfaces far from input data. Thin-shell extrusion prevents volume-fill on environment scans
8. **Flatness detection** -- PCA eigenvalue ratio classifies near-flat octree cells. Flat cells produce a single oriented box instead of running CoACD, reducing hull count and build time
9. **Convex decomposition** -- CoACD splits non-convex geometry into convex hulls
10. **Seam repair** -- detects height discontinuities at octree cell boundaries via capsule sweep, union-finds cells sharing seam snags, merges their bounds, and re-extracts for continuous coverage
11. **Cross-cell deduplication** -- AABB IOU dedup (threshold 0.5) removes duplicate hulls at octree cell boundaries
12. **Bone segmentation** -- for rigged assets: assign each vertex to its dominant bone, generate per-bone hulls in bone-local space
13. **Quantization** -- int16 per-axis quantization against per-hull AABBs (65536 levels, ~0.001% error for typical meshes)
14. **Validation** -- structural checks before writing. Reject oversized hulls, degenerate AABBs, index overflow

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

**Environment scans are auto-detected with explicit opt-out.** Poisson creates watertight meshes, which fills concave environments with solid geometry. The proximity filter and thin-shell extrusion fix this. Chitin auto-detects hollow-shell point distributions (fewer than 5% of points in the inner 50% of the AABB) and enables both. Use `auto_environment=False` or `--no-auto-environment` to disable if the heuristic misfires on a particular scene.

**CoACD compiles to WASM without OpenVDB.** CoACD uses OpenVDB only for manifold repair as a preprocessing step. Building with `-DWITH_3RD_PARTY_LIBS=OFF` drops OpenVDB, Boost, TBB, and spdlog -- leaving pure C++ with header-only deps (CDT, nanoflann, vendored Bullet quickhull). The result is a 558KB `.wasm` module that runs the full MCTS decomposition algorithm in the browser. The trade-off is that input meshes must be manifold; without OpenVDB there is no automatic repair.

**Two-tier dependency model.** The Python compiler (`pip install chitin`) handles the heavy path: point clouds, splats, Poisson reconstruction, environment scan filtering. The browser module (`@autarkis/chitin-lite`) handles the light path: mesh → convex hulls → `.phys`. Both produce the same `.phys` format. Open3D stays on the server; CoACD runs everywhere.

## What chitin is not

- Not a physics engine. It produces collider geometry; something else simulates it.
- Not a splat viewer or walk-mode runtime. It compiles collider artifacts; viewer UX, seed picking, and camera navigation belong upstream or downstream.
- Not a mesh optimizer. It does not simplify or retopologize visual meshes. Collision LOD tiers are fresh decompositions at different concavity thresholds, not decimations.
- Not a cloud service (yet). The build service is local-first. Cloud is a deployment decision, not an architecture one.
- Not a format converter. `.phys` is the primary output. JSON and USD are companions for ecosystems that need them.
