# Chitin Architecture

Chitin is a compiler, not a library.

It takes messy, heterogeneous 3D input -- gaussian splats, photogrammetry scans, rigged characters, raw meshes -- and produces portable, validated physics colliders. The interesting problem is not convex decomposition (CoACD does that). The interesting problem is the boundary: how geometry from dozens of authoring tools becomes trusted runtime physics data that engines can load without thinking.

## The contract

`.phys` is the contract. A binary sidecar that is:

- **Explicit**: every hull, bone assignment, and quantization parameter is inspectable. No opaque engine-specific blobs.
- **Versioned**: format version in the header, layout invariants documented, and the Python/TypeScript readers reject unknown versions and flags.
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
                     │   adapters/          │
                     │   ply, gltf, usd,    │
                     │   mesh (OBJ/STL/OFF) │
                     └─────────┬───────────┘
                               │
                  analyze() + resolve()
                               │
                     ┌─────────▼───────────┐
                     │   stages/            │
                     │   normalize          │
                     │   reconstruct        │
                     │   filter             │
                     │   flatness           │
                     │   decompose          │
                     │   repair             │
                     │   segment (rigged)   │
                     │   splat (covariance) │
                     └─────────┬───────────┘
                               │
                     ┌─────────▼───────────┐
                     │   verify/            │
                     │   probe, sweep, seam │
                     └─────────┬───────────┘
                               │
                     ┌─────────▼───────────┐
                     │   exporters/         │
                     │   .phys, JSON, USD   │
                     │   bundle (artifact)  │
                     └─────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │  Web            │ │  Unity        │ │  Unreal      │
    │  Three.js       │ │  PhysReader   │ │  ChitinPhys  │
    │  Rapier         │ │  MeshCollider │ │  PhysAsset   │
    └────────────────┘ └──────────────┘ └──────────────┘
```

### Compiler (`chitin-core`)

The compiler kernel is a deterministic pipeline. Same input bytes + same config = same output bytes. This is what makes caching work.

The compiler is organized into four module groups:

**`adapters/`** -- input loading. Each adapter (PLY, GLTF/FBX, USD, OBJ/STL/OFF) returns an `AdapterResult` with positions, faces, normals, covariance data, opacity, and optional skin data. The pipeline never sees the source format.

**`stages/`** -- the processing pipeline. Each stage takes typed input + a frozen `ResolvedConfig`, returns typed output, and appends to the `BuildPlan`:

1. **normalize** -- scale-normalization pass: uniformly rescales the input to a target height or footprint (`--target-height`/`--target-footprint`) before decomposition. Skipped for skinned assets
2. **reconstruct** -- Poisson reconstruction via Open3D. Depth auto-selected per cell based on point count (4-7). Each cell runs in a subprocess so Open3D segfaults are isolated
3. **filter** -- proximity filter removes closure surfaces far from input data. Density quantile strips low-confidence Poisson vertices. Thin-shell extrusion prevents volume-fill on environment scans
4. **flatness** -- PCA eigenvalue ratio classifies near-flat octree cells. Flat cells produce a single oriented box instead of running CoACD
5. **decompose** -- CoACD convex decomposition, LOD tier generation, walkable hull extraction, cross-cell AABB IOU deduplication
6. **repair** -- detects height discontinuities at octree cell boundaries via a downward-ray ground sweep, union-finds cells sharing seam snags, merges their bounds, and re-extracts
7. **segment** -- bone segmentation for rigged assets: assign each vertex to its dominant bone
8. **splat** -- covariance normal derivation, anisotropic inflation, octree spatial partitioning, per-cell reconstruction orchestration

**`verify/`** -- post-build quality checks. `raycast.py` provides shared Moller-Trumbore ray-triangle intersection (AABB pre-filtered). `probe.py` fires downward ray grids for coverage metrics. `sweep.py` does ground-reachability analysis -- a step-height-gated flood fill over ground cells; it reports reachable fraction and does not yet evaluate vertical or lateral capsule clearance (`--capsule-height` is currently unused). `seam.py` detects height discontinuities at cell boundaries (used by `stages/repair.py` during the build).

**`exporters/`** -- output serialization. `.phys` binary packing, JSON debug companion, USD Physics output, and artifact bundle (build-plan.json + analysis.json + resolved-config.json).

**`analyze.py` + `resolve.py`** -- the spine. `analyze_arrays()` produces an `InputAnalysis` (facts about the input: format, opacity, covariance, density, skinning). `resolve_config()` turns user `Config` + `InputAnalysis` into a frozen `ResolvedConfig` with a `decisions` dict explaining every auto-override. No downstream code mutates config after this point.

**`core.py`** -- orchestration only (~369 lines). `extract()` dispatches: load adapter -> analyze -> resolve -> pipeline. `extract_from_arrays()`, `extract_from_mesh()`, `extract_from_rigged_mesh()` are the public entry points for direct API use.

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
| ChitinImporter | C++ | Unreal asset import (`UChitinPhysAsset`; no `UBodySetup` yet) | Row-vector (native match) |
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

**Single-threaded Poisson inside workers.** `create_from_point_cloud_poisson` runs with `n_threads=1`. Parallelism comes from the cell-level process pool; letting each worker spawn its own all-core OpenMP pool oversubscribed the machine, segfaulted sporadically under contention (SIGSEGV in `libomp`), and made float accumulation order -- and thus the output mesh -- nondeterministic. Single-threading per cell is what makes "same input bytes + same config = same output bytes" actually hold.

**Environment scans are auto-detected with explicit opt-out.** Poisson creates watertight meshes, which fills concave environments with solid geometry. The proximity filter and thin-shell extrusion fix this. Chitin auto-detects hollow-shell point distributions (fewer than 5% of points in the inner 50% of the AABB) and enables both. Use `auto_environment=False` or `--no-auto-environment` to disable if the heuristic misfires on a particular scene.

**CoACD compiles to WASM without OpenVDB.** CoACD uses OpenVDB only for manifold repair as a preprocessing step. Building with `-DWITH_3RD_PARTY_LIBS=OFF` drops OpenVDB, Boost, TBB, and spdlog -- leaving pure C++ with header-only deps (CDT, nanoflann, vendored Bullet quickhull). The result is a 558KB `.wasm` module that runs the full MCTS decomposition algorithm in the browser. The trade-off is that input meshes must be manifold; without OpenVDB there is no automatic repair.

**Two-tier dependency model.** The Python compiler (`pip install chitin`) handles the heavy path: point clouds, splats, Poisson reconstruction, environment scan filtering. The browser module (`@autarkis/chitin-lite`) handles the light path: mesh → convex hulls → `.phys`. Both produce the same `.phys` format. Open3D stays on the server; CoACD runs everywhere.

## What chitin is not

- Not a physics engine. It produces collider geometry; something else simulates it.
- Not a splat viewer or walk-mode runtime. It compiles collider artifacts; viewer UX, seed picking, and camera navigation belong upstream or downstream.
- Not a mesh optimizer. It does not simplify or retopologize visual meshes. Collision LOD tiers are fresh decompositions at different concavity thresholds, not decimations.
- Not a format converter. `.phys` is the primary output. JSON and USD are companions for ecosystems that need them.

## Maturity

The Python compiler, `.phys` format, and Python/TypeScript readers are tested on every change with golden fixtures. The Python/TypeScript readers reject unknown versions, unknown flags, and trailing data after known blocks. Config validation fails fast on invalid inputs before reaching CoACD or Open3D.

The build service (`chitin-service`) is alpha: synchronous, single-process, exposes a subset of compiler knobs, and does not yet produce the full artifact bundle. The Unity reader (`com.chitin.physics`) and Unreal importer (`ChitinImporter`) parse the happy path but do not enforce the full set of structural invariants that the Python validator checks. Cross-runtime conformance tests are planned but not yet in place.

Rigged GLTF support is experimental (single-primitive, packed accessors). FBX ingest routes through `chitin convert` (Blender headless).
