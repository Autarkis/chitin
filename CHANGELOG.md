# Changelog

All notable changes to Chitin are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The `.phys` binary
format is versioned independently and noted where it changes.

## [0.1.2] - 2026-07-19

- Add `keywords`, `homepage`, and `bugs` to both npm packages so they surface in
  registry search (npm-only; the Python package already carried these).

## [0.1.1] - 2026-07-19

- Add a README for `@autarkis/chitin-web`.
- npm packages publish via GitHub OIDC trusted publishers instead of a token.

## [0.1.0] - 2026-07-19

First public release. `.phys` format version 3.

### Compiler (Python)

- Extract convex-hull colliders from meshes, point clouds, gaussian splats, and
  USD, emitting the `.phys` binary sidecar (plus JSON and USD Physics outputs).
- Gaussian-splat pipeline: covariance-derived normals, opacity filtering, octree
  spatial decomposition, Poisson reconstruction, CoACD convex decomposition,
  flatness detection, seam repair, and cross-cell reconciliation.
- Multi-LOD output: tiered decompositions at multiple concavity thresholds in one
  `.phys`.
- Real-world scale normalization (`--target-height` / `--target-footprint`),
  applied consistently across all extract entry points.
- Experimental rigged-GLTF support: per-bone hulls in bone-local space.
- FBX inputs auto-convert to GLB via headless Blender.
- Robustness: CoACD runs in a bounded subprocess with a bounding-box fallback so
  non-watertight input can never hang the pipeline; the spatial pool uses `spawn`.
- Local build service (`chitin-server`) with content-addressed caching, keyed on
  input kind + dependency versions.

### Format & readers

- `.phys` v3: quantized convex hulls, per-bone bind transforms, and LOD tiers.
- Readers for Python, TypeScript (`@autarkis/chitin-web`), Unity, and Unreal,
  all with nearest-concavity LOD tier selection.
- Hardened validation across Python and the web parser: rejects unknown
  versions/flags, trailing bytes, out-of-range or non-contiguous hull/LOD offsets,
  out-of-range bone indices, and non-finite AABBs and bind transforms.
- Cross-runtime conformance suite: one golden corpus verified by the Python and
  TypeScript readers.

### Browser

- `@autarkis/chitin-lite`: CoACD compiled to WebAssembly with a TypeScript API
  producing the same v3 `.phys` as the Python compiler. Typed errors and writer
  input validation.
- `@autarkis/chitin-web` uses subpath exports so the format reader is
  dependency-free; Rapier and Three.js bindings live under `/rapier` and `/three`.

### Engines

- Unity `com.chitin.physics` (drag-drop ScriptedImporter), Unreal ChitinImporter
  (asset import), and a Three.js + Rapier web bridge.

[0.1.2]: https://github.com/Autarkis/chitin/releases/tag/web-v0.1.2
[0.1.1]: https://github.com/Autarkis/chitin/releases/tag/python-v0.1.1
[0.1.0]: https://github.com/Autarkis/chitin/releases/tag/python-v0.1.0
