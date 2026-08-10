# Spark demo sidecars

These are build-time Chitin outputs. The demo's production-default path loads
them as ordinary runtime assets and does not reconstruct or decompose their
source geometry. The **Chitin setup** tab also exposes explicit, opt-in browser
regeneration for authoring and inspection; a generated candidate remains staged
until it is applied or downloaded.

## `robot-head.phys`

- Visual source: `https://sparkjs.dev/assets/splats/robot-head.spz`
- Chitin input: decoded centers, scales, rotations, and opacity
- Reconstruction: resolution 72, support multiplier 1.75, minimum opacity 0.2,
  shell thickness 1.15 voxels
- Decomposition: threshold 0.025, maximum 96 hulls, maximum 64 vertices per
  hull
- Output: `.phys` v3, 57 hulls, 1,921 vertices, 3,614 triangles

Regenerate this sidecar only when the paired visual source or compiler settings
change. Gaussian data is a generation input and runtime visual; after generation
the stored `.phys` is authoritative for geometry-dependent behavior.

See [`docs/gaussian-splats.md`](../../../../docs/gaussian-splats.md) for the full
asset lifecycle and lighting architecture.

## `l-prism.phys`

Generated from the demo's procedural L-prism mesh. Output: `.phys` v3, 2 hulls.
