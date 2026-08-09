# @autarkis/chitin-web

Load chitin `.phys` collision sidecars in the browser. The dependency-free root
prepares plain typed-array geometry; optional subpaths attach it to Rapier,
Three.js, or a Spark gaussian-splat visual.

## Setup

```bash
npm install @autarkis/chitin-web
```

The package uses subpath exports. The root is dependency-free. Runtime bindings
pull their engines in as optional peer dependencies:

```bash
npm install @dimforge/rapier3d-compat   # for /rapier
npm install three                        # for /three
npm install three @sparkjsdev/spark     # for /spark
```

## Usage

### Parse a .phys file

```typescript
import { parsePhys } from "@autarkis/chitin-web";

const buffer = await fetch("/assets/scene.phys").then((r) => r.arrayBuffer());
const phys = parsePhys(buffer); // throws on malformed input

for (const hull of phys.hulls) {
  console.log(hull.vertices, hull.indices); // Float32Array, Uint16Array
}
```

### Prepare geometry without an engine

`materializePhysHulls()` is the shared boundary used by every runtime adapter.
It selects collision LOD, optionally bakes bone-local hulls through their bind
poses, and returns only typed arrays and plain metadata:

```typescript
import {
  combinePhysGeometry,
  materializePhysHulls,
} from "@autarkis/chitin-web";

const prepared = materializePhysHulls(phys, {
  lodConcavity: 0.3,
  pose: "bind", // complete asset-local geometry
});
const combined = combinePhysGeometry(prepared);
```

Use `pose: "stored"` when an animated consumer needs the original bone-local
hulls. `transformHullVertices()`, `calculateVertexBounds()`, and
`resolvePhysTransform()` are also engine-neutral.

### Add colliders to a Rapier world

```typescript
import RAPIER from "@dimforge/rapier3d-compat";
import { parsePhys } from "@autarkis/chitin-web";
import { addToWorld } from "@autarkis/chitin-web/rapier";

const phys = parsePhys(await fetch("/scene.phys").then((r) => r.arrayBuffer()));
addToWorld(RAPIER, world, phys);
```

For a transformed visual, pass the same TRS to Rapier. Scale is baked into the
collider vertices; translation and rotation are placed on the rigid body:

```typescript
addToWorld(RAPIER, world, phys, undefined, {
  transform: {
    translation: [3, 0, -2],
    rotation: [0, 0, 0, 1],
    scale: [1.5, 1.5, 1.5],
  },
});
```

Use `createColliders(rapier, phys)` if you want the `ColliderDesc`s without
attaching them to a body.

### Pick a LOD tier

For a `.phys` with LOD tiers, select the set nearest a target concavity:

```typescript
import { selectLodHulls } from "@autarkis/chitin-web";
import { createColliders } from "@autarkis/chitin-web/rapier";

const coarseHulls = selectLodHulls(phys, 0.3); // hull data only
const { colliders } = createColliders(rapier, phys, { lodConcavity: 0.3 });
```

Readers with no target pick default to LOD 0 (highest detail).

### Debug meshes

```typescript
import { createDebugMeshes } from "@autarkis/chitin-web/three";

scene.add(createDebugMeshes(phys)); // wireframe hull visualization
```

### Hull shadow proxies

`/three` can turn the same prepared hulls into color-invisible shadow casters:

```typescript
import { createShadowProxy } from "@autarkis/chitin-web/three";

scene.add(createShadowProxy(phys, {
  lodConcavity: 0.3,
  mode: "merged-hulls", // one draw object; no boolean union
}));
```

The proxy casts through Three.js's shadow pass onto ordinary Three receivers.
Whether a gaussian-splat renderer can receive that shadow is a capability of
that renderer's shader pipeline, not part of the `.phys` contract.

### Spark gaussian splats

The Spark adapter pairs one `SplatMesh` with its `.phys` sidecar and optional
shadow proxy under a shared transform root. The application still owns its one
scene-level `SparkRenderer`, and Rapier remains an independent opt-in:

```typescript
import RAPIER from "@dimforge/rapier3d-compat";
import { SparkRenderer } from "@sparkjsdev/spark";
import { addToWorld } from "@autarkis/chitin-web/rapier";
import { loadSparkAsset } from "@autarkis/chitin-web/spark";
import { physTransformFromObject3D } from "@autarkis/chitin-web/three";

scene.add(new SparkRenderer({ renderer }));

const asset = await loadSparkAsset({
  visual: "/chair.spz",
  phys: "/chair.phys",
  shadows: { mode: "collision-lod", lodConcavity: 0.3 },
});
scene.add(asset.root);

const body = addToWorld(RAPIER, world, asset.phys, undefined, {
  transform: physTransformFromObject3D(asset.root),
  lodConcavity: 0.3,
});
```

`asset.bounds` exposes the Spark and bind-posed physics bounds for alignment
diagnostics. `asset.dispose()` releases adapter-owned visual and shadow
resources; the application removes any Rapier body it created separately. For
a supplied `SplatMesh`, its local TRS is adopted by `asset.root` so every layer
stays aligned, then restored if the caller-owned visual is later disposed from
the adapter.

## Other engines

If you use a physics engine other than Rapier, consume
`materializePhysHulls()` directly and pass each hull's typed arrays to that
engine. No Three.js, Spark, or Rapier import is needed.

Colliders are generated by the [chitin](https://github.com/Autarkis/chitin)
compiler (or `@autarkis/chitin-lite` in the browser); this package only reads
their `.phys` output.
