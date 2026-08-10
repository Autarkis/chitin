# Gaussian Splats and `.phys` Sidecars

Chitin treats a Gaussian splat asset as appearance paired with a portable
physics artifact. The visual may be SPZ, PLY, or another renderer-owned format;
Chitin's runtime contract is the sibling `.phys` file.

```text
authoring
Gaussian centers + covariance ── Chitin compiler ──> asset.phys
          │                                             │
          └──────────────> asset.spz                     │

production runtime
asset.spz ──> appearance renderer                        │
asset.phys ──> hull geometry ──┬─> physics / raycasts ───┤
                              ├─> shadow casting         │
                              ├─> shadow reception       │
                              ├─> self-shadowing         │
                              └─> hull-depth AO          │
```

Once the sidecar has been generated, Gaussian data is only required for the
visible render. It is not re-read to reconstruct collision or lighting geometry
on every page load.

## Layering

The integration keeps its foundation independent of Three.js and Spark:

| Layer | Package | Responsibility |
|---|---|---|
| Compilation | `@autarkis/chitin-lite` | Reconstruct a sealed surface from decoded oriented Gaussians, decompose it, and write `.phys` |
| Artifact geometry | `@autarkis/chitin-web` | Parse `.phys`, select LOD, bake bind poses, transform vertices, compute bounds, and combine hull buffers using typed arrays |
| Physics | `@autarkis/chitin-web/rapier` | Convert the shared hull geometry into Rapier colliders |
| Three.js | `@autarkis/chitin-web/three` | Create hull geometry, hidden shadow proxies, and the deferred hull-lighting pass |
| Spark | `@autarkis/chitin-web/spark` | Pair a Spark visual with `.phys` and optionally bridge a directional shadow into Spark's modifier pipeline |

`/spark` does not generate colliders and `/rapier` knows nothing about Spark.
Both consume the same engine-neutral `.phys` geometry.

## Generate a sidecar

### Offline from PLY

The Python compiler remains the production path for raw point clouds, PLY
Gaussian splats, large scans, and inputs that need repair:

```bash
pip install chitin[splat]
chitin extract robot-head.ply -o robot-head.phys
chitin inspect robot-head.phys
chitin validate robot-head.phys
```

Store the result beside the renderer asset:

```text
assets/
  robot-head.spz
  robot-head.phys
```

Regenerate the sidecar when the visual source, its coordinate frame, or the
compiler settings change.

### In the browser from decoded Gaussians

An authoring tool that already has oriented Gaussian arrays can compile them
without first writing a GLB:

```typescript
import {
  ChitinCompiler,
  type GaussianFieldInput,
} from "@autarkis/chitin-lite";

const field: GaussianFieldInput = {
  centers,    // packed xyz
  scales,     // packed positive xyz ellipsoid radii
  quaternions, // packed xyzw
  opacities,  // optional, one value in [0, 1] per Gaussian
};

const compiler = new ChitinCompiler({
  wasm: {
    js: "/chitin/coacd.mjs",
    wasm: "/chitin/coacd.wasm",
    version: "0.2.0",
  },
  workerUrl: new URL("/chitin/worker.js", location.href),
  maxWorkers: 2,
});

const result = await compiler.compileGaussianField(field, {
  reconstruction: {
    resolution: 72,
    supportMultiplier: 1.75,
    minOpacity: 0.2,
    shellThickness: 1.15,
  },
  decompose: {
    threshold: 0.025,
    maxConvexHull: 96,
    maxChVertex: 64,
  },
  componentPolicy: {
    maxHulls: 96,
    detailedComponentMinThreshold: 0,
  },
  onProgress: ({ stage, completed, total }) =>
    console.log(stage, completed, total),
});

const blob = new Blob([result.phys], {
  type: "application/octet-stream",
});
compiler.terminate();
```

The reconstruction rasterizes oriented ellipsoid support, removes tiny voxel
islands, seals ambiguous voxel topology, fills enclosed volume, and passes the
closed boundary mesh through Chitin's normal validation and CoACD pipeline.
`resolution` controls reconstruction sampling; `threshold` controls convex
decomposition detail. They are separate fidelity controls.

Browser compilation uses a module Worker and WebAssembly. Serve the authoring
tool over HTTP or HTTPS; a page opened through `file://` cannot load the Worker.
Browser generation is useful for setup and inspection, but it is not a reason
to regenerate an unchanged production sidecar on every load.

## Load the production asset

`loadSparkAsset()` keeps the Spark visual and Chitin layers under one transform
root. A scene owns one `SparkRenderer`; the asset loader does not create one:

```typescript
import { SparkRenderer } from "@sparkjsdev/spark";
import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import { addToWorld } from "@autarkis/chitin-web/rapier";
import { loadSparkAsset } from "@autarkis/chitin-web/spark";
import { physTransformFromObject3D } from "@autarkis/chitin-web/three";

scene.add(new SparkRenderer({ renderer }));

const robot = await loadSparkAsset({
  visual: "/assets/robot-head.spz",
  phys: "/assets/robot-head.phys",
  shadows: { mode: "merged-hulls" },
});
scene.add(robot.root);

addToWorld(RAPIER, world, robot.phys, undefined, {
  transform: physTransformFromObject3D(robot.root),
});
```

`merged-hulls` combines buffers into one draw object. It does not union,
simplify, or lower the fidelity of the selected hulls. Omit `lodConcavity` to
use the highest-detail tier.

The visual and `.phys` must share a coordinate frame. `robot.bounds` exposes
both visual and bind-posed physics bounds for diagnostics. If an imported
sidecar needs correction, stage translation, rotation, and uniform scale for
preview, then apply the same transform to the hidden lighting proxy and physics
body together.

## Hull-derived lighting

There are two integration paths. Both can use Chitin hulls as the geometry
authority; they differ in where shadow visibility is composited.

### Deferred hull lighting

The Spark demo uses `createDeferredHullLightingPass()`:

1. Render normal scene color, including the Spark appearance.
2. Render the Chitin receiver hulls into a hidden camera-space depth/normal
   buffer.
3. Refresh directional shadow maps from hidden Chitin caster proxies.
4. Reconstruct the visible hull surface position from depth.
5. Compare that hull position with the light's shadow map.
6. Composite the result over the finished appearance image.

The splats provide the color being darkened; they do not provide collision,
receiver position, receiver normal, or occluder geometry.

```typescript
import {
  createDeferredHullLightingPass,
} from "@autarkis/chitin-web/three";

robot.shadowProxy!.traverse((object) => object.layers.set(1));

keyLight.castShadow = true;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;

const lighting = createDeferredHullLightingPass({
  renderer,
  scene,
  camera,
  hullLayer: 1,
  lights: [
    {
      light: keyLight,
      strength: 0.75,
      bias: -0.0002,
      normalBias: 0.025,
    },
  ],
});

renderer.setAnimationLoop(() => {
  lighting.render();
});
```

The pass compares hull normal with light direction when rejecting a far-side
receiver. Camera angle does not control shadow presence, so orbiting to a
grazing view does not fade an otherwise valid shadow. A genuinely light-grazing
surface still transitions out as its direct-light contribution approaches zero.

Source scene depth prevents the composite from shading ordinary scene geometry
that is in front of the hidden hull. `normalBias` is in world units: too little
causes acne, while too much can detach or erase thin shadows. Shadow-map extent
and resolution should be fitted to the asset before increasing bias.

The deferred implementation currently lives in `/three` and uses Three.js
WebGL render targets. The reusable architectural boundary is the engine-neutral
hull geometry from the package root; another renderer can build an equivalent
depth/normal and shadow-visibility pass without depending on Spark.

### Direct Spark modifier

`attachSparkDirectionalShadow()` injects shadow visibility into Spark's world
modifier pipeline. Bind each Gaussian to the closest `.phys` surface first so
receiver positions and normals still come from Chitin:

```typescript
import {
  attachSparkDirectionalShadow,
  bindSparkToPhysSurface,
} from "@autarkis/chitin-web/spark";

const receiver = attachSparkDirectionalShadow({
  visual: robot.visual,
  renderer,
  light: keyLight,
  surface: bindSparkToPhysSurface(robot.visual, robot.phys),
  strength: 0.75,
  normalBias: 0.025,
});
```

This path evaluates the result per visible Gaussian and requires Spark 2.1 plus
Three's `PCFShadowMap`. It is useful when an application wants the lighting
inside the Spark pipeline. The deferred path is preferable when several
appearance renderers should share one final hull-derived composite.

## Self-shadowing and AO

Self-shadowing uses another directional light whose shadow camera sees only the
hull layer. Copy the key light's position and target, and keep the configured
strength equal when self and external shadows should have the same opacity:

```typescript
selfLight.layers.set(1);
selfLight.shadow.camera.layers.set(1);
selfLight.position.copy(keyLight.position);
selfLight.target.position.copy(keyLight.target.position);
selfLight.castShadow = true;
```

The Spark demo approximates hull-depth ambient occlusion with two low-strength
hull-only directional depth maps. This is an inspection control, not baked AO
and not a replacement for a renderer's general scene AO. Both self-shadowing
and this AO approximation use `.phys` hull depth; neither reconstructs geometry
from the SPZ at runtime.

## Spark demo

The demo pairs Spark's robot-head example with a precompiled
`robot-head.phys`. Its **Chitin setup** tab provides:

- stored or replacement `.phys` loading;
- hull fill/edge preview and non-destructive explode inspection;
- Precise, Balanced, and Coarse decomposition presets;
- reconstruction resolution, opacity cutoff, support radius, shell thickness,
  hull budget, and vertices-per-hull controls;
- cancellable browser generation from the decoded Gaussian field;
- staged alignment before applying a sidecar to the scene;
- download of the staged `.phys` candidate;
- independent appearance and hull-preview visibility.

The **Scene** tab controls HDR environment, live lighting, external and self
shadow strength, hull-depth AO, and the animated hull caster.

Run it through the supplied local-server launcher. Do not double-click the HTML
file directly.

Windows:

```text
integrations\walktest\START_SPARK_DEMO.cmd
```

macOS or Linux:

```bash
cd integrations/walktest
./start_spark_demo.sh
```

Both launchers prepare the local npm packages, build the demo, bind the server
to `127.0.0.1:3219`, and open
`http://127.0.0.1:3219/spark-demo`.

## Verification

```bash
cd integrations/web
npm test
npm run build

cd ../walktest
npm run prepare:release-candidate
npm run build:spark-demo
npx playwright test tests/spark-demo.spec.ts --project=chrome
```

The browser test checks the stored sidecar, setup controls, staged/apply
workflow, hull and appearance visibility, deferred-lighting mode, equal
external/self strength, and lighting/AO toggles. Local browser generation is an
authoring path and can take substantially longer than the normal runtime test.

## Current limits

- The production visual and `.phys` must already share a known coordinate
  frame; bounds diagnostics and staged alignment expose mismatches but do not
  infer semantic registration.
- Browser Gaussian reconstruction accepts decoded centers, positive scales,
  xyzw quaternions, and optional opacity. It is not an SPZ file decoder API.
- The browser reconstruction grid is capped at eight million voxels and a
  longest-axis resolution of 160.
- Deferred hull lighting currently supports Three.js/WebGL directional shadow
  maps. Other renderers consume the engine-neutral hull buffers and implement
  their own final composite.
- Hull fidelity is determined by reconstruction and decomposition settings.
  High-detail `.phys` output can closely follow the visible capture; the
  lighting path does not impose a separate low-detail proxy.
