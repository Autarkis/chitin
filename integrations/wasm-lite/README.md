# @autarkis/chitin-lite

Convex decomposition and `.phys` sidecar generation in the browser. Takes mesh vertices + faces, runs CoACD via WebAssembly, and writes portable `.phys` files that any chitin consumer can read.

## Setup

```bash
npm install @autarkis/chitin-lite
```

You also need the CoACD WASM module. It ships as a separate package,
[`@autarkis/chitin-wasm`](https://www.npmjs.com/package/@autarkis/chitin-wasm),
so it can be loaded straight from a CDN (npm CDNs send CORS headers; GitHub
release assets do not):

```
https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.mjs
https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.wasm
```

To pin your own copy, `npm install @autarkis/chitin-wasm` and serve the two
files from your app.

## Usage

### Compile a GLB off the main thread

`compileGlb()` is the user-facing one-shot API. It accepts a `File`, `Blob`,
`ArrayBuffer`, typed-array view, `URL`, or URL string and returns the `.phys`
bytes, hulls, source facts, reuse facts, and the versioned compilation report.

```typescript
import { compileGlb } from "@autarkis/chitin-lite";

const result = await compileGlb(fileInput.files[0], {
  wasm: {
    js: "/coacd/coacd.mjs",
    wasm: "/coacd/coacd.wasm",
    version: "0.2.0",
  },
  decompose: { threshold: 0.10 },
  signal: abortController.signal,
  onProgress: ({ stage, message }) => console.log(stage, message),
});

console.log(result.phys, result.hulls, result.source, result.reuse, result.report);
```

For repeated compiles, reuse a `ChitinCompiler` so its Worker and loaded WASM
stay warm, then call `terminate()` when the UI no longer needs it.

```typescript
import { ChitinCompiler } from "@autarkis/chitin-lite";

const compiler = new ChitinCompiler({
  wasm: { js: "/coacd/coacd.mjs", wasm: "/coacd/coacd.wasm", version: "0.2.0" },
  maxWorkers: 2,
});
const result = await compiler.compileGlb(file, {
  componentPolicy: { maxHulls: 128 },
  onProgress: ({ message, completed, total, eta_ms }) => {
    console.log(message, completed, total, eta_ms);
  },
});
compiler.terminate();
```

This path compiles static triangle geometry from the active scene of a
self-contained GLB 2.0 file. It applies the full node hierarchy, preserves mesh
instancing, and merges indexed/unindexed primitives. Before decomposition it
welds exactly coincident render-seam vertices, removes exact duplicate and
zero-area triangles, and processes disconnected triangle components
independently. The interactive compiler applies a deterministic scene-aware
budget: parts are ranked by scene-relative size, tiny parts collapse to one
convex approximation, and hollow shells and scene-dominant bodies retain
guarded minimum thresholds so a coarse detail setting cannot over-simplify them.
Per-hull vertex ceilings adapt separately from scene-relative size and
roundness. Full thresholds, MCTS defaults, and `componentPolicy` knobs are
documented in
[`docs/usage.md`](../../docs/usage.md#interactive-compiler-budget); this
README covers the package API surface only.

`componentPolicy.maxHulls` is enforced as one total budget and must satisfy the
one-hull-per-component minimum plus any configured hollow-shell reservations.
`decompose.maxConvexHull` retains its low-level meaning as an additional
per-component ceiling. Use `componentPolicy: { enabled: false }` for uniform
full-detail-per-part behavior, or `maxHulls: -1` to opt out of the total cap.
Skins, morph targets, non-triangle primitives, external buffers, and
Draco/meshopt compression are rejected with `UNSUPPORTED_GLTF` instead of being
silently miscompiled. Only
the `interactive` profile is currently accepted; its report verdict remains
`not_evaluated` until artifact-level outcome checks ship.

Artifact-fit measurement is available as explicit opt-in work:

```typescript
const result = await compiler.compileGlb(file, {
  quality: { surfaceSamples: 2048, volumeSamples: 8192 },
});
console.log(result.report.metrics.source_surface_coverage);
console.log(result.report.metrics.false_fill_fraction);
console.log(result.report.metrics.deep_false_fill_fraction);
```

The sampler measures source-surface representation and collider volume in
source-free space. It is disabled during normal interactive compilation.
Metric definitions and verdict semantics live in the
[compilation report contract](../../docs/compilation-report.md).

A reusable compiler caches the prepared geometry for the same immutable
`Blob`/`File`. It also caches completed component results. Slider changes can
therefore reuse the one-hull results for unchanged scene-small parts even when
the active detailed run was cancelled. `result.reuse` reports whether prepared
geometry was reused and how many component results were cache hits. Up to six
recent configurations are retained per component, bounding memory while keeping
nearby detail comparisons fast. Progress events report `completed`,
`total`, and `eta_ms` while components are running. Up to two workers run in
parallel by default; set `maxWorkers` from 1 through 4 on `ChitinCompiler` when
the host needs a different CPU/memory tradeoff. Results are assembled in source
component order, so scheduling does not change artifact ordering. A custom
`workerFactory` must return a fresh Worker instance on every call because each
pool slot owns one worker.

```typescript
componentPolicy: {
  enabled: true,
  maxHulls: 128,
  smallComponentMaxDiagonalRatio: 0.2,
  smallComponentMaxVolumeRatio: 0.005,
  smallComponentThreshold: 1.0,
  detailedComponentMinThreshold: 0.10,
  importantComponentMaxThreshold: 0.14,
  importantComponentMaxOccupancyRatio: 0.50,
  hollowShellMaxOccupancyRatio: 0.05,
  hollowShellMaxThreshold: 0.05,
  hollowShellMinHulls: 8,
  minHullVertices: 8,
  maxHullVertices: 96,
}
```

### Compile a Gaussian field off the main thread

`compileGaussianField()` takes raw Gaussian splat parameters, reconstructs a
surface via Poisson WASM, filters by proximity, decomposes with CoACD, and writes
a `.phys` sidecar. The pipeline is:

1. **TS preprocessing** — estimate normals from the covariance ellipsoids,
   inflate splats by scale to produce oriented point samples.
2. **Poisson WASM reconstruction** — screened Poisson surface reconstruction at
   the requested octree depth.
3. **Proximity filter** — discard reconstructed faces whose vertices are far from
   any input Gaussian (controlled by `surfaceRatio`).
4. **CoACD decomposition** — convex decomposition of the filtered mesh.
5. **`.phys` write** — portable sidecar identical to the GLB path output.

```typescript
import { ChitinCompiler } from "@autarkis/chitin-lite";

const compiler = new ChitinCompiler({
  wasm: {
    js: "/wasm/coacd.mjs",
    wasm: "/wasm/coacd.wasm",
    poissonJs: "/wasm/poisson.mjs",
    poissonWasm: "/wasm/poisson.wasm",
  },
  maxWorkers: 2,
});

const { phys, hulls, report } = await compiler.compileGaussianField(
  { centers, scales, quaternions, opacities },
  {
    poissonDepth: 7,
    densityQuantile: 0.1,
    surfaceRatio: 0.5,
    onProgress: ({ stage, message }) => console.log(stage, message),
  },
);
```

#### `GaussianFieldInput`

```typescript
interface GaussianFieldInput {
  centers: ArrayLike<number>;    // flat xyz, length N*3
  scales: ArrayLike<number>;     // flat xyz scales, length N*3
  quaternions: ArrayLike<number>; // flat xyzw rotations, length N*4
  opacities?: ArrayLike<number>; // per-Gaussian opacity [0,1], length N
}
```

`opacities` gates `densityQuantile` filtering — Gaussians below the quantile are
discarded before Poisson reconstruction. When omitted, all Gaussians contribute.

### Initialize the WASM module

```typescript
import { initFromUrl } from "@autarkis/chitin-lite";

// Point to wherever you host the WASM build output
await initFromUrl(
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.mjs",
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.wasm",
);
```

### Decompose a mesh

```typescript
import { decompose, writePhys } from "@autarkis/chitin-lite";

// vertices: Float64Array (N*3), faces: Int32Array (M*3)
const result = await decompose(vertices, faces, {
  threshold: 0.05, // concavity threshold (lower = more hulls, tighter fit)
});

console.log(`${result.hulls.length} convex hulls`);
```

### Write a .phys sidecar

```typescript
const phys = writePhys(result.hulls);
// phys is an ArrayBuffer -- save it, send it, or feed it to @autarkis/chitin-web
```

### Build a compilation report

The low-level array API can produce the same versioned report shape as the
Python compiler. Because this path does not yet run profile-specific artifact
checks, its verdict is explicitly `not_evaluated`.

```typescript
import { createCompilationReport } from "@autarkis/chitin-lite";

const report = createCompilationReport({
  profile: "interactive",
  input: {
    kind: "typed_arrays",
    source_vertices: vertices.length / 3,
    processed_vertices: vertices.length / 3,
    mesh_vertices: vertices.length / 3,
  },
  hulls: result.hulls,
  phys_bytes: phys.byteLength,
  runtime: {
    kind: "browser_wasm",
    implementation: "@autarkis/chitin-lite",
    version: "0.2.0",
    compiler_version: "0.2.0+chitin-wasm0.2.0",
    dependencies: { "@autarkis/chitin-wasm": "0.2.0" },
  },
});

console.log(report.verdict.status); // "not_evaluated"
```

See the [compilation report contract](../../docs/compilation-report.md) for
serialization and compatibility rules.

### Full pipeline: GLB to a Rapier world

```typescript
import RAPIER from "@dimforge/rapier3d-compat";
import { compileGlb } from "@autarkis/chitin-lite";
import { parsePhys } from "@autarkis/chitin-web";
import { createColliders } from "@autarkis/chitin-web/rapier";

const { phys: physBuffer, report } = await compileGlb(file, {
  wasm: {
    js: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.mjs",
    wasm: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.wasm",
    version: "0.2.0",
  },
});
await RAPIER.init();
const physFile = parsePhys(physBuffer);
const { colliders } = createColliders(RAPIER, physFile);
console.log(report.verdict.status); // "not_evaluated"
```

### Off the main thread (worker API)

`decompose()` runs CoACD synchronously on the calling thread, so a heavy mesh
freezes the UI for the whole run (a detailed torus is several seconds).
`ChitinWorkerClient` moves the work into a Web Worker, keeps the wasm loaded across
calls, and supports cancellation via `AbortSignal`.

```typescript
import { ChitinWorkerClient, writePhys } from "@autarkis/chitin-lite";

const worker = new ChitinWorkerClient(
  {
    js: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.mjs",
    wasm: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.wasm",
  },
);

const controller = new AbortController();
const result = await worker.decompose(vertices, faces, { threshold: 0.05 }, {
  signal: controller.signal, // controller.abort() terminates the run
  checkManifold: true, // recommended for the low-level API -> NON_MANIFOLD
  onState: (state) => console.log(state), // "loading-wasm" -> "decomposing" -> "done"
});
const phys = writePhys(result.hulls);

worker.terminate(); // release the worker when done
```

The input `vertices`/`faces` buffers are transferred to the worker by default
(zero-copy) and detached on your side; pass `{ transferInput: false }` to keep
them. Cancellation terminates the worker (CoACD can't be interrupted mid-run)
and the next `decompose()` spawns a fresh one automatically. Native aborts and
out-of-memory failures also discard the worker so a retry starts with a clean
WASM runtime.

The worker resolves the module via `new URL("./worker.js", import.meta.url)`,
which bundlers (Vite, webpack, Rollup) handle natively. Without a bundler (for example, loading from a CDN),
pass `workerUrl` pointing at the package's `dist/worker.js`:

```typescript
new DecomposeWorker(wasmUrls, {
  workerUrl: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-lite@0.2.0/dist/worker.js",
});
```

### Errors

Failures throw (or reject with) a `ChitinError` carrying a `code`:

| Code | Meaning |
|------|---------|
| `INVALID_GLB` | malformed GLB container or glTF document |
| `UNSUPPORTED_GLTF` | valid glTF feature that cannot be preserved by the static compiler |
| `LOAD_ERROR` | URL or Blob input could not be loaded |
| `COMPILER_BUSY` | another call is already using this reusable compiler |
| `INVALID_MESH` | malformed input geometry (shape, finiteness, index bounds) |
| `INVALID_CONFIG` | a decompose option is out of range |
| `NON_MANIFOLD` | a connected part is open, non-manifold, or degenerate; context includes topology counts |
| `OUT_OF_MEMORY` | the wasm heap could not grow during decomposition |
| `CANCELLED` | the call was aborted and the worker terminated |
| `WORKER_ERROR` | the worker failed to load or crashed |

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.05 low-level; 0.10 effective minimum high-level | CoACD concavity threshold. Lower = more hulls, tighter fit. Set `detailedComponentMinThreshold: 0` to remove the interactive minimum. |
| `maxConvexHull` | -1 low-level | Per-call maximum. In `compileGlb`, this is an additional ceiling for each connected component; use `componentPolicy.maxHulls` for the total budget. |
| `prepResolution` | 50 | Preprocessing resolution. |
| `sampleResolution` | 2000 | Surface sampling resolution. |
| `mctsNodes` | 20 low-level; 8 high-level | MCTS tree width. |
| `mctsIteration` | 150 low-level; 40 high-level | MCTS iterations per node. |
| `mctsMaxDepth` | 3 low-level; 2 high-level | MCTS max search depth. |
| `maxChVertex` | 256 | Max vertices per convex hull. |
| `merge` | true | Merge small adjacent hulls. |

The component-policy fields shown above apply only to high-level GLB
compilation. They do not change the low-level `decompose(vertices, faces)` API.

## Constraints

Gaussian field compilation requires the Poisson WASM URLs (`poissonJs` /
`poissonWasm`) in the `wasm` config object. Calls to `compileGaussianField`
without them reject with `WORKER_ERROR`.

The WASM build requires each connected input part to be a closed manifold and
excludes OpenVDB's manifold repair to keep the module small. The high-level GLB
compiler checks this automatically and rejects with `NON_MANIFOLD` before
entering CoACD. The error identifies the connected part and reports boundary,
non-manifold, and degenerate counts. Use the full Python Chitin compiler when
the geometry needs automatic manifold repair, or close the mesh in a modelling
tool and retry. The low-level array API remains explicit: call
`checkManifold(vertices, faces)` or pass `checkManifold: true` to its worker.
