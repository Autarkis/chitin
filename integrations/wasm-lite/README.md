# @autarkis/chitin-lite

Convex decomposition and `.phys` sidecar generation in the browser. Takes mesh vertices + faces, runs CoACD via WebAssembly, and writes portable `.phys` files that any chitin consumer can read.

## Setup

```bash
npm install @autarkis/chitin-lite
```

You also need the CoACD WASM module. It ships as a separate package,
[`@autarkis/chitin-coacd-wasm`](https://www.npmjs.com/package/@autarkis/chitin-coacd-wasm),
so it can be loaded straight from a CDN (npm CDNs send CORS headers; GitHub
release assets do not):

```
https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.mjs
https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.wasm
```

To pin your own copy, `npm install @autarkis/chitin-coacd-wasm` and serve the two
files from your app.

## Usage

### Initialize the WASM module

```typescript
import { initFromUrl } from "@autarkis/chitin-lite";

// Point to wherever you host the WASM build output
await initFromUrl(
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.mjs",
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.wasm",
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

### Full pipeline: GLB to .phys in the browser

```typescript
import RAPIER from "@dimforge/rapier3d-compat";
import { initFromUrl, decompose, writePhys } from "@autarkis/chitin-lite";
import { parsePhys } from "@autarkis/chitin-web";
import { createColliders } from "@autarkis/chitin-web/rapier";

// 1. Init WASM
await initFromUrl(
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.mjs",
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.wasm",
);

// 2. Load mesh (from Three.js, your own loader, etc.)
const vertices = new Float64Array(geometry.attributes.position.array);
const faces = new Int32Array(geometry.index.array);

// 3. Decompose
const result = await decompose(vertices, faces, { threshold: 0.05 });

// 4. Write .phys
const physBuffer = writePhys(result.hulls);

// 5. Read it back and create Rapier colliders
await RAPIER.init();
const physFile = parsePhys(physBuffer);
const { colliders } = createColliders(RAPIER, physFile);
```

### Off the main thread (worker API)

`decompose()` runs CoACD synchronously on the calling thread, so a heavy mesh
freezes the UI for the whole run (a detailed torus is several seconds).
`DecomposeWorker` moves the work into a Web Worker, keeps the wasm loaded across
calls, and supports cancellation via `AbortSignal`.

```typescript
import { DecomposeWorker, writePhys } from "@autarkis/chitin-lite";

const worker = new DecomposeWorker({
  js: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.mjs",
  wasm: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.wasm",
});

const controller = new AbortController();
const result = await worker.decompose(vertices, faces, { threshold: 0.05 }, {
  signal: controller.signal, // controller.abort() terminates the run
  checkManifold: true, // optional precheck -> rejects with NON_MANIFOLD
  onState: (state) => console.log(state), // "loading-wasm" -> "decomposing" -> "done"
});
const phys = writePhys(result.hulls);

worker.terminate(); // release the worker when done
```

The input `vertices`/`faces` buffers are transferred to the worker by default
(zero-copy) and detached on your side; pass `{ transferInput: false }` to keep
them. Cancellation terminates the worker (CoACD can't be interrupted mid-run) and
the next `decompose()` spawns a fresh one automatically.

The worker resolves the module via `new URL("./worker.js", import.meta.url)`,
which bundlers (Vite, webpack, Rollup) handle natively. Without a bundler — e.g.
loading from a CDN — pass `workerUrl` pointing at the package's `dist/worker.js`:

```typescript
new DecomposeWorker(wasmUrls, {
  workerUrl: "https://cdn.jsdelivr.net/npm/@autarkis/chitin-lite@0.2.0/dist/worker.js",
});
```

### Errors

Failures throw (or reject with) a `ChitinError` carrying a `code`:

| Code | Meaning |
|------|---------|
| `INVALID_MESH` | malformed input geometry (shape, finiteness, index bounds) |
| `INVALID_CONFIG` | a decompose option is out of range |
| `NON_MANIFOLD` | input failed the optional `checkManifold` precheck |
| `OUT_OF_MEMORY` | the wasm heap could not grow during decomposition |
| `CANCELLED` | the call was aborted and the worker terminated |
| `WORKER_ERROR` | the worker failed to load or crashed |

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.05 | CoACD concavity threshold. Lower = more hulls, tighter fit. |
| `maxConvexHull` | -1 | Max hulls (-1 = unlimited). |
| `prepResolution` | 50 | Preprocessing resolution. |
| `sampleResolution` | 2000 | Surface sampling resolution. |
| `mctsNodes` | 20 | MCTS tree width. |
| `mctsIteration` | 150 | MCTS iterations per node. |
| `mctsMaxDepth` | 3 | MCTS max search depth. |
| `maxChVertex` | 256 | Max vertices per convex hull. |
| `merge` | true | Merge small adjacent hulls. |

## Constraints

Input meshes must be manifold (watertight, no self-intersections). The WASM build excludes OpenVDB's manifold repair to keep the module under 600KB. OBJ, GLB, and STL files from standard modeling tools are typically manifold. If your mesh isn't, run it through a manifold repair tool first. To catch a bad mesh before decomposition, call `checkManifold(vertices, faces)` (or pass `checkManifold: true` to the worker), which throws `NON_MANIFOLD` on a boundary or non-manifold edge.
