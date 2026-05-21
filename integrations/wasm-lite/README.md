# @autarkis/chitin-lite

Convex decomposition and `.phys` sidecar generation in the browser. Takes mesh vertices + faces, runs CoACD via WebAssembly, and writes portable `.phys` files that any chitin consumer can read.

## Setup

```bash
npm install @autarkis/chitin-lite
```

You also need the CoACD WASM module. Build it from `integrations/wasm/` or host the pre-built `coacd.js` + `coacd.wasm` alongside your app.

## Usage

### Initialize the WASM module

```typescript
import { initFromUrl } from "@autarkis/chitin-lite";

// Point to wherever you host the WASM build output
await initFromUrl("/wasm/coacd.js", "/wasm/coacd.wasm");
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
import RAPIER from "@dimforge/rapier3d";
import { initFromUrl, decompose, writePhys } from "@autarkis/chitin-lite";
import { parsePhys, createColliders } from "@autarkis/chitin-web";

// 1. Init WASM
await initFromUrl("/wasm/coacd.js", "/wasm/coacd.wasm");

// 2. Load mesh (from Three.js, your own loader, etc.)
const vertices = new Float64Array(geometry.attributes.position.array);
const faces = new Int32Array(geometry.index.array);

// 3. Decompose
const result = await decompose(vertices, faces, { threshold: 0.05 });

// 4. Write .phys
const physBuffer = writePhys(result.hulls);

// 5. Read it back and create Rapier colliders
const physFile = parsePhys(physBuffer);
const { colliders } = createColliders(rapier, physFile);
```

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

Input meshes must be manifold (watertight, no self-intersections). The WASM build excludes OpenVDB's manifold repair to keep the module under 600KB. OBJ, GLB, and STL files from standard modeling tools are typically manifold. If your mesh isn't, run it through a manifold repair tool first.
