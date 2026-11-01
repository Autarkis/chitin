# @autarkis/chitin-wasm

Native algorithms compiled to WebAssembly for chitin-lite: CoACD convex
decomposition and Poisson surface reconstruction.

Contains two WASM modules. [`@autarkis/chitin-lite`](https://www.npmjs.com/package/@autarkis/chitin-lite)
provides the TypeScript API that loads these binaries. The packages have
independent versions.

**Source build:** this package is not on npm as of 2026-10-31. Build the four
files in `integrations/wasm/` and serve them from your app. The examples
below use `/coacd/` for those self-hosted files. Maintainer build and release
instructions are in [RELEASING.md](RELEASING.md).

## Usage

### CoACD

```typescript
import { initFromUrl, decompose, writePhys } from "@autarkis/chitin-lite";

await initFromUrl(
  "/coacd/coacd.mjs",
  "/coacd/coacd.wasm",
);

const result = await decompose(vertices, faces, { threshold: 0.05 });
const phys = writePhys(result.hulls); // ArrayBuffer, a v3 .phys sidecar
```

### Poisson

```typescript
import { compileGaussianField } from "@autarkis/chitin-lite";
import type { GaussianFieldInput } from "@autarkis/chitin-lite";

const field: GaussianFieldInput = {
  centers: new Float32Array([/* x,y,z, ... */]),
  scales: new Float32Array([/* sx,sy,sz, ... */]),
  rotations: new Float32Array([/* qw,qx,qy,qz, ... */]),
  opacities: new Float32Array([/* alpha, ... */]),
};
const result = await compileGaussianField(field, {
  wasm: {
    js: "/coacd/coacd.mjs",
    wasm: "/coacd/coacd.wasm",
    poissonJs: "/poisson/poisson.mjs",
    poissonWasm: "/poisson/poisson.wasm",
  },
  poissonDepth: 8,
});
console.log(result.phys, result.hulls, result.report);
```

When using published binaries, pin their version in your app or CDN URL.

## Contents

- `coacd.mjs` — Emscripten ES module (`export default createCoACD`)
- `coacd.wasm` — compiled convex decomposer
- `poisson.mjs` — Emscripten ES module (`export default createPoisson`)
- `poisson.wasm` — compiled Poisson surface reconstructor

## Licensing

The `chitin` glue and packaging are MIT (`LICENSE`). The compiled WASM modules
statically include third-party libraries: **CoACD** (MIT) and **CDT** (Mozilla
Public License 2.0) in `coacd.wasm`; **PoissonRecon** (MIT, M. Kazhdan) in
`poisson.wasm`. Their notices and the full MPL-2.0 text ship in the package —
see `THIRD-PARTY-NOTICES.md` and `LICENSE-MPL-2.0.txt`.

Built via Emscripten from `integrations/wasm/` in the
[chitin repo](https://github.com/Autarkis/chitin).
