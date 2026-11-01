# Chitin WASM Build

Compiles [CoACD](https://github.com/SarahWeiii/CoACD) and [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) to WebAssembly via Emscripten. Produces ES modules + `.wasm` binaries for convex decomposition and surface reconstruction in the browser or a WebWorker.

CoACD is built with `-DWITH_3RD_PARTY_LIBS=OFF`, which strips OpenVDB, Boost, TBB, and spdlog. The core algorithm (MCTS search, concavity metric, plane clipping, convex hull) is unchanged. This build has no automatic manifold repair; input meshes must be manifold.

The CoACD binding enables CoACD's final convex-hull decimation pass, so
`maxChVertex` limits the output vertices per hull. The Node functional test exercises a dense sphere and fails if the emitted hull
exceeds its requested vertex cap.

`src/poisson_bind.cpp` wraps Kazhdan's PoissonRecon via Embind, exposing `poissonReconstruct(positions, normals, depth, densityQuantile)`. Single-threaded (WASM constraint).

## Prerequisites

- [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) (pinned to 5.0.7 in CI)
- CMake >= 3.24

CoACD is pinned to tag **1.0.11**: its `CoACD()` signature matches the 18-argument
call in `src/coacd_bind.cpp`, so an upstream API change can't silently break the
build. PoissonRecon is pinned to commit **`262b0f5`**. `build.sh` clones both automatically if the sources aren't already present.

## Build

```bash
# activate emscripten
source /path/to/emsdk/emsdk_env.sh

# build (clones CoACD 1.0.11 + PoissonRecon to /tmp on first run)
./build.sh
```

Output lands in `dist/`:
- `coacd.mjs` -- Emscripten ES module (`export default createCoACD`)
- `coacd.wasm` -- the compiled decomposer
- `poisson.mjs` -- Emscripten ES module (`export default createPoisson`)
- `poisson.wasm` -- the compiled reconstructor

Override sources or refs:

```bash
COACD_SRC=/path/to/coacd ./build.sh
COACD_REF=<tag> ./build.sh
POISSON_SRC=/path/to/poisson ./build.sh
POISSON_REF=<commit> ./build.sh
```

Both are ES modules (not UMD/CommonJS) so `import(url)` works from a CDN in the
browser and from Node. Built for `web,worker,node`; `test/decompose.test.mjs`
imports CoACD under Node.

## CI and releases

The `wasm-build` and `browser-gate` jobs in `.github/workflows/ci.yml` build
the modules and exercise packed browser consumers. The `wasm-v*` tag workflow
builds, runs Node/wrapper checks, attaches GitHub release assets, and attempts
npm publication.

The WASM package is not on npm as of 2026-10-31. Build all four files and
serve them from your app:

```
/coacd/coacd.mjs
/coacd/coacd.wasm
/coacd/poisson.mjs
/coacd/poisson.wasm
```
