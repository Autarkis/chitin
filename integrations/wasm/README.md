# Chitin WASM Build

Compiles [CoACD](https://github.com/SarahWeiii/CoACD) and [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) to WebAssembly via Emscripten. Produces ES modules + `.wasm` binaries for convex decomposition and surface reconstruction in the browser or a WebWorker.

CoACD is built with `-DWITH_3RD_PARTY_LIBS=OFF`, which strips OpenVDB, Boost, TBB, and spdlog. The core algorithm (MCTS search, concavity metric, plane clipping, convex hull) is unchanged. The only trade-off: no automatic manifold repair. Input meshes must already be manifold.

The CoACD binding enables CoACD's final convex-hull decimation pass, so
`maxChVertex` is an enforced output limit rather than inert configuration. The
Node functional test exercises a dense sphere and fails if the emitted hull
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
- `coacd.mjs` (~96KB) -- Emscripten ES module (`export default createCoACD`)
- `coacd.wasm` (~558KB) -- the compiled decomposer
- `poisson.mjs` (~TBD) -- Emscripten ES module (`export default createPoisson`)
- `poisson.wasm` (~TBD) -- the compiled reconstructor

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

The `wasm` job in `.github/workflows/ci.yml` builds both modules, runs the Node
functional tests, and enforces a size band on every PR that touches
`integrations/wasm/`. On a `wasm-v*` tag `release-wasm.yml` attaches all four
files to the GitHub Release and publishes them to npm as
[`@autarkis/chitin-wasm`](https://www.npmjs.com/package/@autarkis/chitin-wasm),
which is the one to use at runtime -- npm CDNs send CORS headers, so it is
fetchable cross-origin from the browser (release assets are not):

```
https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.mjs
https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/coacd.wasm
https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/poisson.mjs
https://cdn.jsdelivr.net/npm/@autarkis/chitin-wasm@0.2.0/poisson.wasm
```

## Output size

| File | Size |
|------|------|
| `coacd.wasm` | ~558 KB |
| `coacd.mjs` | ~96 KB |
| `poisson.wasm` | ~TBD |
| `poisson.mjs` | ~TBD |

For comparison, the open3d 0.19.0 Python wheel (cp312) is ~427 MB on Linux (`manylinux_2_31_x86_64`), ~98 MB on macOS (`universal2`), and ~66 MB on Windows (`win_amd64`).
