# CoACD WASM Build

Compiles [CoACD](https://github.com/SarahWeiii/CoACD) to WebAssembly via Emscripten. Produces `coacd.js` + `coacd.wasm` -- state-of-the-art convex decomposition running in the browser or a WebWorker.

Built with `-DWITH_3RD_PARTY_LIBS=OFF`, which strips OpenVDB, Boost, TBB, and spdlog. The core algorithm (MCTS search, concavity metric, plane clipping, convex hull) is unchanged. The only trade-off: no automatic manifold repair. Input meshes must already be manifold.

## Prerequisites

- [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) (pinned to 5.0.7 in CI)
- CMake >= 3.24

CoACD is pinned to tag **1.0.11**: its `CoACD()` signature matches the 18-argument
call in `src/coacd_bind.cpp`, so an upstream API change can't silently break the
build. `build.sh` clones that tag automatically if the source isn't already present.

## Build

```bash
# activate emscripten
source /path/to/emsdk/emsdk_env.sh

# build (clones CoACD 1.0.11 to /tmp/coacd-src on first run)
./build.sh
```

Output lands in `dist/`:
- `coacd.js` (~96KB) -- Emscripten module loader
- `coacd.wasm` (~558KB) -- the compiled decomposer

Override the CoACD source with `COACD_SRC=/path/to/coacd ./build.sh`, or the tag
with `COACD_REF=<tag> ./build.sh`.

The module is built for `web,worker,node` so the same output runs in the browser,
a WebWorker, and Node (the CI functional test in `test/decompose.test.cjs` loads
it under Node and asserts a known concave mesh decomposes into multiple hulls).

## CI and releases

`.github/workflows/build-wasm.yml` builds the module, runs the Node functional
test, and enforces a size band on every PR that touches `integrations/wasm/`.
On a `wasm-v*` tag it attaches `coacd.js` + `coacd.wasm` to the GitHub Release,
so consumers can fetch a versioned build directly:

```
https://github.com/Autarkis/chitin/releases/download/wasm-v0.1.2/coacd.js
https://github.com/Autarkis/chitin/releases/download/wasm-v0.1.2/coacd.wasm
```

## Output size

| File | Size |
|------|------|
| `coacd.wasm` | ~558 KB |
| `coacd.js` | ~96 KB |

For comparison, Open3D's Python wheel is ~400 MB.
