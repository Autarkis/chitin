# CoACD WASM Build

Compiles [CoACD](https://github.com/SarahWeiii/CoACD) to WebAssembly via Emscripten. Produces `coacd.js` + `coacd.wasm` -- state-of-the-art convex decomposition running in the browser or a WebWorker.

Built with `-DWITH_3RD_PARTY_LIBS=OFF`, which strips OpenVDB, Boost, TBB, and spdlog. The core algorithm (MCTS search, concavity metric, plane clipping, convex hull) is unchanged. The only trade-off: no automatic manifold repair. Input meshes must already be manifold.

## Prerequisites

- [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) (tested with 5.0.7)
- CMake >= 3.24
- CoACD source (cloned automatically or provide `COACD_SRC`)

## Build

```bash
# clone CoACD if you haven't already
git clone --depth 1 --recurse-submodules https://github.com/SarahWeiii/CoACD.git /tmp/coacd-src

# activate emscripten
source /path/to/emsdk/emsdk_env.sh

# build
./build.sh
```

Output lands in `dist/`:
- `coacd.js` (~96KB) -- Emscripten module loader
- `coacd.wasm` (~558KB) -- the compiled decomposer

Override the CoACD source location with `COACD_SRC=/path/to/coacd ./build.sh`.

## Output size

| File | Size |
|------|------|
| `coacd.wasm` | ~558 KB |
| `coacd.js` | ~96 KB |

For comparison, Open3D's Python wheel is ~400 MB.
