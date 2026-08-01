# CoACD WASM Build

Compiles [CoACD](https://github.com/SarahWeiii/CoACD) to WebAssembly via Emscripten. Produces `coacd.mjs` (an ES module) + `coacd.wasm` for convex decomposition in the browser or a WebWorker.

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
- `coacd.mjs` (~96KB) -- Emscripten ES module (`export default createCoACD`)
- `coacd.wasm` (~558KB) -- the compiled decomposer

Override the CoACD source with `COACD_SRC=/path/to/coacd ./build.sh`, or the tag
with `COACD_REF=<tag> ./build.sh`.

It is an ES module (not UMD/CommonJS) so `import(url)` works from a CDN in the
browser and from Node. Built for `web,worker,node`; `test/decompose.test.mjs`
imports it under Node.

## CI and releases

The `wasm` job in `.github/workflows/ci.yml` builds the module, runs the Node
functional test, and enforces a size band on every PR that touches
`integrations/wasm/`. On a `wasm-v*` tag `release-wasm.yml` both attaches `coacd.mjs` + `coacd.wasm` to the GitHub
Release and publishes them to npm as
[`@autarkis/chitin-coacd-wasm`](https://www.npmjs.com/package/@autarkis/chitin-coacd-wasm),
which is the one to use at runtime -- npm CDNs send CORS headers, so it is
fetchable cross-origin from the browser (release assets are not):

```
https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.mjs
https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.wasm
```

## Output size

| File | Size |
|------|------|
| `coacd.wasm` | ~558 KB |
| `coacd.mjs` | ~96 KB |

For comparison, Open3D's Python wheel is ~400 MB.
