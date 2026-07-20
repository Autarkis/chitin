# @autarkis/chitin-coacd-wasm

CoACD convex decomposition compiled to WebAssembly, packaged as an ES module.
[`@autarkis/chitin-lite`](https://www.npmjs.com/package/@autarkis/chitin-lite) is
the TypeScript API; this is the `coacd.mjs` + `coacd.wasm` it drives.

The separate package lets the binary be versioned independently. npm CDNs send
CORS headers, so it can be loaded from jsDelivr without hosting it yourself.

## Usage

```typescript
import { initFromUrl, decompose, writePhys } from "@autarkis/chitin-lite";

await initFromUrl(
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.mjs",
  "https://cdn.jsdelivr.net/npm/@autarkis/chitin-coacd-wasm@0.2.0/coacd.wasm",
);

const result = await decompose(vertices, faces, { threshold: 0.05 });
const phys = writePhys(result.hulls); // ArrayBuffer, a v3 .phys sidecar
```

To pin your own copy instead of the CDN, install the package and serve the two
files from your app, then point `initFromUrl` at those paths.

## Contents

- `coacd.mjs` — Emscripten ES module (`export default createCoACD`)
- `coacd.wasm` — the compiled decomposer

## Licensing

The `chitin` glue and packaging are MIT (`LICENSE`). The compiled `coacd.wasm`
statically includes two third-party libraries: **CoACD** (MIT) and **CDT**
(Mozilla Public License 2.0). Their notices and the full MPL-2.0 text ship in the
package — see `THIRD-PARTY-NOTICES.md` and `LICENSE-MPL-2.0.txt`.

Built via Emscripten from `integrations/wasm/` in the
[chitin repo](https://github.com/Autarkis/chitin). CoACD input meshes must be
manifold; the build excludes OpenVDB's repair to keep the module small.
