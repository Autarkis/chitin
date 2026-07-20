# Quickstart — compile colliders in the browser

A single static page that takes a mesh, runs **CoACD convex decomposition in the
browser** via [`@autarkis/chitin-lite`](https://www.npmjs.com/package/@autarkis/chitin-lite),
writes a `.phys` sidecar, reads it back with
[`@autarkis/chitin-web`](https://www.npmjs.com/package/@autarkis/chitin-web), and
drops the resulting convex hulls into a live [Rapier](https://rapier.rs) physics
world rendered with [Three.js](https://threejs.org). Pick a shape, drag the
concavity slider, and rain objects onto the compiled colliders.

Everything loads from CDNs — the packages from esm.sh, the CoACD wasm from
jsDelivr (`@autarkis/chitin-coacd-wasm`) — so there is no build step and nothing
to host but this page.

## Run it locally

Serve the folder over HTTP (ES modules and `fetch` need a real origin, not
`file://`):

```bash
python -m http.server -d examples/quickstart 8000
# open http://localhost:8000
```

## How it works

```
sample mesh ──▶ chitin-lite.decompose(threshold)   # CoACD wasm
            ──▶ chitin-lite.writePhys()             # → .phys ArrayBuffer
            ──▶ chitin-web.parsePhys()              # read the sidecar back
            ──▶ selectLodHulls()  → Three.js meshes # one color per hull
            ──▶ chitin-web/rapier addToWorld()      # fixed physics colliders
```

`samples.json` holds a few watertight source meshes (torus, table, L-beam,
star, …). They are compiled live in your browser — nothing here is precomputed.

## Deploying

`.github/workflows/pages.yml` publishes this folder to GitHub Pages when run
manually from the Actions tab. Deploy it after `@autarkis/chitin-coacd-wasm` is
published, since the page loads the wasm from that CDN URL.
