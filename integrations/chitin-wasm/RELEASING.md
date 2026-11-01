# Releasing `@autarkis/chitin-wasm`

This package contains the native CoACD and Poisson WebAssembly modules.
Changesets manages its version; `scripts/changeset-publish.sh` excludes it
because the binaries require an Emscripten build.

## Build and stage

From the repository root, with the pinned Emscripten environment active:

```bash
( cd integrations/wasm && bash build.sh )
cp integrations/wasm/dist/{coacd,poisson}.{mjs,wasm} integrations/chitin-wasm/
```

All four files are required. The package's `prepack` check rejects missing
artifacts. Build TypeScript wrappers and use the packed-consumer tests in
`integrations/walktest` to verify Worker loading, parsing, and Rapier use.
Record the toolchain version and checksums of the binaries and tarball.

## Existing release workflow

`release-wasm.yml` runs on `wasm-v*` tags. It validates the tag against the
package version, builds binaries, runs Node/wrapper checks, attaches GitHub
release assets, and attempts npm publication.

Before publication, verify npm package ownership and trusted-publisher
configuration for the actual workflow and environment. Publish the tested tarball
with an explicit distribution tag. Check registry/CDN
availability before updating consumer examples to that version.

Check the registry for the published version. Source builds can be served
locally before npm publication.
