# Chitin browser gates

This package contains two independent Playwright suites:

- `npm test` runs the optional capsule walktest against the `.phys` path in
  `WALKTEST_PHYS`.
- `npm run test:release-candidate` builds and packs Chitin's npm packages,
  installs those tarballs into this isolated consumer, and exercises the real
  packaged Worker, CoACD WASM, `.phys` reader/writer, and Rapier adapter in
  Chromium, Firefox, and WebKit.

The release-candidate gate does not publish packages or create release tags.
It serves Worker/WASM files copied back out of the installed tarballs, so a
missing package file fails before publication.

## Run the release-candidate gate

Build CoACD WASM once if `../wasm/dist/coacd.mjs` and `coacd.wasm` do not exist:

```bash
cd integrations/wasm
bash build.sh
```

Then install and run the browser gate:

```bash
cd integrations/walktest
npm ci
npx playwright install chromium firefox webkit
npm run test:release-candidate
```

The test verifies:

- all three local npm tarballs install and import successfully
- the packaged module Worker loads the packaged CoACD module and binary
- cancellation terminates the Worker and the next compilation recovers
- input buffers use the documented transferable behavior
- two identical compilations produce identical `.phys` hashes within each
  browser/runtime
- the generated `.phys` parses through `@autarkis/chitin-web`
- the public Rapier adapter attaches every compiled hull and participates in a
  physics step
- the packaged browser compiler produces a valid v1 `CompilationReport` and
  does not imply a profile pass before outcome checks run

Same-runtime hashes are the determinism contract tested here. Matching hashes
between Python and WASM, or among different browser engines, are not required.

### Recorded determinism result

The packaged release-candidate gate on 2026-08-31 compiled its fixed L-prism
fixture twice in each browser with `@autarkis/chitin-lite` 0.2.0,
`@autarkis/chitin-wasm` 0.3.0, and `@autarkis/chitin-web` 0.1.2. Both runs
produced SHA-256
`b0ac832370ad5138ba7c1777bb0a3d595ccdf5728600f7c06d783a9306ab7068`
within Chromium, Firefox, and WebKit. The three same-runtime comparisons all
reported `deterministic: true` in
[GitHub Actions run 33428211957](https://github.com/Autarkis/chitin/actions/runs/33428211957).

The identical value across those three engines is recorded evidence from that
run, not a cross-browser parity guarantee. The enforced contract remains two
matching compilations within one supported runtime/toolchain.
