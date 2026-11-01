# Chitin Collider Lab

Upload a self-contained GLB 2.0 file, compile collision hulls in a Worker,
inspect them in the viewport, run a Rapier simulation, and download `.phys`.
The selected asset stays in your browser.

The demo uses local package builds. Its API may be ahead of the npm release.

Use `START_DEMO.cmd` on Windows or `./start_demo.sh` on macOS/Linux to prepare
packages and open the demo. Opening `index.html` directly will not work:
Workers and WebAssembly need a local HTTP server.

The bundled samples cover smooth, hollow multipart, and organic geometry.
Toggle source and collider layers, inspect individual hulls with **Explode**,
or measure surface coverage and false fill. Collider reveal and explosion
animations respect reduced-motion settings. Sample licenses and provenance
are in [public/assets/README.md](public/assets/README.md).

## Run locally

Build the package dependencies first:

```bash
cd integrations/wasm-lite
npm ci
npm run build

cd ../web
npm ci
npm run build
```

The CoACD artifacts must exist at `integrations/chitin-wasm/coacd.mjs` and
`coacd.wasm`. Copy them from `integrations/wasm/dist/` after running the WASM
build when necessary. Then:

```bash
cd examples/quickstart
npm ci
npm run dev
# http://127.0.0.1:4179
```

Alternatively, run `npm start` to start the same server and open the browser.

## Verify

```bash
npm run build
npx playwright install chromium firefox webkit
npm test
npm run benchmark:samples
```

Browser tests cover Worker/WASM compilation, reports, downloads, simulation,
and recovery in Chromium, Firefox, and WebKit. They also check that rejected
open meshes remain visible with a repair explanation.

Quality thresholds are in
[benchmarks/sample-quality.json](benchmarks/sample-quality.json); measurement
definitions are in [benchmarks/README.md](benchmarks/README.md).

## Supported inputs and controls

- Inputs: `File`, `Blob`, buffers, typed-array views, and URLs through
  `ChitinCompiler.compileGlb()`.
- Geometry: static triangle meshes in the active GLB scene.
- Topology: each connected part must be closed and manifold. Open geometry is
  rejected before CoACD with component and edge counts; use full Python Chitin
  for automatic manifold repair.
- Output: convex hulls, `.phys` bytes, source statistics, and
  `CompilationReport` v1.
- Profiles: `interactive`, `walkable`, and `robotics`. The latter two request
  sampled quality diagnostics; unavailable native checks remain unevaluated.
- Hull budget: two workers share a deterministic scene budget.
  See [Interactive compiler budget](../../docs/usage.md#interactive-compiler-budget)
  for the ceiling, MCTS search settings, and the shell/importance/vertex guards.
- Fit controls: named Precise, Game prop, and Coarse presets set the same
  concavity-tolerance slider exposed for manual tuning.
- Recompilation: prepared `File` geometry, the source preview, warm WASM workers,
  and completed compatible components are reused when the detail slider restarts
  a build. Obsolete work is cancelled as soon as the slider moves; the previous
  collider remains visible until its replacement is complete. The result line
  reports component reuse, and each component keeps its six most-recent settings
  to bound memory while preserving fast back-and-forth comparisons.
- Progress: connected-part completion and an estimated remaining time are shown
  during decomposition.
- Verdict: interactive remains `not_evaluated`. Walkable/robotics can fail
  measured quality thresholds, but cannot claim a complete pass while required
  capsule/walkable probes or snug-fit checks remain unavailable in WASM.
- Quality diagnostics: the result panel can run deterministic, sampled
  source-surface coverage, collider-volume precision, and false-fill
  measurements on demand. `?qualityBenchmark=1` enables the same work
  automatically for regression testing.

Local `file:` dependencies provide the compiler API used by this demo.
