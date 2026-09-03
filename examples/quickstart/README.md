# Chitin Collider Lab

The user-facing browser workflow for Chitin: drop a self-contained GLB 2.0,
preview its source geometry, compile convex colliders in a Worker, inspect the
versioned report, and download the resulting `.phys` sidecar.

The viewport reveals finished colliders with a short staggered scale-and-fade
animation so the transition from source geometry to collision geometry is
legible. The non-destructive **Explode** control separates individual hulls for
inspection without changing the downloadable `.phys`. Both transitions respect
the operating system's reduced-motion preference.

The demo runs against the local package builds while Chitin remains
unpublished. It does not upload the selected asset.

Collider Lab is also the maintained shell for Chitin's
[Frontier Labs](../../docs/frontier-labs.md). The current release demonstrates
the completed-GLB/WASM journey. Future admitted stages extend this same surface
with backend comparison, a persistent compiler session, fragment batches,
progressive capture tiles, and atomic Rapier replacement; they do not create a
parallel demo or imply that WebGPU already ships.

Do not open `index.html` directly: the browser's `file://` mode cannot run the
module Worker or load WebAssembly. On Windows, double-click `START_DEMO.cmd` in
this directory. It prepares the local packages, starts the required local-only
server, and opens the demo automatically. On macOS or Linux, run `./start_demo.sh`
instead.

It includes verified offline fixtures for dense smooth, hollow multipart, and
organic geometry. The result view compares measured source/collider complexity
and offers on-demand, deterministic artifact-fit sampling for surface coverage,
volume precision, and false fill. Sampling does not change the report verdict;
profile acceptance checks remain explicitly unevaluated. Fixture provenance and
licenses are recorded in [`public/assets/README.md`](public/assets/README.md).

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

The browser suite compiles the built-in and bundled real-asset GLBs through the
real Worker/WASM path, checks the returned report, and verifies the `.phys`
download in Chromium, Firefox, and WebKit. It also uploads an intentionally open
GLB and verifies that the source remains visible while the UI explains which
connected part needs full-compiler repair. The tests also verify that the
collider reveal reaches its completed state.

`npm run benchmark:samples` is the artifact-quality lane. Its regression floors
and roadmap targets live in [`benchmarks/sample-quality.json`](benchmarks/sample-quality.json),
with metric semantics documented in [`benchmarks/README.md`](benchmarks/README.md).

## Current contract

- Inputs: `File`, `Blob`, buffers, typed-array views, and URLs through
  `ChitinCompiler.compileGlb()`.
- Geometry: static triangle meshes in the active GLB scene.
- Topology: each connected part must be closed and manifold. Open geometry is
  rejected before CoACD with component and edge counts; use full Python Chitin
  for automatic manifold repair.
- Output: convex hulls, `.phys` bytes, source statistics, and
  `CompilationReport` v1.
- Profile: `interactive` only.
- Interactive planning: two workers plan against a deterministic hull budget.
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
- Verdict: `not_evaluated` until artifact-level outcome checks are implemented.
- Quality diagnostics: the result panel can run deterministic, sampled
  source-surface coverage, collider-volume precision, and false-fill
  measurements on demand. `?qualityBenchmark=1` enables the same work
  automatically for regression testing.

The package dependencies intentionally use local `file:` references. Switch
them to released versions only when the npm artifacts containing this API are
ready to publish.

## Planned evidence surface

The Lab will keep stage and end-to-end evidence distinct:

- `wasm | webgpu` requested and effective backend;
- cold and warm time to first usable and final accepted collider;
- source triangles/components and workload class;
- peak memory, hull complexity, quality, and explicit failure reason;
- fragment-batch latency distribution and convex-bypass rate; and
- capture-stage attribution, time to first safe walkable tile, and coverage
  over time.

Complete WebGPU rows appear only after canonical GPU output exists. Earlier
clip+hull numbers remain visibly labeled as partial-stage benchmarks.
