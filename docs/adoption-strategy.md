# Chitin adoption strategy

> Status: **accepted product direction**
> Date: 2026-09-02
> North star: [Frontier Labs](frontier-labs.md)
> Engineering program: [Portable WebGPU CoACD](webgpu-decomposition.md)
> Browser journey: [chitin#31](https://github.com/Autarkis/chitin/issues/31)

## Thesis

Chitin's adoption wedge is private, local geometry-to-collider compilation for
browser 3D, WebXR, user-generated content, and generated assets. A developer can
hand Chitin ordinary mesh arrays, receive convex hull arrays, and install them
in a physics engine without uploading the asset or adopting a Chitin-specific
artifact format. Validation, provenance, and `.phys` are higher-assurance layers
over that simple engine call.

The portable WebGPU program exists to make this workflow interactive on larger
and more dynamic inputs. Offline Lingot catalog throughput is not its product
justification. The longer horizon is described by the Frontier Labs: generated,
captured, and runtime-mutated geometry entering a persistent physicalization
session.

## Existing product substrate

This strategy extends working surfaces:

- `decompose()` accepts typed mesh arrays and returns `ConvexHull[]`;
- `compileMesh()` and `compileGlb()` return hulls, `.phys`, and a report;
- `compileGaussianField()` reconstructs and decomposes Gaussian inputs;
- `ChitinWorkerClient` moves synchronous WASM work off the main thread;
- Collider Lab supports local GLB upload, inspection, simulation, cancellation,
  recompilation, and `.phys` download; and
- `@autarkis/chitin-web` supplies parsing, Three.js debug geometry, and Rapier
  integration.

The adoption task is to make these surfaces obvious, packaged, measured, and
progressively accelerated—not to hide them behind Lingot.

## Product layers

```text
decompose(positions, indices, options) -> ConvexHull[]
        |
        +-> compileMesh / compileGlb -> hulls + report + optional .phys
        +-> compileGaussianField     -> hulls + report + optional .phys
                |
                +-> Rapier / Three.js / USD / engine adapters
                +-> acceptance, provenance, CAS, and Lingot ledger
```

Principles:

1. Raw-hull users do not need to understand `.phys`, acceptance profiles, or
   Lingot.
2. `.phys` is a durable, fast-loading, multi-runtime artifact—not an API
   tollbooth.
3. Reports are truthful but progressively presented: concise result first,
   detailed evidence on demand.
4. Timeout, invalid topology, device loss, and missing repair capability are
   explicit failures. Chitin does not return a plausible substitute box.
5. Assets remain local in the browser workflow.
6. The persistent compiler session reuses workers, devices, pipelines, arenas,
   and prepared geometry across interactive compiles.

## Audiences and entry points

| Audience | Entry point | Primary value |
|---|---|---|
| Browser 3D, WebXR, UGC | raw hulls, compiler session, Collider Lab, Rapier helper | local time-to-usable-collider |
| Generative 3D | mesh/provider adapters, revision-aware session | physics immediately after geometry generation |
| Reality capture | Gaussian/point adapters, tiled progressive compile | early safe traversal and reusable collision |
| Robotics and simulation | hulls, `.phys`, USD Physics, reports | validated reproducible geometry |
| Lingot | exact artifact refs, logical builds, reports | auditable minting and training inputs |

## Adoption sequence

### A0 — Trust the shipping WASM path

Maintain measured determinism/provenance, explicit timeout failure, and
reproducible native/WASM baselines. Most first-time users encounter WASM before
a certified GPU path; fallback quality is part of the product.

### A1 — Publish the browser journey

Complete [chitin#31](https://github.com/Autarkis/chitin/issues/31): packed npm
artifacts, a short GLB-to-Rapier path, clear recovery, direct hull consumption,
and Collider Lab as the maintained public entry point.

### A2 — Publish benchmark-backed proof

Use [chitin#30](https://github.com/Autarkis/chitin/issues/30) as the evidence
surface. Record source size, workload, policy, cold/warm latency, peak memory,
hull complexity, quality, failure behavior, backend, build identity, and exact
artifact identity. Do not conflate stage, end-to-end, cold, or warm results.

### A3 — Accelerate without overclaiming

Follow the numerical and performance gates in the WebGPU program. G3 may
publish a clip+hull stage benchmark; only G6 can support a complete GPU
decomposition claim. Product admission waits for end-to-end evidence.

### A4 — Establish the persistent physicalization runtime

Add a long-lived session with warm runtime state, component batches,
prioritization, cancellation, convex bypass, progressive results, and atomic
runtime replacement. Preserve the one-shot APIs as convenience wrappers.
Tracked by [chitin#111](https://github.com/Autarkis/chitin/issues/111).

### A5 — Prove the Frontier Labs

Deliver workload-specific Labs for generated assets, topology-change events,
and captured environments. Each Lab earns its own time-to-usable and acceptance
claim; none inherits "instant" from an isolated CoACD kernel benchmark.
Tracked by [chitin#112](https://github.com/Autarkis/chitin/issues/112),
[chitin#113](https://github.com/Autarkis/chitin/issues/113), and
[chitin#114](https://github.com/Autarkis/chitin/issues/114).

## Support policy

- NVIDIA, AMD, and tested Apple cells form the initial certified matrix.
- Intel and other adapters receive no release commitment until tested.
- `auto` is capability- and measurement-based, not a vendor denylist. An
  uncertified adapter may run best-effort and must identify that status.
- forced `webgpu` fails structurally when requirements are unmet.
- WASM/native fallback is explicit and separately reported.

## Metrics

### Product

- time to first usable and final accepted collider;
- cold/warm latency by workload, triangles, components, and search regime;
- median, p95, and worst latency for event workloads;
- successful compile and recovery rate by asset class;
- peak memory, frame disruption, and output complexity;
- time for a new developer to reach a working Rapier collider; and
- downstream integrations and reproducible external failures after release.

### Correctness

- requested profile verdict and complete failure classification;
- coverage, false fill, volume precision, walkability where applicable;
- same-scope repeatability and exact artifact digest;
- logical build identity across vendor realizations; and
- CPU-oracle divergence for admitted WebGPU outputs.

The 5× G2+G3 gate is an engineering continuation criterion. End-to-end time to
a usable, accepted collider is the adoption metric.

## Relationship to Lingot

Chitin's external browser product does not wait on Lingot experiments. Chitin
owns physicalization. Lingot consumes the same reports and artifacts, maintains
cross-repository identity and evidence, and measures whether collider detail is
useful to training. Cadastre registers the selected exact physics reference;
Gesso may preserve visual or capture evidence without owning collision.

## Non-goals

- Advertising complete GPU acceleration before both gates and admission pass.
- Requiring `.phys` or a Lingot ledger for raw-hull consumers.
- Calling completed-file download "streaming geometry."
- Treating topology-change compilation as a per-physics-tick obligation.
- Implying soft-body support from fracture-event collider rebuilding.
- Creating parallel APIs, exporters, or demos where existing surfaces can
  evolve.
