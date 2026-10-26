# Chitin portable WebGPU CoACD implementation plan

> Status: **accepted direction; blocked on explicit kill gates**
> Date: 2026-09-02
> Stack program: [`lingot/docs/webgpu-coacd-program.md`](../../lingot/docs/webgpu-coacd-program.md)
> Product horizon: [Frontier Labs](frontier-labs.md)

## Objective

Port CoACD's expensive geometry graph to portable WebGPU compute on NVIDIA,
AMD, and Apple: complete clip/cap topology, batched convex hulls, real `Rv`/`Hb`
costs, merge, budgets, and final hull materialization. Candidate geometry stays
in GPU buffers. The small MCTS controller remains on the host and reads only a
bounded scalar cost vector per expansion.

This is the shared acceleration core for Chitin's runtime physicalization
direction. It must serve one-shot assets, many-small fragment batches, and
spatially partitioned capture geometry without weakening correctness. It is not
itself the complete generative, destruction, or capture product.

The program remains gated. A trace-backed `f32` predicate replay is a hard
numerical gate, thin/wiry geometry must have a viable cost path, and combined
GPU clip+hull must beat pinned CPU stages by at least 5× before downstream port
work proceeds.

## Product and adoption thesis

The primary payoff is Chitin adoption outside Lingot: local, interactive
geometry-to-collider compilation for browser 3D, WebXR, UGC, generated assets,
runtime topology changes, and captured environments. The existing raw-hull
APIs and Collider Lab remain the delivery surfaces.

The engine gate and product claim are deliberately separate:

- G3 may publish a labeled clip+hull stage benchmark;
- G6 completes validated GPU decomposition;
- G7 integrates the backend and persistent compiler-session contract; and
- G8 admits certified product claims from end-to-end workload evidence.

The adoption contract is [adoption-strategy.md](adoption-strategy.md). The
three end-state workflows are [frontier-labs.md](frontier-labs.md).

## North-star workload shapes

The kernel design and benchmarks must not assume every request is one ordinary
furniture mesh:

| Shape | Representative input | Engine concern |
|---|---|---|
| novel object | generated or uploaded multipart mesh | time to first usable collider |
| fragment batch | many small, often already-convex shards | scheduling, convex bypass, latency tail |
| capture tiles | uneven reconstructed spatial components | bounded memory, prioritization, progressive output |
| strict artifact | robotics/simulation mesh | acceptance, repeatability, provenance |

CoACD acceleration does not automatically accelerate decoding, fracture,
Poisson reconstruction, spatial partitioning, or physics installation. Those
stages remain separately timed in end-to-end evidence.

## Shipping-contract prerequisites

- [x] [#100](https://github.com/Autarkis/chitin/issues/100) — measured WASM
  determinism, provenance, and build identity baseline.
- [x] [#102](https://github.com/Autarkis/chitin/issues/102) — timeout is a
  structured failure, never AABB substitution.
- [ ] [#103](https://github.com/Autarkis/chitin/issues/103) — shared manifold
  admission and preprocessing boundary.
- [ ] [lingot#64](https://github.com/Autarkis/lingot/issues/64) — logical build
  identity versus exact artifact realization.

## Backend and preprocessing contract

```ts
type DecomposeBackend = "auto" | "webgpu" | "wasm";
```

Forced `webgpu` validates and splits a closed manifold component, invokes GPU
decomposition with preprocessing off, and fails structurally if unavailable or
invalid. It never silently runs WASM. `auto` may select another backend, but a
fallback is a separate reported attempt.

The WebGPU kernel does not initially port CoACD's manifold remesher. Non-manifold
inputs either pass through an explicitly reported repair-capable stage whose
output receives a new effective-input digest, or return `repair_required` when
repair is unavailable. CPU repair is never described as GPU work.

## Execution model

```text
host worker / persistent compiler session
  component split + manifold preflight + convex classification
  seeded MCTS tree/UCB/termination
       |
       | admitted component/candidate batches
       v
GPU arenas
  clip/cap -> batched hull -> Rv/Hb -> scalar cost buffer
       |
       | bounded scalar/status readback
       v
host updates tree; geometry descriptors remain GPU-resident
       |
       v
GPU merge/budgets/canonical hull output -> final bounded readback
       |
       v
shared validation -> ConvexHull[] -> optional .phys -> runtime install/replace
```

The persistent product session reuses the worker, device, pipelines, arenas,
and prepared geometry. The one-shot APIs remain wrappers over that lifecycle.
Progressive and replacement semantics are product contracts layered above the
canonical engine result; they do not permit partial geometry to masquerade as a
final accepted artifact.

Default CoACD search is approximately 150 iterations, depth 3, and 20
candidates per node. Roughly 450 synchronization points are accepted as the
initial design and measured. GPU RNG/UCB/tree buffers remain out of scope unless
profiling proves synchronization dominates.

## Numerical and geometry gates

### G0/G0.5 — trace-backed `f32` policy

Topology depends on predicate signs: clipping, intersection/loop identity, cap
winding, hull horizon, and face orientation. Portable WGSL lacks general `f64`.

The gate records C++ oracle traces and replays classification, intersection
identity, boundary loops, cap topology/winding, hull topology, failure classes,
and candidate ordering under normalized `float32` plus versioned quantized
predicates.

- **Pass:** one scale-relative policy satisfies the admitted corpus without
  asset-specific exceptions.
- **Fail:** stop before production topology WGSL. Emulated `f64` requires a new
  explicit performance-backed decision.

Tracked by [#91](https://github.com/Autarkis/chitin/issues/91),
[#108](https://github.com/Autarkis/chitin/issues/108), and
[#101](https://github.com/Autarkis/chitin/issues/101). The current checked-in
program record is inconclusive; issue state, not an earlier provisional result,
controls the gate.

### G0.6 — thin/wiry cost viability

Near-zero hull volume can make CoACD's normalized cost undefined or noisy even
in the CPU reference. [#109](https://github.com/Autarkis/chitin/issues/109)
must select and validate a cost or pre-decomposition strategy before the GPU
port bakes a known-broken thin-geometry path into WGSL.

### G3 — performance continuation

Warmed combined G2 clip/cap plus G3 hull must be at least **5× faster** than the
pinned CPU stages on the admitted corpus. The corpus includes ordinary
furniture, a generated-style multipart prop, many-small fragment batches, and
uneven scan-like components. Cold and warm results, scheduling overhead, input
shape, and latency distributions remain separate.

This is an engineering continuation gate, not a complete product claim.

## Harness policy

Reuse Lingot's proven WebGPU conformance pattern:

- authoritative source to committed generated WGSL;
- artifact drift tests;
- NumPy semantic goldens;
- identical deterministic vectors through NumPy and WGSL;
- first-divergence reports at predicate/topology level; and
- hardware tests on NVIDIA, AMD, and Apple while CPU layout/artifact tests stay
  mandatory without a GPU.

Chitin copies the harness shape, not Lingot renderer shaders or ownership.

## Dependency-ordered implementation

### G1 — runtime, deterministic primitives, and mesh arena

Worker-owned device/adapter selection, portable limits, aligned layouts,
bounded suballocation, deterministic scan/compaction/stable-sort/unique/fixed
reductions, pipeline caching, cancellation, timeouts, device loss, progress,
and structured capacity errors.

The arena and scheduler must admit both large components and batches of many
small independent components without unchecked writes or starvation.

### G2 — robust clip and cap

Implement classification, edge-plane intersections, stable identity and
deduplication, positive/negative scatter/remap, boundary graph, loop extraction,
deterministic cap triangulation/winding, and explicit coplanar/degenerate
handling. No geometry readback for CPU topology repair.

### G3 — batched hulls and 5× gate

Implement deterministic many-small hull construction: work assignment, extreme
selection, horizon/face replacement, compaction, 1D/2D/coplanar classification,
volume, and canonical emission. Execute the performance gate above before G4.

### G4 — `Rv`, `Hb`, and candidate ordering

Implement mesh/hull volumes, per-side and total `Rv`, sampled bidirectional
Hausdorff behavior, deterministic reductions, and stable ordering. Costs remain
GPU-resident until one bounded vector is read for a host expansion. The selected
thin-geometry policy from G0.6 is part of this contract.

### G5 — host-driven MCTS over GPU batches

Keep seeded RNG, UCB, nodes, visits, rewards, paths, bounds, and termination on
the host. Submit candidate batches against resident geometry; read scalar
costs; retain winning GPU descriptors. The host never recomputes geometry or
cost.

Measure both single-object search and concurrent independent component queues.
Support priority, supersession, and cancellation at safe boundaries for
interactive sessions. Move tree state to GPU only if profiling justifies a new
decision.

### G6 — merge, budgets, and canonical output

Run merge scoring/execution and hull/hull-vertex budgets on GPU, emit canonical
buffers, perform one final bounded readback, and use shared validation. Return
canonical raw hull arrays first; `.phys` persistence is optional. No
GPU-specific acceptance exception exists.

Define compatibility rules for replacing a coarse usable hull set with a final
accepted set without changing object transforms or identity incorrectly.

### G7 — browser and runtime integration

Wire `auto | webgpu | wasm`, common topology preflight, memory admission,
long-lived compiler-session lifecycle, progress/cancel/device loss, explicit
fallback, and logical/artifact identity.

Extend the existing Collider Lab. Add workload selection, backend comparison,
batch submission, convex bypass, progressive result inspection, and atomic
Rapier installation/replacement. Preserve simple one-shot and raw-hull APIs.

### G8 — vendor conformance and product admission

Run one versioned corpus/report on NVIDIA, AMD, and the local Apple MacBook Pro.
Require valid topology, shared-profile acceptance, same-scope repeatability,
CPU-divergence distributions, and exercised repair/fallback/device-loss/capacity
failures.

Product evidence additionally reports:

- novel-object time to first usable and final accepted collider;
- fragment-batch median/p95/worst event latency and frame disruption;
- scan-tile time to first safe walkable region; and
- stage attribution so Poisson, decoding, fracture, or installation bottlenecks
  are not credited to or hidden by CoACD.

Intel and other adapters remain uncertified best-effort until admitted by
hardware evidence.

## Identity fields

The report separates:

- `build_identity`: canonical source/effective-input hashes, algorithm and
  policy versions, resolved configuration/profile, and repair policy;
- `artifact_digest`: exact `.phys` byte digest;
- requested/effective backend and attempt ID;
- preprocessing/repair and effective-input evidence;
- runtime/adapter and deterministic scope;
- workload/session/revision identity where applicable; and
- fallback or failure reason.

One logical build may have multiple vendor artifacts. They remain distinct and
require independent acceptance evidence.

## Filed work

- [#90](https://github.com/Autarkis/chitin/issues/90) — parent engine epic.
- [#31](https://github.com/Autarkis/chitin/issues/31) — product/adoption epic.
- [#91](https://github.com/Autarkis/chitin/issues/91),
  [#108](https://github.com/Autarkis/chitin/issues/108), and
  [#101](https://github.com/Autarkis/chitin/issues/101) — reference and
  numerical gate.
- [#109](https://github.com/Autarkis/chitin/issues/109) — thin-geometry cost
  viability.
- [#92](https://github.com/Autarkis/chitin/issues/92) through
  [#99](https://github.com/Autarkis/chitin/issues/99) — G1-G8 implementation
  and admission.
- [#28](https://github.com/Autarkis/chitin/issues/28) and
  [#30](https://github.com/Autarkis/chitin/issues/30) — runtime integration and
  public evidence.
- [#111](https://github.com/Autarkis/chitin/issues/111) — persistent runtime
  contract; [#112](https://github.com/Autarkis/chitin/issues/112),
  [#113](https://github.com/Autarkis/chitin/issues/113), and
  [#114](https://github.com/Autarkis/chitin/issues/114) — Frontier Labs.

## Success definition

Success requires all hard gates, one shared Chitin WGSL geometry implementation
on NVIDIA/AMD/Apple, accepted raw hull and optional `.phys` output,
host-scheduled search without geometry readback, explicit repair semantics, and
no substitute geometry on failure. The admitted backend runs through existing
APIs plus the persistent compiler session and demonstrates end-to-end value on
the Frontier workload shapes.
