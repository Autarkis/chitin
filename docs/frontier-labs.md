# Frontier Labs: Chitin's runtime physicalization horizon

> Status: **north-star product direction**
> Date: 2026-09-02
> Engineering program: [Portable WebGPU CoACD](webgpu-decomposition.md)
> Adoption strategy: [Chitin adoption strategy](adoption-strategy.md)

## Thesis

Chitin's end state is an instant physicalization layer: newly generated,
captured, or runtime-mutated geometry can acquire usable collision geometry
locally, without a server-side asset pipeline. The portable WebGPU CoACD engine
is the shared acceleration core. A persistent compiler session, workload-aware
policies, progressive results, and physics-runtime replacement semantics turn
that core into a product.

This document describes the target state, not the current shipping claim.
Public capability claims remain tied to the conformance and end-to-end evidence
required by the engineering program.

## Three workload classes

| Lab | Geometry source | Runtime shape | Primary success metric |
|---|---|---|---|
| **Generative Asset Lab** | completed or progressively produced mesh | one novel object, optionally revised | time to first usable collider |
| **Dynamic Environment Lab** | fracture or committed topology-change event | many small independent fragments | event latency, frame disruption, atomic replacement |
| **Reality Capture Lab** | Gaussian field, point cloud, or scan tiles | large noisy spatial field | time to first walkable region, progressive coverage |

The existing Collider Lab remains the shared shell and evidence surface. The
three Labs extend it with workload-specific flows; they do not create competing
compiler APIs or unrelated demo applications.

## Shared runtime contract

The Labs converge on a long-lived physicalization session:

```text
geometry adapter
  GLB | triangle arrays | component batch | Gaussian field | scan tiles
        |
        v
ChitinCompilerSession
  warm worker/device/pipelines/arenas
  admission + workload policy + priority + cancellation
  convex fast path | WebGPU CoACD | WASM CoACD | Poisson reconstruction
        |
        +----> usable hull set ----------------------+
        |                                            |
        +----> optional refined hull set             v
        |                                      physics runtime
        +----> optional .phys + report          atomic install/replace
```

Required principles:

1. Geometry arrays are the foundational input; GLB, PLY, SPZ/SOG, and provider
   responses are adapters.
2. `ConvexHull[]` is the lowest-latency output. `.phys` remains the durable,
   validated, content-addressed artifact when persistence is useful.
3. A fragment that is already convex takes a verified convex fast path instead
   of paying for full decomposition.
4. Progressive output is monotonic and replaceable: a usable coarse collider
   may be installed before a refined accepted artifact is ready.
5. Replacement in Rapier or another runtime is atomic. Partial installation
   cannot leave a body with a mixed or incomplete collider set.
6. Timeout, device loss, invalid topology, repair-required, and capacity errors
   remain explicit failed attempts, never plausible substitute geometry.

## Lab 1: Generative Asset Lab

### Target experience

A user requests or imports a generated object. Once usable topology is
available, Chitin physicalizes it locally and the object can immediately enter
a Rapier scene. Collider geometry appears as an inspectable overlay and can be
refined or persisted as `.phys`.

### Delivery sequence

1. Completed generated GLB or typed mesh to local compiler session.
2. Provider-neutral adapter contract; no generative vendor becomes a Chitin
   core dependency.
3. Warm compilation and direct hull-to-Rapier installation.
4. Revision-aware compilation when an actual producer exposes stable
   progressive topology or mesh deltas.

"Streaming" means progressive topology only when the producer supplies such a
contract. A download progress bar for one final GLB is not described as
streaming compilation.

### Evidence

- generation-complete to first usable collider;
- cold and warm compiler latency by triangles/components;
- revision reuse and cancelled-work recovery;
- first usable versus final accepted collider quality; and
- exact backend, artifact, and failure identity.

## Lab 2: Dynamic Environment Lab

### Target experience

A fracture or other committed topology-change event produces new components.
Chitin classifies the batch, bypasses decomposition for verified convex shards,
decomposes only the concave remainder, and atomically replaces the physics
representation while the simulation continues.

This is an event-time path inside a live simulation, not a claim that Chitin
runs on every physics tick. Soft-body simulation is not implied: continuous
deformation requires a separate representation and update contract.

### Evidence

- shard-batch admission and convex-bypass rate;
- median, p95, and worst event latency;
- main-thread/frame disruption;
- collider replacement atomicity and contact stability;
- cancellation when a newer topology revision supersedes work; and
- memory reuse across repeated events.

## Lab 3: Reality Capture Lab

### Target experience

A user imports captured Gaussian or point data. Chitin partitions it spatially,
reconstructs and filters surface tiles, and exposes coarse walkable collision as
soon as a safe region is available. Remaining tiles refine in the background
and replace their earlier representations atomically.

Gaussian splats are source evidence here, not Lingot's render-of-record or
durable scene identity. Cadastre may register the resulting structured spatial
and physics references; Gesso may attach capture evidence without elevating the
splat to load-bearing identity.

### Evidence

- decode/preprocess, Poisson, decomposition, validation, and installation time
  reported separately;
- time to first safe walkable region;
- coverage over time and final walkability/clearance;
- peak browser memory and tile scheduling behavior;
- coarse-to-refined replacement count; and
- explicit rejected, repair-required, and capacity outcomes.

CoACD acceleration alone does not establish this Lab. End-to-end profiling must
show whether decoding, spatial partitioning, Poisson reconstruction,
decomposition, or runtime installation sets the latency. Any WebGPU Poisson or
scan-preprocessing program requires its own measured case.

## Workload policies

Names are provisional until implemented, but the roadmap distinguishes these
intentions:

| Policy | Optimizes for | Cannot silently trade away |
|---|---|---|
| `interactive_object` | short user wait on a novel object | topology validity and explicit failure |
| `dynamic_fragment` | event latency and batch throughput | atomic replacement and convex correctness |
| `navigable_capture` | early safe traversal and progressive coverage | clearance/walkability acceptance |
| `robotics` | strict reusable artifact quality | provenance, coverage, and fallback rejection |

Policies are not marketing aliases for arbitrary thresholds. Each receives a
versioned acceptance contract and representative corpus before it ships.

## Repository ownership

- **Chitin** owns geometry admission, decomposition/reconstruction, compiler
  sessions, hulls, reports, `.phys`, and runtime adapters.
- **Atrium** owns the interactive Lab/editor experience and local Apple browser
  evidence; it does not own Chitin geometry shaders.
- **Cadastre** registers spatial identity and exact selected physics references.
- **Gesso** records visual/capture evidence and content-addressed attachments;
  splats remain non-load-bearing presentation or source evidence.
- **Lingot** owns cross-repository orchestration, identity contracts, ledgers,
  and stack-level admission—not Chitin's decomposition implementation.
- **chitin-ingest** supplies headless NVIDIA/AMD execution and batch evidence.

## Roadmap relationship

The G0-G8 WebGPU program remains hard-gated engineering. It delivers the shared
geometry engine and truthful backend admission. Frontier work begins alongside
it only where the shipping WASM path can validate a product contract without
prejudging GPU success.

- Before complete GPU output: specify session, batching, progressive-result,
  and atomic-replacement contracts; prototype bounded Labs on WASM where useful.
- After G3: publish only labeled clip+hull stage evidence.
- After G6: enable complete WebGPU compilation behind explicit admission.
- After G7/G8: publish certified Labs with end-to-end workload evidence.

The destination is not merely "CoACD, but faster." It is one trustworthy
geometry core exposed through a persistent physicalization runtime and proven
against generated, captured, and runtime-mutated worlds.

## Tracking

- [#111](https://github.com/Autarkis/chitin/issues/111) — persistent
  physicalization session and progressive collider contract.
- [#112](https://github.com/Autarkis/chitin/issues/112) — Generative Asset Lab.
- [#113](https://github.com/Autarkis/chitin/issues/113) — Dynamic Environment
  Lab.
- [#114](https://github.com/Autarkis/chitin/issues/114) — Reality Capture Lab.
