# Compile contract

Public API contract for `@autarkis/chitin-lite` — the browser/WASM compilation pipeline.

## Input types

| Function | Accepts |
|---|---|
| `compileGlb` | `ArrayBuffer`, `ArrayBufferView`, `Blob`, `File`, `URL`, `string` (URL) |
| `compileMesh` | `Float64Array` (flat xyz vertices), `Int32Array` (triangle indices) |
| `compileGaussianField` | `GaussianFieldInput` (centers, scales, rotations, opacities, SH) |

The Three.js adapter (`@autarkis/chitin-lite/three`) provides:
- `geometryToMesh(geometry)` — extract vertices/faces from a `BufferGeometry`
- `collectMeshes(root)` — merge all meshes under an `Object3D` with world transforms

## Compilation stages

Progress events are emitted in this fixed order:

| Stage | Description |
|---|---|
| `reading-input` | Fetching or reading the input bytes |
| `parsing-input` | Parsing GLB structure / preparing mesh |
| `validating-input` | Manifold checks, component analysis |
| `loading-wasm` | Initializing CoACD WASM workers |
| `decomposing` | Convex decomposition (reported per component) |
| `verifying` | Optional quality measurement |
| `writing-phys` | Serializing the `.phys` sidecar |
| `done` | Compilation complete |

`onProgress` receives `CompilationProgress`:

```ts
interface CompilationProgress {
  stage: CompilationStage;
  message?: string;
  completed?: number;   // components done (decomposing stage)
  total?: number;        // total components
  elapsed_ms?: number;
  eta_ms?: number;
}
```

## Error codes

Every error is a `ChitinError` with a stable `code`, human-readable `message`, and structured metadata.

| Code | Stage | Retryable | When |
|---|---|---|---|
| `INVALID_GLB` | `parsing-input` | No | Malformed GLB container or glTF document |
| `UNSUPPORTED_GLTF` | `parsing-input` | No | Valid glTF feature this compiler cannot handle |
| `LOAD_ERROR` | `reading-input` | Yes* | URL fetch failed or HTTP error (5xx retryable) |
| `COMPILER_BUSY` | `reading-input` | Yes | Reusable compiler has an in-flight call |
| `INVALID_MESH` | `parsing-input` | No | Malformed geometry (shape, finiteness, index bounds) |
| `INVALID_CONFIG` | `validating-input` | No | Decompose option out of valid range |
| `NON_MANIFOLD` | `validating-input` | No | Open, non-manifold, or degenerate mesh |
| `OUT_OF_MEMORY` | `decomposing` | No | WASM heap exhausted |
| `CANCELLED` | any | Yes | Caller aborted via `AbortSignal` |
| `TIMEOUT` | any | Yes | Compilation exceeded `timeout` option |
| `WORKER_ERROR` | any | No | Worker crashed or failed to load |

Every `ChitinError` carries:
- `code` — stable string from the table above
- `message` — human-readable description
- `stage` — which compilation stage was active (or `null`)
- `suggestion` — actionable next step for the user (or `null`)
- `retryable` — whether re-calling with the same input may succeed
- `context` — structured metadata: `mesh_name`, `mesh_index`, `primitive_index`, `component_index`, etc.

## Cancellation

Pass an `AbortSignal` via the `signal` option. The signal is checked between stages and during
worker communication. Aborting during `decomposing` terminates the worker(s); a fresh worker
spawns on the next call.

After cancellation, the same `ChitinCompiler` instance can be reused for a new compilation.

## Timeout

Pass `timeout` (milliseconds) via options. If the compilation exceeds the deadline, it rejects
with code `TIMEOUT`. The signal and workers are cleaned up. The compiler can be reused afterward.

## Retry and recovery

- After `CANCELLED` or `TIMEOUT`: the compiler resets and accepts a new call.
- After `WORKER_ERROR` or `OUT_OF_MEMORY`: the crashed worker is terminated; a fresh one spawns
  on the next call. WASM is re-initialized transparently.
- After `COMPILER_BUSY`: await the active call or create a second `ChitinCompiler`.
- The `terminate()` method releases all workers immediately.

## Compilation report

Every successful compilation returns a `CompilationReport` (version 1):

```ts
interface CompilationReport {
  report_version: 1;
  status: "complete" | "rejected";
  profile: string | null;
  verdict: CompilationVerdict;
  input: { kind, source_vertices, processed_vertices, mesh_vertices };
  output: { collider_kind, hull_count, vertex_count, triangle_count, lod_tier_count, byte_length };
  timings_ms: Record<string, number>;
  warnings: CompilationWarning[];
  metrics: Record<string, CompilationMetric>;
  processing: { pipeline, fallbacks, refinements };
  runtime: CompilationRuntime;
  reproducibility: { scope, deterministic, artifact_sha256 };
  config: { requested, effective };
  artifacts: Record<string, string>;
}
```

The report is validated at construction time. Cross-boundary consumers can re-validate with
`validateCompilationReport(report)`.

## Worker lifecycle

`ChitinCompiler` manages a pool of Web Workers (default 2, max 4). Workers are:
- Spawned lazily on first compile
- Reused across compilations (WASM stays loaded)
- Terminated and respawned after `WORKER_ERROR` / `OUT_OF_MEMORY`
- Terminated on `cancel` / `timeout` (respawned on next call)
- Released by `terminate()` or the one-shot functions (`compileGlb`, `compileMesh`)
