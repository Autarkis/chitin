# GPU geometry compute (wgpu)

Issue #92. Single-threaded WebGPU compute backend for chitin geometry
operations.

## Modules

- `worker.py` — `GPUWorker`: device lifecycle, pipeline caching,
  bind-group creation, dispatch, cancellation, device-loss handling
- `errors.py` — `CapacityError`, `DeviceLostError`, `ShaderCompilationError`
- `layouts.py` — GPU struct layouts (Point, Triangle, Plane, MeshHeader,
  IntersectionPoint, HullHeader, OutputHeader) with WebGPU alignment,
  `BoundedArena` allocator, `ArenaSet`
- `primitives.py` — CPU reference implementations of GPU compute primitives:
  prefix sum (exclusive/inclusive), compact, stable sort, unique,
  reduce (sum/min/max), segmented scan. Deterministic, matching future
  WGSL shader behavior for NumPy golden conformance.
