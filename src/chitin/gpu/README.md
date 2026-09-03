# GPU geometry compute (wgpu)

Issue #92. Single-threaded WebGPU compute backend for chitin geometry
operations.

## Modules

- `worker.py` — `GPUWorker`: device lifecycle, pipeline caching,
  bind-group creation, dispatch, command-boundary cancellation, and device-loss
  invalidation
- `errors.py` — `CapacityError`, `DeviceLostError`, `OperationCancelledError`,
  `ShaderCompilationError`
- `layouts.py` — GPU struct layouts (Point, Triangle, Plane, MeshHeader,
  IntersectionPoint, HullHeader, OutputHeader). NumPy packing and WGSL field
  layout are validated from the same definitions.
- `dispatch.py` — `KernelSpec` metadata validates every one-workgroup kernel's
  inputs and constructs its bindings consistently.
- `primitives.py` — CPU reference implementations only for shipped WGSL
  kernels: exclusive scan, compact, sum reduction, and segmented scan. Their
  fixed-width arithmetic and accumulation order match the shader contract.

GPU-resident arenas intentionally arrive with the first geometry stage that
uses them. A bookkeeping-only allocator would not provide the persistent memory
reuse promised by the compiler-session design.
