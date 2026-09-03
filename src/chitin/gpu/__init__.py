"""WebGPU compute backend for chitin geometry operations."""

from chitin.gpu.dispatch import (
    dispatch_compact,
    dispatch_prefix_sum_exclusive,
    dispatch_reduce_sum,
    dispatch_segmented_scan,
)
from chitin.gpu.errors import CapacityError, DeviceLostError, ShaderCompilationError
from chitin.gpu.layouts import (
    ALL_LAYOUTS,
    ArenaDescriptor,
    ArenaKind,
    ArenaSet,
    BoundedArena,
    StructLayout,
)
from chitin.gpu.primitives import (
    compact,
    prefix_sum_exclusive,
    prefix_sum_inclusive,
    reduce_max,
    reduce_min,
    reduce_sum,
    segmented_scan,
    stable_sort_by_key,
    unique_sorted,
    unique_with_counts,
)
from chitin.gpu.wgsl_gen import check_drift, generate_all_structs, layout_to_wgsl
from chitin.gpu.worker import GPULimits, GPUWorker

__all__ = [
    "ALL_LAYOUTS",
    "ArenaDescriptor",
    "ArenaKind",
    "ArenaSet",
    "BoundedArena",
    "CapacityError",
    "DeviceLostError",
    "GPULimits",
    "GPUWorker",
    "ShaderCompilationError",
    "StructLayout",
    "check_drift",
    "compact",
    "dispatch_compact",
    "dispatch_prefix_sum_exclusive",
    "dispatch_reduce_sum",
    "dispatch_segmented_scan",
    "generate_all_structs",
    "layout_to_wgsl",
    "prefix_sum_exclusive",
    "prefix_sum_inclusive",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
    "segmented_scan",
    "stable_sort_by_key",
    "unique_sorted",
    "unique_with_counts",
]
