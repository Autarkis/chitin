"""WebGPU compute backend for chitin geometry operations."""

from chitin.gpu.dispatch import (
    dispatch_compact,
    dispatch_prefix_sum_exclusive,
    dispatch_reduce_sum,
    dispatch_segmented_scan,
)
from chitin.gpu.errors import (
    CapacityError,
    DeviceLostError,
    OperationCancelledError,
    ShaderCompilationError,
)
from chitin.gpu.layouts import (
    ALL_LAYOUTS,
    StructLayout,
)
from chitin.gpu.primitives import (
    compact,
    prefix_sum_exclusive,
    reduce_sum,
    segmented_scan,
)
from chitin.gpu.wgsl_gen import check_drift, generate_all_structs, layout_to_wgsl
from chitin.gpu.worker import GPULimits, GPUWorker

__all__ = [
    "ALL_LAYOUTS",
    "CapacityError",
    "DeviceLostError",
    "GPULimits",
    "GPUWorker",
    "OperationCancelledError",
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
    "reduce_sum",
    "segmented_scan",
]
