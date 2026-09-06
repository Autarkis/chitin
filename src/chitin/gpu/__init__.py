"""WebGPU compute backend for chitin geometry operations."""

from chitin.gpu.clip import ClipEmitResult, dispatch_clip_emit
from chitin.gpu.dispatch import (
    dispatch_compact,
    dispatch_prefix_sum_exclusive,
    dispatch_reduce_sum,
    dispatch_segmented_scan,
)
from chitin.gpu.dispatch_multi import (
    dispatch_compact_multi,
    dispatch_prefix_sum_exclusive_multi,
    dispatch_reduce_sum_multi,
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
from chitin.gpu.weld import WeldResult, weld_intersections
from chitin.gpu.wgsl_gen import check_drift, generate_all_structs, layout_to_wgsl
from chitin.gpu.worker import GPULimits, GPUWorker

__all__ = [
    "ALL_LAYOUTS",
    "CapacityError",
    "ClipEmitResult",
    "DeviceLostError",
    "GPULimits",
    "GPUWorker",
    "OperationCancelledError",
    "ShaderCompilationError",
    "StructLayout",
    "WeldResult",
    "check_drift",
    "compact",
    "dispatch_clip_emit",
    "dispatch_compact",
    "dispatch_compact_multi",
    "dispatch_prefix_sum_exclusive",
    "dispatch_prefix_sum_exclusive_multi",
    "dispatch_reduce_sum",
    "dispatch_reduce_sum_multi",
    "dispatch_segmented_scan",
    "generate_all_structs",
    "layout_to_wgsl",
    "prefix_sum_exclusive",
    "reduce_sum",
    "segmented_scan",
    "weld_intersections",
]
