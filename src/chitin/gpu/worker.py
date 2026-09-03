"""Single-threaded WebGPU compute worker for chitin geometry operations.

Not thread-safe: a `GPUWorker` owns one wgpu device and must be driven from a
single thread. wgpu-py's sync entry points are not reentrant across threads.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from chitin.gpu.errors import CapacityError, DeviceLostError, ShaderCompilationError

try:
    import wgpu

    _HAS_WGPU = True
except ImportError:
    wgpu = None  # type: ignore[assignment]
    _HAS_WGPU = False


@dataclass(frozen=True)
class GPULimits:
    max_buffer_size: int
    max_workgroup_size: int
    max_storage_buffers: int
    max_compute_invocations_per_workgroup: int
    max_workgroups_per_dimension: int


class GPUWorker:
    def __init__(self, power_preference: str = "high-performance") -> None:
        if not _HAS_WGPU:
            raise RuntimeError("wgpu is not installed; GPUWorker unavailable")

        self._device_lost = False
        adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        self._device = adapter.request_device_sync()
        self.limits = self._read_limits(self._device)
        self._pipeline_cache: dict[str, object] = {}
        self._cancelled = False

    @staticmethod
    def available() -> bool:
        if not _HAS_WGPU:
            return False
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        except (RuntimeError, OSError):
            return False
        return adapter is not None

    @staticmethod
    def _read_limits(device: object) -> GPULimits:
        raw = device.limits
        return GPULimits(
            max_buffer_size=int(raw["max-buffer-size"]),
            max_workgroup_size=int(raw["max-compute-workgroup-size-x"]),
            max_storage_buffers=int(raw["max-storage-buffers-per-shader-stage"]),
            max_compute_invocations_per_workgroup=int(
                raw["max-compute-invocations-per-workgroup"]
            ),
            max_workgroups_per_dimension=int(
                raw["max-compute-workgroups-per-dimension"]
            ),
        )

    def _check_alive(self) -> None:
        if self._device_lost:
            raise DeviceLostError("GPU device has been lost")

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise RuntimeError("Operation cancelled")

    def create_buffer(self, size: int, usage: int) -> object:
        self._check_alive()
        if size > self.limits.max_buffer_size:
            raise CapacityError(
                f"requested buffer size {size} exceeds device max_buffer_size "
                f"{self.limits.max_buffer_size}"
            )
        try:
            return self._device.create_buffer(size=size, usage=usage)
        except Exception as exc:
            if self._device_lost:
                raise DeviceLostError("GPU device has been lost") from exc
            raise

    def create_compute_pipeline(
        self, wgsl_code: str, entry_point: str = "main"
    ) -> object:
        self._check_alive()
        cache_key = hashlib.sha256(f"{wgsl_code}:{entry_point}".encode()).hexdigest()
        cached = self._pipeline_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            shader = self._device.create_shader_module(code=wgsl_code)
        except Exception as exc:
            raise ShaderCompilationError(
                f"failed to compile WGSL shader: {exc}"
            ) from exc
        try:
            pipeline = self._device.create_compute_pipeline(
                layout="auto",
                compute={"module": shader, "entry_point": entry_point},
            )
        except Exception as exc:
            if self._device_lost:
                raise DeviceLostError("GPU device has been lost") from exc
            raise ShaderCompilationError(
                f"failed to create compute pipeline: {exc}"
            ) from exc
        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def create_bind_group(
        self, pipeline: object, group_index: int, entries: list[dict]
    ) -> object:
        self._check_alive()
        layout = pipeline.get_bind_group_layout(group_index)
        return self._device.create_bind_group(layout=layout, entries=entries)

    def dispatch_compute(
        self,
        pipeline: object,
        bind_groups: list[object],
        workgroup_counts: tuple[int, int, int],
    ) -> None:
        self._check_alive()
        self._check_cancelled()
        encoder = self._device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        for index, bind_group in enumerate(bind_groups):
            compute_pass.set_bind_group(index, bind_group)
        compute_pass.dispatch_workgroups(*workgroup_counts)
        compute_pass.end()
        self._device.queue.submit([encoder.finish()])
        self._check_alive()

    def read_buffer(self, buffer: object, size: int) -> bytes:
        self._check_alive()
        data = self._device.queue.read_buffer(buffer, size=size)
        return bytes(data)

    def submit_and_wait(self) -> None:
        self._check_alive()
        encoder = self._device.create_command_encoder()
        self._device.queue.submit([encoder.finish()])
        self._check_alive()

    def cancel(self) -> None:
        self._cancelled = True

    def reset_cancel(self) -> None:
        self._cancelled = False

    def set_progress_callback(self, callback: object | None) -> None:
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int) -> None:
        cb = getattr(self, "_progress_callback", None)
        if cb is not None:
            cb(current, total)

    @property
    def pipeline_cache_size(self) -> int:
        return len(self._pipeline_cache)

    def clear_pipeline_cache(self) -> None:
        self._pipeline_cache.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        device = getattr(self, "_device", None)
        if device is not None:
            destroy = getattr(device, "destroy", None)
            if destroy is not None:
                destroy()
