from __future__ import annotations

import struct

import pytest

from chitin.gpu.errors import CapacityError, DeviceLostError
from chitin.gpu.worker import _HAS_WGPU, GPUWorker

if _HAS_WGPU:
    import wgpu


def test_worker_available() -> None:
    assert isinstance(GPUWorker.available(), bool)


@pytest.mark.skipif(not _HAS_WGPU, reason="wgpu not installed")
def test_worker_limits() -> None:
    with GPUWorker() as worker:
        assert isinstance(worker.limits.max_buffer_size, int)
        assert worker.limits.max_buffer_size > 0
        assert worker.limits.max_workgroup_size > 0
        assert worker.limits.max_storage_buffers > 0
        assert worker.limits.max_compute_invocations_per_workgroup > 0
        assert worker.limits.max_workgroups_per_dimension > 0


@pytest.mark.skipif(not _HAS_WGPU, reason="wgpu not installed")
def test_create_buffer() -> None:
    with GPUWorker() as worker:
        buffer = worker.create_buffer(
            size=256, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        assert buffer is not None


@pytest.mark.skipif(not _HAS_WGPU, reason="wgpu not installed")
def test_capacity_error() -> None:
    with GPUWorker() as worker:
        with pytest.raises(CapacityError):
            worker.create_buffer(
                size=worker.limits.max_buffer_size + 1,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )


@pytest.mark.skipif(not _HAS_WGPU, reason="wgpu not installed")
def test_trivial_compute() -> None:
    wgsl_code = """
    @group(0) @binding(0)
    var<storage, read_write> out_buf: array<u32>;

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        out_buf[0] = 42u;
    }
    """
    with GPUWorker() as worker:
        buffer = worker.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        pipeline = worker.create_compute_pipeline(wgsl_code, entry_point="main")
        bind_group = worker._device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": 4}}
            ],
        )
        worker.dispatch_compute(pipeline, [bind_group], (1, 1, 1))
        result = worker.read_buffer(buffer, 4)
        (value,) = struct.unpack("<I", result)
        assert value == 42


@pytest.mark.skipif(not _HAS_WGPU, reason="wgpu not installed")
def test_device_lost_flag() -> None:
    with GPUWorker() as worker:
        worker._device_lost = True
        with pytest.raises(DeviceLostError):
            worker.create_buffer(
                size=256, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        with pytest.raises(DeviceLostError):
            worker.submit_and_wait()
