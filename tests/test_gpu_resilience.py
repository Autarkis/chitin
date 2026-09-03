import pytest

from chitin.gpu.errors import DeviceLostError, OperationCancelledError
from chitin.gpu.worker import _HAS_WGPU, GPUWorker

TRIVIAL_WGSL = """
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1)
fn main() { data[0] = 42u; }
"""


@pytest.fixture(scope="module")
def gpu_worker():
    if not _HAS_WGPU or not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as worker:
        yield worker


def _make_pipeline_and_bind(worker):
    pipeline = worker.create_compute_pipeline(TRIVIAL_WGSL)
    buffer = worker.create_buffer(4, usage=0x80 | 0x08)
    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [{"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": 4}}],
    )
    return pipeline, [bind_group]


class TestWorkerCancellation:
    def test_cancel_discards_completed_operation_result(self, gpu_worker):
        pipeline, bind_groups = _make_pipeline_and_bind(gpu_worker)
        token = gpu_worker.begin_operation()
        gpu_worker.dispatch_compute(pipeline, bind_groups, (1, 1, 1))
        gpu_worker.cancel()
        with pytest.raises(OperationCancelledError, match="superseded"):
            gpu_worker.check_operation(token)

    def test_new_operation_is_valid_after_cancel(self, gpu_worker):
        gpu_worker.cancel()
        token = gpu_worker.begin_operation()
        gpu_worker.check_operation(token)


class TestDeviceLoss:
    def test_mark_device_lost_invalidates_worker(self, gpu_worker):
        gpu_worker.mark_device_lost()
        try:
            with pytest.raises(DeviceLostError):
                gpu_worker.create_buffer(4, usage=0x80)
        finally:
            gpu_worker._device_lost = False

    def test_read_buffer_raises_when_worker_is_lost(self, gpu_worker):
        buffer = gpu_worker.create_buffer(4, usage=0x80 | 0x08)
        gpu_worker.mark_device_lost()
        try:
            with pytest.raises(DeviceLostError):
                gpu_worker.read_buffer(buffer, 4)
        finally:
            gpu_worker._device_lost = False
