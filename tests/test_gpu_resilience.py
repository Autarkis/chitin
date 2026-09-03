import pytest

from chitin.gpu.errors import CapacityError, DeviceLostError
from chitin.gpu.layouts import (
    ArenaDescriptor,
    ArenaKind,
    BoundedArena,
    FieldDef,
    StructLayout,
)
from chitin.gpu.worker import _HAS_WGPU, GPUWorker

TRIVIAL_WGSL = """
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1)
fn main() { data[0] = 42u; }
"""


def _test_layout():
    return StructLayout("Test", [FieldDef("val", "I", 0)], stride=4)


def _test_arena(max_elements=10):
    desc = ArenaDescriptor(
        kind=ArenaKind.SINGLE_LARGE,
        layout=_test_layout(),
        max_elements=max_elements,
        label="test",
    )
    return BoundedArena(desc)


class TestArenaOverflow:
    def test_full_then_overflow(self):
        arena = _test_arena(10)
        arena.allocate(10)
        with pytest.raises(CapacityError):
            arena.allocate(1)

    def test_partial_fill_then_overflow(self):
        arena = _test_arena(10)
        arena.allocate(5)
        arena.allocate(5)
        with pytest.raises(CapacityError):
            arena.allocate(1)

    def test_partial_fill_overflow_mid(self):
        arena = _test_arena(10)
        arena.allocate(5)
        with pytest.raises(CapacityError):
            arena.allocate(6)

    def test_reset_then_reuse(self):
        arena = _test_arena(10)
        arena.allocate(10)
        arena.reset()
        offset, count = arena.allocate(10)
        assert offset == 0
        assert count == 10


@pytest.fixture(scope="module")
def gpu_worker():
    if not _HAS_WGPU or not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


def _make_pipeline_and_bind(worker):
    pipeline = worker.create_compute_pipeline(TRIVIAL_WGSL)
    buf = worker.create_buffer(4, usage=0x80 | 0x08)  # STORAGE | MAP_READ
    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [{"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": 4}}],
    )
    return pipeline, [bind_group], buf


class TestWorkerCancellation:
    def test_cancel_prevents_dispatch(self, gpu_worker):
        pipeline, bind_groups, _ = _make_pipeline_and_bind(gpu_worker)
        gpu_worker.cancel()
        try:
            with pytest.raises(RuntimeError, match="cancelled"):
                gpu_worker.dispatch_compute(pipeline, bind_groups, (1, 1, 1))
        finally:
            gpu_worker.reset_cancel()

    def test_reset_cancel_re_enables(self, gpu_worker):
        pipeline, bind_groups, _ = _make_pipeline_and_bind(gpu_worker)
        gpu_worker.cancel()
        gpu_worker.reset_cancel()
        gpu_worker.dispatch_compute(pipeline, bind_groups, (1, 1, 1))

    def test_cancel_mid_sequence(self, gpu_worker):
        pipeline, bind_groups, _ = _make_pipeline_and_bind(gpu_worker)
        gpu_worker.dispatch_compute(pipeline, bind_groups, (1, 1, 1))
        gpu_worker.cancel()
        try:
            with pytest.raises(RuntimeError, match="cancelled"):
                gpu_worker.dispatch_compute(pipeline, bind_groups, (1, 1, 1))
        finally:
            gpu_worker.reset_cancel()


class TestDeviceLoss:
    def test_create_buffer_raises(self, gpu_worker):
        gpu_worker._device_lost = True
        try:
            with pytest.raises(DeviceLostError):
                gpu_worker.create_buffer(4, usage=0x80)
        finally:
            gpu_worker._device_lost = False

    def test_create_compute_pipeline_raises(self, gpu_worker):
        gpu_worker._device_lost = True
        try:
            with pytest.raises(DeviceLostError):
                gpu_worker.create_compute_pipeline(TRIVIAL_WGSL)
        finally:
            gpu_worker._device_lost = False

    def test_dispatch_compute_raises(self, gpu_worker):
        pipeline, bind_groups, _ = _make_pipeline_and_bind(gpu_worker)
        gpu_worker._device_lost = True
        try:
            with pytest.raises(DeviceLostError):
                gpu_worker.dispatch_compute(pipeline, bind_groups, (1, 1, 1))
        finally:
            gpu_worker._device_lost = False

    def test_read_buffer_raises(self, gpu_worker):
        buf = gpu_worker.create_buffer(4, usage=0x80 | 0x08)
        gpu_worker._device_lost = True
        try:
            with pytest.raises(DeviceLostError):
                gpu_worker.read_buffer(buf, 4)
        finally:
            gpu_worker._device_lost = False
