"""prefix_sum_gpu: GPU-only exclusive scan returning (offsets_buf, total_buf).

The multi-block path (n > 256) must stay on the GPU (no mid-scan readback) and
return a total buffer that is itself readable (COPY_SRC), like the single-block
path's.
"""

import numpy as np
import pytest

from chitin.gpu.dispatch_multi import _buffer_from_numpy, prefix_sum_gpu
from chitin.gpu.worker import GPUWorker


@pytest.fixture(scope="module")
def worker():
    if not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


@pytest.mark.parametrize("n", [1, 255, 256, 257, 1000, 65536])
def test_prefix_sum_gpu_matches_numpy(worker, n):
    rng = np.random.default_rng(n)
    counts = rng.integers(0, 8, size=n, dtype=np.int32)
    expected = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)[:-1])).astype(
        np.int32
    )
    expected_total = int(counts.sum())

    input_buf = _buffer_from_numpy(worker, counts)

    real_read = worker.read_buffer

    def forbidden(*args, **kwargs):
        raise AssertionError("prefix_sum_gpu must not read back mid-pipeline")

    worker.read_buffer = forbidden
    try:
        offsets_buf, total_buf = prefix_sum_gpu(worker, input_buf, n)
    finally:
        worker.read_buffer = real_read

    offsets = np.frombuffer(worker.read_buffer(offsets_buf, n * 4), dtype=np.int32)
    total = np.frombuffer(worker.read_buffer(total_buf, 4), dtype=np.int32)[0]

    np.testing.assert_array_equal(offsets, expected)
    assert int(total) == expected_total


def test_prefix_sum_gpu_rejects_over_cap(worker):
    n = 256 * 256 + 1
    input_buf = _buffer_from_numpy(worker, np.zeros(n, dtype=np.int32))
    with pytest.raises(ValueError):
        prefix_sum_gpu(worker, input_buf, n)
