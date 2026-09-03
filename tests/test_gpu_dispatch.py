"""GPU dispatch golden tests — run WGSL shaders on real hardware,
compare byte-identical output against CPU reference and golden vectors."""

import numpy as np
import pytest

from chitin.gpu.dispatch import (
    dispatch_compact,
    dispatch_prefix_sum_exclusive,
    dispatch_reduce_sum,
    dispatch_segmented_scan,
)
from chitin.gpu.worker import GPUWorker

# Golden vectors from test_gpu_golden.py
GOLDEN_SCAN_INPUT = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32)
GOLDEN_SCAN_EXCLUSIVE_I32 = np.array([0, 3, 4, 8, 9, 14, 23, 25], dtype=np.int32)
GOLDEN_SCAN_TOTAL = 31

GOLDEN_COMPACT_VALUES = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
GOLDEN_COMPACT_MASK = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int32)
GOLDEN_COMPACT_RESULT = np.array([10, 30, 40, 70], dtype=np.int32)
GOLDEN_COMPACT_COUNT = 4

GOLDEN_SEG_VALUES = np.array([1, 2, 3, 10, 20, 30, 100], dtype=np.int32)
GOLDEN_SEG_IDS = np.array([0, 0, 0, 1, 1, 1, 2], dtype=np.int32)
GOLDEN_SEG_RESULT = np.array([0, 1, 3, 0, 10, 30, 0], dtype=np.int32)


@pytest.fixture(scope="module")
def worker():
    if not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


class TestGPUPrefixSum:
    def test_golden_vector(self, worker):
        result, total = dispatch_prefix_sum_exclusive(worker, GOLDEN_SCAN_INPUT)
        np.testing.assert_array_equal(result, GOLDEN_SCAN_EXCLUSIVE_I32)
        assert total == GOLDEN_SCAN_TOTAL

    def test_empty(self, worker):
        result, total = dispatch_prefix_sum_exclusive(
            worker, np.array([], dtype=np.int32)
        )
        assert len(result) == 0
        assert total == 0

    def test_single_element(self, worker):
        result, total = dispatch_prefix_sum_exclusive(
            worker, np.array([42], dtype=np.int32)
        )
        np.testing.assert_array_equal(result, np.array([0], dtype=np.int32))
        assert total == 42

    def test_max_workgroup(self, worker):
        rng = np.random.default_rng(0xBEEF)
        values = rng.integers(0, 100, size=256, dtype=np.int32)
        result, total = dispatch_prefix_sum_exclusive(worker, values)
        expected = np.zeros(256, dtype=np.int32)
        acc = np.int32(0)
        for i in range(256):
            expected[i] = acc
            acc += values[i]
        np.testing.assert_array_equal(result, expected)
        assert total == int(acc)


class TestGPUReduceSum:
    def test_golden_vector(self, worker):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        result = dispatch_reduce_sum(worker, values)
        assert result == pytest.approx(36.0)

    def test_single(self, worker):
        result = dispatch_reduce_sum(worker, np.array([42.5], dtype=np.float32))
        assert result == pytest.approx(42.5)

    def test_empty(self, worker):
        result = dispatch_reduce_sum(worker, np.array([], dtype=np.float32))
        assert result == 0.0


class TestGPUCompact:
    def test_golden_vector(self, worker):
        result, count = dispatch_compact(
            worker, GOLDEN_COMPACT_VALUES, GOLDEN_COMPACT_MASK
        )
        np.testing.assert_array_equal(result, GOLDEN_COMPACT_RESULT)
        assert count == GOLDEN_COMPACT_COUNT

    def test_all_kept(self, worker):
        values = np.array([1, 2, 3], dtype=np.int32)
        mask = np.array([1, 1, 1], dtype=np.int32)
        result, count = dispatch_compact(worker, values, mask)
        np.testing.assert_array_equal(result, values)
        assert count == 3

    def test_none_kept(self, worker):
        values = np.array([1, 2, 3], dtype=np.int32)
        mask = np.array([0, 0, 0], dtype=np.int32)
        _result, count = dispatch_compact(worker, values, mask)
        assert count == 0

    def test_empty(self, worker):
        _result, count = dispatch_compact(
            worker,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
        assert count == 0


class TestGPUSegmentedScan:
    def test_golden_vector(self, worker):
        result = dispatch_segmented_scan(worker, GOLDEN_SEG_VALUES, GOLDEN_SEG_IDS)
        np.testing.assert_array_equal(result, GOLDEN_SEG_RESULT)

    def test_single_segment(self, worker):
        values = np.array([1, 2, 3, 4], dtype=np.int32)
        ids = np.array([0, 0, 0, 0], dtype=np.int32)
        result = dispatch_segmented_scan(worker, values, ids)
        np.testing.assert_array_equal(result, np.array([0, 1, 3, 6], dtype=np.int32))

    def test_each_own_segment(self, worker):
        values = np.array([10, 20, 30], dtype=np.int32)
        ids = np.array([0, 1, 2], dtype=np.int32)
        result = dispatch_segmented_scan(worker, values, ids)
        np.testing.assert_array_equal(result, np.array([0, 0, 0], dtype=np.int32))
