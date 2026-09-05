"""GPU classify stage tests — verify GPU classification matches CPU reference."""

import numpy as np
import pytest

from chitin.f32_policy import DEFAULT_POLICY
from chitin.f32_predicates import classify_plane_f32
from chitin.gpu.worker import GPUWorker
from chitin.trace_fixtures import FIXTURES


@pytest.fixture(scope="module")
def worker():
    if not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


def _classify_cpu(vertices, plane_point, plane_normal):
    result = classify_plane_f32(
        vertices.astype(np.float64),
        plane_point.astype(np.float64),
        plane_normal.astype(np.float64),
        DEFAULT_POLICY,
    )
    return result.signs


ALL_FIXTURE_NAMES = list(FIXTURES.keys())


class TestGPUClassifyCPUParity:
    @pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
    def test_fixture_parity(self, worker, fixture_name):
        from chitin.f32_replay import generate_test_planes
        from chitin.gpu.classify import dispatch_classify

        vertices, _faces = FIXTURES[fixture_name]()
        vertices = vertices.astype(np.float64)

        for plane_point, plane_normal in generate_test_planes(
            vertices, seed=42, n_random=3
        ):
            cpu_signs = _classify_cpu(vertices, plane_point, plane_normal)
            gpu_signs, _dots = dispatch_classify(
                worker,
                vertices.astype(np.float32),
                plane_point.astype(np.float32),
                plane_normal.astype(np.float32),
                grid_bits=DEFAULT_POLICY.grid_bits,
            )
            np.testing.assert_array_equal(
                gpu_signs,
                cpu_signs,
                err_msg=f"{fixture_name}: GPU/CPU sign mismatch",
            )


class TestGPUClassifyEdgeCases:
    def test_all_positive(self, worker):
        from chitin.gpu.classify import dispatch_classify

        vertices = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, _ = dispatch_classify(worker, vertices, plane_point, plane_normal)
        assert all(s == 1 for s in signs)

    def test_all_negative(self, worker):
        from chitin.gpu.classify import dispatch_classify

        vertices = np.array(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0], [7.0, -8.0, 9.0]],
            dtype=np.float32,
        )
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, _ = dispatch_classify(worker, vertices, plane_point, plane_normal)
        assert all(s == -1 for s in signs)

    def test_on_plane(self, worker):
        from chitin.gpu.classify import dispatch_classify

        vertices = np.array(
            [[1.0, 0.0, 3.0], [4.0, 0.0, 6.0], [7.0, 0.0, 9.0]],
            dtype=np.float32,
        )
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, _ = dispatch_classify(worker, vertices, plane_point, plane_normal)
        assert all(s == 0 for s in signs)

    def test_mixed_signs(self, worker):
        from chitin.gpu.classify import dispatch_classify

        vertices = np.array(
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = dispatch_classify(worker, vertices, plane_point, plane_normal)
        assert signs[0] == 1
        assert signs[1] == -1
        assert signs[2] == 0
        assert dots[0] > 0
        assert dots[1] < 0
        assert dots[2] == 0.0

    def test_empty(self, worker):
        from chitin.gpu.classify import dispatch_classify

        signs, dots = dispatch_classify(
            worker,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        assert len(signs) == 0
        assert len(dots) == 0

    def test_large_vertex_count(self, worker):
        from chitin.gpu.classify import dispatch_classify

        rng = np.random.default_rng(0xBEEF)
        vertices = rng.standard_normal((512, 3)).astype(np.float32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        gpu_signs, _ = dispatch_classify(worker, vertices, plane_point, plane_normal)
        cpu_signs = _classify_cpu(vertices, plane_point, plane_normal)
        np.testing.assert_array_equal(gpu_signs, cpu_signs)
