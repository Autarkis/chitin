"""Regression cases from spent calibration; never imports G2 fixture data."""

import importlib.util
from pathlib import Path

import pytest

from chitin.gpu.clip_pipeline import dispatch_clip_pipeline
from chitin.gpu.worker import GPUWorker


@pytest.fixture(scope="module")
def gate():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_gpu_gate.py"
    spec = importlib.util.spec_from_file_location("calibration_repair_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker():
    try:
        return GPUWorker()
    except RuntimeError:
        pytest.skip("no GPU adapter")


@pytest.mark.parametrize(
    "fixture,index",
    [("box", 197), ("curved_pipe_quarter", 12)]
    + [("cross_bracket", i) for i in (59, 61, 81, 83, 92, 97)]
    + [("h_shape_corrected", i) for i in (0, 50, 100, 150)],
)
def test_spent_calibration_repairs(gate, worker, fixture, index):
    vertices, faces = gate.CALIBRATION_FIXTURES[fixture]()
    point, normal = gate.generate_planes(vertices, fixture)[index]
    result = dispatch_clip_pipeline(worker, vertices, faces, point, normal)
    assert result.status.is_success
    reference = gate.run_cpu_oracle(vertices, faces, point, normal)
    criteria = gate.check_criteria(result, reference, normal, len(vertices), vertices)
    assert all(criteria.values()), criteria
