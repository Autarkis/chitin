import numpy as np

from chitin.result import Hull
from chitin.verify.coverage import coverage_report


def _box_hull(center=(0.0, 0.0, 0.0), half=1.0):
    c = np.asarray(center, dtype=np.float32)
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float32,
    )
    verts = c + half * signs
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.uint32,
    )
    return Hull(vertices=verts, indices=faces.ravel())


def test_all_points_inside_box_covered():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.9, 0.9, (500, 3))
    report = coverage_report([_box_hull()], pts)
    assert report["covered_fraction"] == 1.0
    assert report["uncovered_count"] == 0
    assert report["slack_p50"] > 0.0
    assert report["sample_count"] == 500


def test_outside_points_uncovered():
    rng = np.random.default_rng(0)
    inside = rng.uniform(-0.9, 0.9, (100, 3))
    outside = rng.uniform(-0.9, 0.9, (100, 3)) + np.array([10.0, 0.0, 0.0])
    report = coverage_report([_box_hull()], np.vstack([inside, outside]))
    assert report["covered_fraction"] == 0.5
    assert report["uncovered_count"] == 100


def test_on_face_point_within_tolerance():
    pts = np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.0, 0.5, 0.5]])
    report = coverage_report([_box_hull()], pts)
    assert report["covered_fraction"] == round(2 / 3, 4)


def test_no_hulls_means_zero_coverage():
    pts = np.zeros((10, 3))
    pts[:, 0] = np.linspace(-1, 1, 10)
    report = coverage_report([], pts)
    assert report["covered_fraction"] == 0.0
    assert report["uncovered_count"] == 10


def test_empty_points():
    report = coverage_report([_box_hull()], np.empty((0, 3)))
    assert report["covered_fraction"] == 0.0
    assert report["sample_count"] == 0


def test_per_cell_worst_decile():
    rng = np.random.default_rng(1)
    covered_pts = rng.uniform(-0.9, 0.9, (50, 3))
    uncovered_pts = rng.uniform(-0.9, 0.9, (50, 3)) + np.array([10.0, 0.0, 0.0])
    pts = np.vstack([covered_pts, uncovered_pts])
    cells = [np.arange(50), np.arange(50, 100)]
    report = coverage_report([_box_hull()], pts, cell_indices=cells)
    assert report["cell_count"] == 2
    assert report["worst_cell_fraction"] == 0.0
    assert report["worst_decile_fraction"] == 0.0
    assert report["worst_cells"][0]["cell"] == 1
    assert report["worst_cells"][0]["samples"] == 50


def test_subsampling_is_deterministic():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.9, 0.9, (1000, 3))
    a = coverage_report([_box_hull()], pts, max_samples=100)
    b = coverage_report([_box_hull()], pts, max_samples=100)
    assert a["sample_count"] == 100
    assert a["input_count"] == 1000
    assert a == b


def test_two_hulls_union_coverage():
    rng = np.random.default_rng(3)
    near = rng.uniform(-0.9, 0.9, (100, 3))
    far = rng.uniform(-0.9, 0.9, (100, 3)) + np.array([10.0, 0.0, 0.0])
    hulls = [_box_hull(), _box_hull(center=(10.0, 0.0, 0.0))]
    report = coverage_report(hulls, np.vstack([near, far]))
    assert report["covered_fraction"] == 1.0
