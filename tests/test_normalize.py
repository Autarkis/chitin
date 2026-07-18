import numpy as np
import pytest

from chitin import Config, extract_from_arrays, extract_from_mesh
from chitin.stages.normalize import normalize_to_target


def _box_points(half=(1.0, 1.0, 1.0)):
    """8 corners of an axis-aligned box centered at the origin."""
    h = np.asarray(half, dtype=np.float64)
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float64,
    )
    return h * signs


def test_no_target_is_noop():
    pts = _box_points()
    out, stats = normalize_to_target(pts)
    assert stats == {}
    assert np.array_equal(out, pts)


def test_empty_input_is_noop():
    out, stats = normalize_to_target(np.empty((0, 3)), target_height=1.0)
    assert stats == {}
    assert len(out) == 0


def test_height_match_scales_up_axis_to_target():
    # Source height (Y extent) is 0.4 -> target 0.55 (a nightstand-ish rescale).
    pts = _box_points(half=(0.5, 0.2, 0.45))
    out, stats = normalize_to_target(pts, target_height=0.55)

    assert stats["normalized"] is True
    assert stats["normalize_matched"] == "height"
    assert stats["normalize_is_flat"] is False
    assert stats["normalize_scale"] == pytest.approx(0.55 / 0.4)

    ext = out.max(axis=0) - out.min(axis=0)
    assert ext[1] == pytest.approx(0.55)  # up axis now metric
    # Uniform scale preserves proportions.
    assert ext[0] / ext[2] == pytest.approx(1.0 / 0.9)


def test_height_match_respects_up_axis():
    # Z-up source: extent along axis 2 is the height to match.
    pts = _box_points(half=(0.5, 0.5, 0.1))
    out, stats = normalize_to_target(pts, target_height=0.75, up_axis=2)
    ext = out.max(axis=0) - out.min(axis=0)
    assert stats["normalize_matched"] == "height"
    assert ext[2] == pytest.approx(0.75)


def test_flat_object_matches_footprint_not_height():
    # A rug: height 0.02, footprint 2.0 -> flat. Matching height would blow the
    # footprint up ~50x; flat-guard matches the footprint instead.
    pts = _box_points(half=(1.0, 0.01, 0.8))  # ext: x=2.0, y=0.02, z=1.6
    out, stats = normalize_to_target(pts, target_height=0.55, target_footprint=2.0)

    assert stats["normalize_is_flat"] is True
    assert stats["normalize_matched"] == "footprint"
    assert stats["normalize_scale"] == pytest.approx(2.0 / 2.0)  # already 2.0 wide
    ext = out.max(axis=0) - out.min(axis=0)
    assert max(ext[0], ext[2]) == pytest.approx(2.0)


def test_flat_object_without_footprint_target_falls_back_to_height():
    # No footprint target supplied: flat-guard cannot fire, height is matched.
    pts = _box_points(half=(1.0, 0.01, 0.8))
    out, stats = normalize_to_target(pts, target_height=0.5)
    assert stats["normalize_matched"] == "height"
    ext = out.max(axis=0) - out.min(axis=0)
    assert ext[1] == pytest.approx(0.5)


def test_tall_object_is_not_flat():
    # A shelf unit: 2.0 tall, 0.8 wide -> not flat, match height.
    pts = _box_points(half=(0.4, 1.0, 0.15))  # ext: x=0.8, y=2.0, z=0.3
    _, stats = normalize_to_target(pts, target_height=2.0, target_footprint=0.8)
    assert stats["normalize_is_flat"] is False
    assert stats["normalize_matched"] == "height"


def test_degenerate_extent_does_not_divide_by_zero():
    # All points share a plane: zero height. With only a height target this is
    # a flagged no-op rather than an inf scale.
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float64)
    out, stats = normalize_to_target(pts, target_height=0.5)
    # height (y) is 0; flat-guard wants footprint but none given -> height path
    # with source 0 -> degenerate.
    assert stats.get("normalized") is False
    assert np.array_equal(out, pts)


def test_scale_is_about_origin():
    # A base-on-floor model (min y = 0) keeps its base on the floor after scale.
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 1]], dtype=np.float64)
    out, _ = normalize_to_target(pts, target_height=1.0)  # y ext 2 -> 1, scale .5
    assert out[:, 1].min() == pytest.approx(0.0)
    assert out.max(axis=0)[1] == pytest.approx(1.0)


def _unit_box(height_y=2.0):
    """8 corners of an axis-aligned box, extent height_y along +y, 12 triangles."""
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, height_y, 0],
            [0, height_y, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, height_y, 1],
            [0, height_y, 1],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return v, f


def test_extract_from_mesh_honors_target_height():
    # Direct mesh entry point must normalize like file-based extract() does.
    v, f = _unit_box(height_y=2.0)
    r = extract_from_mesh(v, f, config=Config(target_height=10.0))
    assert r.build_plan.detected.get("normalized") is True
    assert r.build_plan.detected.get("normalize_scale") == pytest.approx(5.0)
    assert r.hulls
    allv = np.vstack([h.vertices for h in r.hulls])
    height = float(allv[:, 1].max() - allv[:, 1].min())
    assert height == pytest.approx(10.0, rel=0.05)


def test_extract_from_arrays_honors_target_height():
    # Under 100 points returns early, but normalization still runs first, so we
    # can assert the entry point applies the target without needing Open3D.
    pts = np.zeros((50, 3), dtype=np.float64)
    pts[:, 1] = np.linspace(0.0, 2.0, 50)  # y-extent 2
    r = extract_from_arrays(pts, config=Config(target_height=10.0))
    assert r.build_plan.detected.get("normalized") is True
    assert r.build_plan.detected.get("normalize_scale") == pytest.approx(5.0)
