import pytest

from chitin.stages.decompose import (
    ADAPTIVE_HIGH_FACES,
    ADAPTIVE_LOW_FACES,
    ADAPTIVE_MIN_RESOLUTION,
    adaptive_preprocess_resolution,
)


def test_low_poly_assets_get_the_floor():
    # A 676-face catalog panel: the case that took 36s at res=50, ~1s at 30.
    assert adaptive_preprocess_resolution(676, 50) == ADAPTIVE_MIN_RESOLUTION
    assert adaptive_preprocess_resolution(ADAPTIVE_LOW_FACES, 50) == 30


def test_dense_scans_keep_configured_resolution():
    assert adaptive_preprocess_resolution(ADAPTIVE_HIGH_FACES, 50) == 50
    assert adaptive_preprocess_resolution(500_000, 50) == 50


def test_ramps_monotonically_between_anchors():
    configured = 50
    prev = adaptive_preprocess_resolution(ADAPTIVE_LOW_FACES, configured)
    for faces in range(2_000, ADAPTIVE_HIGH_FACES, 5_000):
        res = adaptive_preprocess_resolution(faces, configured)
        assert prev <= res <= configured
        prev = res


def test_never_exceeds_configured_even_when_configured_below_floor():
    # If someone sets a deliberately low resolution, adaptive must not raise it.
    assert adaptive_preprocess_resolution(100, 20) == 20
    assert adaptive_preprocess_resolution(80_000, 20) == 20
    assert adaptive_preprocess_resolution(5_000, 20) == 20


def test_midpoint_is_between_floor_and_configured():
    mid_faces = (ADAPTIVE_LOW_FACES + ADAPTIVE_HIGH_FACES) // 2
    res = adaptive_preprocess_resolution(mid_faces, 50)
    assert 30 < res < 50
    assert res == pytest.approx((30 + 50) / 2, abs=1)
