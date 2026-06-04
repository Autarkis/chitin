import numpy as np

from chitin.analyze import analyze_arrays


def _hollow_shell(n=2000, radius=2.0, rng=None):
    rng = rng or np.random.default_rng(0)
    pts = rng.standard_normal((n, 3))
    return radius * pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _solid_block(n=2000, half=2.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.uniform(-half, half, (n, 3))


def test_hollow_shell_detected_as_environment():
    analysis = analyze_arrays(_hollow_shell())
    assert analysis.is_environment_likely
    assert analysis.inner_density_ratio < 0.05


def test_solid_block_not_environment():
    analysis = analyze_arrays(_solid_block())
    assert not analysis.is_environment_likely
    # Uniform fill puts ~12.5% of points in the inner half-extent box.
    assert analysis.inner_density_ratio > 0.05


def test_small_cloud_never_environment():
    # Fewer than 1000 points returns the dummy ratio 1.0.
    analysis = analyze_arrays(_hollow_shell(n=500))
    assert not analysis.is_environment_likely
    assert analysis.inner_density_ratio == 1.0


def test_small_volume_never_environment():
    # Hollow but tiny: bbox volume below the 10.0 floor is skipped.
    analysis = analyze_arrays(_hollow_shell(radius=0.5))
    assert not analysis.is_environment_likely


def test_boundary_just_above_threshold():
    # Shell plus exactly 6% interior points: ratio over 0.05, not env.
    rng = np.random.default_rng(1)
    shell = _hollow_shell(n=1880, rng=rng)
    interior = rng.uniform(-0.4, 0.4, (120, 3))
    analysis = analyze_arrays(np.vstack([shell, interior]))
    assert analysis.inner_density_ratio >= 0.05
    assert not analysis.is_environment_likely
