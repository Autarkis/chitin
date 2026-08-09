import numpy as np

from chitin.analyze import analyze_arrays
from chitin.config import Config
from chitin.resolve import resolve_config

ROOM = (6.0, 3.0, 5.0)


def _plane(rng, n, axis, value, size, jitter=0.02):
    """Scan-like samples on one axis-aligned face of a room."""
    pts = np.empty((n, 3))
    pts[:, axis] = value + rng.normal(0.0, jitter, n)
    for other in (i for i in range(3) if i != axis):
        pts[:, other] = rng.uniform(0.0, size[other], n)
    return pts


def _room(rng=None, size=ROOM, n_per_face=900):
    rng = rng or np.random.default_rng(0)
    faces = [(axis, value) for axis in range(3) for value in (0.0, size[axis])]
    return np.vstack([_plane(rng, n_per_face, a, v, size) for a, v in faces])


def _pillar(rng, center=(3.0, 2.5), half=0.35, height=3.0, n=3000):
    pts = np.empty((n, 3))
    pts[:, 0] = rng.uniform(center[0] - half, center[0] + half, n)
    pts[:, 1] = rng.uniform(0.0, height, n)
    pts[:, 2] = rng.uniform(center[1] - half, center[1] + half, n)
    return pts


def _shelf_row(rng, n=3000):
    """A mid-floor row of shelving: tall, thin, spanning most of the room."""
    pts = np.empty((n, 3))
    pts[:, 0] = rng.uniform(1.0, 5.0, n)
    pts[:, 1] = rng.uniform(0.0, 2.0, n)
    pts[:, 2] = rng.uniform(2.3, 2.7, n)
    return pts


def _solid_block(n=4000, half=2.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.uniform(-half, half, (n, 3))


def test_empty_room_detected_by_both_signals():
    analysis = analyze_arrays(_room())

    assert analysis.is_environment_likely
    assert analysis.inner_density_ratio < 0.05
    assert analysis.wall_faces == 4
    assert analysis.floor_coverage >= 0.35
    assert not analysis.is_environment_ambiguous


def test_room_with_central_pillar_is_still_an_environment():
    rng = np.random.default_rng(1)
    positions = np.vstack([_room(rng=rng), _pillar(rng)])
    analysis = analyze_arrays(positions)

    # The pillar fills the inner AABB, so density alone would call this a solid.
    assert analysis.inner_density_ratio > 0.05
    assert analysis.wall_faces >= 2
    assert analysis.is_environment_likely
    assert not analysis.is_environment_ambiguous


def test_room_with_mid_floor_shelving_is_still_an_environment():
    rng = np.random.default_rng(2)
    positions = np.vstack([_room(rng=rng), _shelf_row(rng)])
    analysis = analyze_arrays(positions)

    assert analysis.inner_density_ratio > 0.05
    assert analysis.is_environment_likely


def test_solid_block_has_no_shell_signature():
    analysis = analyze_arrays(_solid_block())

    # A filled volume spreads through each face slab instead of hugging a plane.
    assert analysis.wall_faces == 0
    assert analysis.floor_coverage == 0.0
    assert not analysis.is_environment_likely


def test_solid_block_density_is_flagged_ambiguous():
    analysis = analyze_arrays(_solid_block())

    assert 0.05 <= analysis.inner_density_ratio < 0.20
    assert analysis.is_environment_ambiguous


def test_dense_fill_is_not_ambiguous():
    # A tight cloud well past the band is unambiguously a solid object.
    rng = np.random.default_rng(3)
    positions = rng.normal(0.0, 1.0, (4000, 3)) * np.array([3.0, 3.0, 3.0])
    analysis = analyze_arrays(positions)

    assert analysis.inner_density_ratio >= 0.20
    assert not analysis.is_environment_ambiguous
    assert not analysis.is_environment_likely


def test_analysis_dict_carries_the_shell_signals():
    data = analyze_arrays(_room()).to_dict()

    assert data["wall_faces"] == 4
    assert data["floor_coverage"] >= 0.35
    assert data["is_environment_ambiguous"] is False


def test_cluttered_room_resolves_to_thin_shell():
    rng = np.random.default_rng(4)
    analysis = analyze_arrays(np.vstack([_room(rng=rng), _pillar(rng)]))
    resolved = resolve_config(Config(), analysis)

    assert resolved.thin_shell
    assert resolved.surface_proximity_filter == 5.0
    assert "wall faces" in resolved.decisions["is_environment"]


def test_force_environment_overrides_detection():
    analysis = analyze_arrays(_solid_block())
    resolved = resolve_config(Config(force_environment=True), analysis)

    assert resolved.thin_shell
    assert resolved.surface_proximity_filter == 5.0
    assert resolved.decisions["is_environment"] == "forced: --environment"


def test_no_auto_environment_still_disables_detection():
    analysis = analyze_arrays(_room())
    resolved = resolve_config(Config(auto_environment=False), analysis)

    assert not resolved.thin_shell
    assert resolved.surface_proximity_filter == 5.0
    assert "is_environment" not in resolved.decisions


def test_point_cloud_gets_proximity_default_without_environment():
    analysis = analyze_arrays(_solid_block())
    resolved = resolve_config(Config(), analysis)

    assert not resolved.thin_shell
    assert resolved.surface_proximity_filter == 5.0
    assert resolved.decisions["surface_proximity_filter"] == (
        "auto: splat reconstruction default"
    )


def test_point_cloud_proximity_default_can_be_disabled():
    analysis = analyze_arrays(_solid_block())
    resolved = resolve_config(Config(surface_proximity_filter=0.0), analysis)

    assert resolved.surface_proximity_filter == 0.0
    assert "surface_proximity_filter" not in resolved.decisions


def test_environment_detected_proximity_default_can_be_disabled():
    rng = np.random.default_rng(4)
    analysis = analyze_arrays(np.vstack([_room(rng=rng), _pillar(rng)]))
    resolved = resolve_config(Config(surface_proximity_filter=0.0), analysis)

    assert resolved.thin_shell
    assert resolved.surface_proximity_filter == 0.0


def test_mesh_input_gets_no_proximity_default():
    analysis = analyze_arrays(_solid_block(), face_count=100)
    resolved = resolve_config(Config(), analysis)

    assert resolved.surface_proximity_filter == 0.0
    assert "surface_proximity_filter" not in resolved.decisions
