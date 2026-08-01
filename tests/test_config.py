import pytest

from chitin import Config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"concavity": 0}, "concavity"),
        ({"opacity_threshold": -0.1}, "opacity_threshold"),
        ({"opacity_threshold": 1.1}, "opacity_threshold"),
        ({"poisson_depth": 0}, "poisson_depth"),
        ({"min_hull_vertices": 3}, "min_hull_vertices"),
        ({"max_hulls": 0}, "max_hulls"),
        ({"coacd_preprocess_mode": "maybe"}, "coacd_preprocess_mode"),
        ({"coacd_preprocess_resolution": 0}, "coacd_preprocess_resolution"),
        ({"poisson_density_quantile": -0.1}, "poisson_density_quantile"),
        ({"splat_surface_ratio": -0.1}, "splat_surface_ratio"),
        ({"spatial_split_threshold": 0}, "spatial_split_threshold"),
        ({"surface_proximity_filter": -1}, "surface_proximity_filter"),
        ({"thin_shell_thickness": -1}, "thin_shell_thickness"),
        ({"flatness_threshold": 1.1}, "flatness_threshold"),
        ({"max_decompose_vertices": 99}, "max_decompose_vertices"),
        ({"lod_concavities": [0.1, 0]}, "lod_concavities"),
        # a tier at or below the base concavity would be finer than LOD0
        ({"concavity": 0.3, "lod_concavities": [0.3]}, "lod_concavities"),
        ({"concavity": 0.3, "lod_concavities": [0.5, 0.2]}, "lod_concavities"),
        ({"target_height": 0}, "target_height"),
        ({"target_height": -1.5}, "target_height"),
        ({"target_footprint": 0}, "target_footprint"),
        ({"up_axis": 3}, "up_axis"),
        ({"flat_aspect_ratio": 0}, "flat_aspect_ratio"),
        ({"coacd_timeout": 0}, "coacd_timeout"),
        ({"coacd_timeout": -5}, "coacd_timeout"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_config_accepts_normalization_params():
    config = Config(target_height=0.55, target_footprint=2.0, up_axis=2)
    assert config.target_height == 0.55
    assert config.target_footprint == 2.0
    assert config.up_axis == 2


def test_config_defaults_to_reproducible_decomposition():
    # The default has to be the reproducible one: manifest hashes, the output
    # cache and the robotics gate all assume the same input gives the same
    # bytes, which only holds with CoACD pinned to a single thread.
    from chitin.config import DEFAULT_COACD_TIMEOUT_SECONDS

    assert Config().coacd_deterministic is True
    assert Config().coacd_timeout == DEFAULT_COACD_TIMEOUT_SECONDS
    assert Config(coacd_deterministic=False).coacd_deterministic is False


def test_config_adaptive_preprocess_defaults_on():
    assert Config().coacd_adaptive_preprocess is True
    assert Config(coacd_adaptive_preprocess=False).coacd_adaptive_preprocess is False


def test_config_accepts_default_values():
    config = Config()
    assert config.concavity == 0.05
