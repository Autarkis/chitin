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
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_config_accepts_default_values():
    config = Config()
    assert config.concavity == 0.05
