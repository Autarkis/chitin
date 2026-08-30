import numpy as np
import trimesh

from chitin import Config, extract_from_mesh
from chitin.stages.decompose import split_mesh_components


def _translated_box(extents, translation=(0.0, 0.0, 0.0)):
    return trimesh.creation.box(
        extents=extents,
        transform=trimesh.transformations.translation_matrix(translation),
    )


def _l_shape():
    return trimesh.util.concatenate(
        [
            _translated_box([2.0, 0.5, 0.5]),
            _translated_box([0.5, 2.0, 0.5], [0.75, 0.75, 0.0]),
        ]
    )


def _u_shape():
    return trimesh.util.concatenate(
        [
            _translated_box([2.0, 0.4, 0.6]),
            _translated_box([0.4, 1.6, 0.6], [-0.8, 1.0, 0.0]),
            _translated_box([0.4, 1.6, 0.6], [0.8, 1.0, 0.0]),
        ]
    )


def _connected_u_prism():
    """A single watertight, connected concave solid extruded along Z."""
    outline = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [0.6, 2.0],
            [0.6, 0.4],
            [-0.6, 0.4],
            [-0.6, 2.0],
            [-1.0, 2.0],
        ],
        dtype=np.float64,
    )
    vertices = np.vstack(
        [
            np.column_stack([outline, np.full(len(outline), -0.3)]),
            np.column_stack([outline, np.full(len(outline), 0.3)]),
        ]
    )
    cap = np.asarray(
        [
            [0, 1, 4],
            [0, 4, 5],
            [1, 2, 3],
            [1, 3, 4],
            [0, 5, 6],
            [0, 6, 7],
        ],
        dtype=np.int32,
    )
    bottom = cap[:, ::-1]
    top = cap + len(outline)
    sides = []
    for current in range(len(outline)):
        following = (current + 1) % len(outline)
        sides.extend(
            [
                [current, following, following + len(outline)],
                [current, following + len(outline), current + len(outline)],
            ]
        )
    return vertices, np.vstack([bottom, top, np.asarray(sides, dtype=np.int32)])


def test_split_mesh_components_compacts_and_reindexes():
    mesh = _l_shape()
    components = split_mesh_components(mesh.vertices, mesh.faces)

    assert [len(vertices) for vertices, _ in components] == [8, 8]
    assert [len(faces) for _, faces in components] == [12, 12]
    for vertices, faces in components:
        assert faces.min() == 0
        assert faces.max() == len(vertices) - 1


def test_preprocess_off_l_shape_decomposes_per_solid_without_fallback():
    mesh = _l_shape()
    result = extract_from_mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
        config=Config(
            concavity=0.05,
            coacd_preprocess_mode="off",
            coacd_timeout=15.0,
        ),
    )

    assert len(result.hulls) == 2
    assert result.build_plan.detected["mesh_component_count"] == 2
    assert result.build_plan.detected.get("fallback_hulls", 0) == 0


def test_component_decomposition_aggregates_lod_tiers():
    mesh = _l_shape()
    result = extract_from_mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
        config=Config(concavity=0.05, lod_concavities=[0.1], coacd_timeout=15.0),
    )

    assert result.lod_tiers is not None
    assert [(tier.concavity, len(tier.hulls)) for tier in result.lod_tiers] == [
        (0.1, 2)
    ]


def test_original_u_shape_repro_at_point_zero_one_no_longer_stalls():
    mesh = _u_shape()

    result = extract_from_mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
        config=Config(concavity=0.01, coacd_timeout=15.0),
    )

    assert len(result.hulls) == 3
    assert result.build_plan.detected["mesh_component_count"] == 3
    assert result.build_plan.detected.get("fallback_hulls", 0) == 0


def test_preprocess_off_connected_watertight_solid_terminates():
    vertices, faces = _connected_u_prism()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight
    assert mesh.is_volume

    result = extract_from_mesh(
        vertices,
        faces,
        config=Config(
            concavity=0.05,
            coacd_preprocess_mode="off",
            coacd_timeout=15.0,
        ),
    )

    assert len(result.hulls) >= 2
    assert result.build_plan.detected.get("fallback_hulls", 0) == 0
