import numpy as np
import pytest

try:
    import trimesh
except ImportError:
    trimesh = None


@pytest.fixture
def box_mesh():
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


@pytest.fixture
def sphere_points():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((500, 3))
    return (pts / np.linalg.norm(pts, axis=1, keepdims=True)).astype(np.float64)


@pytest.fixture
def two_bone_rig():
    left = trimesh.creation.box(
        extents=[1, 1, 1],
        transform=trimesh.transformations.translation_matrix([-1, 0, 0]),
    )
    right = trimesh.creation.box(
        extents=[1, 1, 1],
        transform=trimesh.transformations.translation_matrix([1, 0, 0]),
    )
    combined = trimesh.util.concatenate([left, right])
    vertices = np.asarray(combined.vertices, dtype=np.float32)
    faces = np.asarray(combined.faces, dtype=np.int32)

    joint_indices = np.zeros((len(vertices), 4), dtype=np.int32)
    joint_weights = np.zeros((len(vertices), 4), dtype=np.float64)
    for i, v in enumerate(vertices):
        bone = 0 if v[0] < 0 else 1
        joint_indices[i, 0] = bone
        joint_weights[i, 0] = 1.0

    # Row-vector convention (matches GLTF column-major storage reshaped to numpy)
    inverse_bind_matrices = {
        0: np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
            dtype=np.float64,
        ),
        1: np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 1]],
            dtype=np.float64,
        ),
    }

    return {
        "vertices": vertices,
        "faces": faces,
        "joint_indices": joint_indices,
        "joint_weights": joint_weights,
        "bone_names": ["left_arm", "right_arm"],
        "inverse_bind_matrices": inverse_bind_matrices,
    }
