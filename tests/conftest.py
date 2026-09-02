import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

CI_CORPUS_DIGESTS = {
    "box": "f2778d3f5ddb58e309bf903667899940b9cbed192b9102791194d889697f125c",
    "icosphere": "5ff57d55f916ed43e9c54c359f7bb2bde9545248e426625d26bfc853025e0e87",
    "thin_panel": "874302dd2e001fb74d1235ba9302100464a78f246e4d48f831d013a9fddf57c5",
    "l_shape": "48a45262c932e01d278544522f15d45d25b59a9f3bc920f5ddcbbdd0e8f68420",
    "thin_u_channel": "1b1223a253fedc2a86e6c000d6a5d873bf586aab0bbdf04f5175bcae9bb36e40",
    "cross_bracket": "fa2e275d0da0bc8c539b0b772e0795d3818f883af9a91c3b2a810544f174593e",
    "staircase": "9f9be026aecacccb40891d0c60b1874b70fbcdbc9c8d972c4d9300fb4b0760c5",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def verify_corpus_integrity():
    """Verify CI-tier corpus digests when CHITIN_GATE_FINAL is set."""
    if os.environ.get("CHITIN_GATE_FINAL", "").lower() not in ("1", "true", "yes"):
        return
    traces_dir = Path(__file__).parent / "fixtures" / "traces"
    for name, expected in CI_CORPUS_DIGESTS.items():
        npz = traces_dir / name / "arrays.npz"
        if not npz.exists():
            pytest.fail(f"CHITIN_GATE_FINAL: missing CI corpus fixture {name}")
        actual = _sha256_file(npz)
        if actual != expected:
            pytest.fail(
                f"CHITIN_GATE_FINAL: {name}/arrays.npz digest mismatch. "
                f"Expected {expected[:16]}..., got {actual[:16]}..."
            )


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
def box_hull():
    """Factory for axis-aligned box Hulls with valid triangle faces."""
    from chitin.result import Hull

    def make(center=(0.0, 0.0, 0.0), half=(1.0, 1.0, 1.0)):
        c = np.asarray(center, dtype=np.float32)
        h = np.broadcast_to(np.asarray(half, dtype=np.float32), (3,))
        signs = np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
            dtype=np.float32,
        )
        verts = c + h * signs
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

    return make


@pytest.fixture
def sphere_points():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((500, 3))
    return (pts / np.linalg.norm(pts, axis=1, keepdims=True)).astype(np.float64)


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def _rigged_result_cache(two_bone_rig):
    from chitin import Config, extract_from_rigged_mesh

    return extract_from_rigged_mesh(
        **two_bone_rig,
        config=Config(concavity=0.5),
    )


@pytest.fixture
def rigged_result(_rigged_result_cache):
    """Fresh copy of the shared standard rig extraction."""
    return copy.deepcopy(_rigged_result_cache)


@pytest.fixture(scope="session")
def _rigged_lod_result_cache(two_bone_rig):
    from chitin import Config, extract_from_rigged_mesh

    return extract_from_rigged_mesh(
        **two_bone_rig,
        config=Config(concavity=0.2, lod_concavities=[0.3, 0.7]),
    )


@pytest.fixture
def rigged_lod_result(_rigged_lod_result_cache):
    """Fresh copy of the shared rig extraction with LOD tiers."""
    return copy.deepcopy(_rigged_lod_result_cache)
