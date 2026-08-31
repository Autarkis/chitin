import numpy as np

from chitin._metric_names import (
    COLLIDER_VOLUME_PRECISION,
    DEEP_FALSE_FILL_FRACTION,
    FALSE_FILL_FRACTION,
    QUALITY_METHOD,
    QUALITY_VOLUME_SAMPLES,
)
from chitin.result import Hull
from chitin.verify.volume import _radical_inverse, volume_report

_BOX_FACES = np.array(
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


def _box_vertices(center=(0.0, 0.0, 0.0), half=1.0) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float64,
    )
    return c + half * signs


def _box_hull(center=(0.0, 0.0, 0.0), half=1.0) -> Hull:
    verts = _box_vertices(center, half).astype(np.float32)
    return Hull(vertices=verts, indices=_BOX_FACES.ravel())


def _icosphere(
    radius: float = 1.0, subdivisions: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """A small, watertight icosphere mesh (vertices, triangle faces)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_faces = []

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key in cache:
                return cache[key]
            nonlocal verts
            m = (verts[i] + verts[j]) / 2.0
            verts = np.vstack([verts, m])
            idx = len(verts) - 1
            cache[key] = idx
            return idx

        for face in faces:
            a, b, c = face
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        faces = np.array(new_faces, dtype=np.int64)

    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return verts, faces


def test_radical_inverse_known_values():
    assert _radical_inverse(1, 2) == 0.5
    assert _radical_inverse(2, 2) == 0.25
    assert _radical_inverse(3, 2) == 0.75
    assert _radical_inverse(4, 2) == 0.125
    assert _radical_inverse(1, 3) == 1.0 / 3.0
    assert _radical_inverse(2, 3) == 2.0 / 3.0
    assert abs(_radical_inverse(3, 3) - 1.0 / 9.0) < 1e-12


def test_unit_cube_hull_matches_source_precisely():
    verts = _box_vertices(half=1.0)
    hull = _box_hull(half=1.0)

    result = volume_report(
        [hull],
        verts,
        _BOX_FACES,
        volume_samples=2048,
    )

    assert result.collider_volume_samples > 0
    assert result.collider_volume_precision is not None
    assert result.collider_volume_precision > 0.98
    assert result.false_fill_fraction is not None
    assert result.false_fill_fraction < 0.02
    assert result.deep_false_fill_fraction is not None
    assert result.deep_false_fill_fraction == 0.0


def test_oversized_hull_around_sphere_has_significant_false_fill():
    verts, faces = _icosphere(radius=1.0, subdivisions=2)
    # A cube hull that circumscribes the sphere leaves its corners empty of
    # source volume -- a canonical false-fill case.
    hull = _box_hull(half=1.0)

    result = volume_report(
        [hull],
        verts,
        faces,
        volume_samples=4096,
    )

    assert result.collider_volume_precision is not None
    # Sphere volume / circumscribing-cube volume = pi/6 =~ 0.524, so most of
    # the collider is false fill.
    assert result.false_fill_fraction is not None
    assert result.false_fill_fraction > 0.3
    assert result.deep_false_fill_fraction is not None
    assert result.deep_false_fill_fraction > 0.0


def test_too_few_collider_samples_returns_none():
    verts, faces = _icosphere(radius=1.0, subdivisions=1)
    hull = _box_hull(half=1.0)

    # collider_volume_samples can never exceed volume_samples, so requesting
    # fewer samples than min_collider_samples guarantees no signal.
    result = volume_report(
        [hull],
        verts,
        faces,
        volume_samples=20,
        min_collider_samples=32,
    )

    assert result.collider_volume_samples < 32
    assert result.collider_volume_precision is None
    assert result.false_fill_fraction is None
    assert result.deep_false_fill_fraction is None


def test_to_coverage_dict_shape():
    verts = _box_vertices(half=1.0)
    hull = _box_hull(half=1.0)

    result = volume_report([hull], verts, _BOX_FACES, volume_samples=256)
    d = result.to_coverage_dict()

    assert set(d.keys()) == {
        COLLIDER_VOLUME_PRECISION,
        FALSE_FILL_FRACTION,
        DEEP_FALSE_FILL_FRACTION,
        QUALITY_METHOD,
        QUALITY_VOLUME_SAMPLES,
    }
    assert d[QUALITY_METHOD] == "deterministic_halton_v1"
    assert d[QUALITY_VOLUME_SAMPLES] == 256
