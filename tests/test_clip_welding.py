import numpy as np

from chitin.f32_policy import DEFAULT_POLICY
from chitin.f32_predicates import (
    _ear_clip_loop,
    _extract_loops,
    clip_mesh_f32,
    clip_mesh_f64,
    extract_cap_f64,
)


def _tetrahedron():
    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 1.0, 0.0])
    return vertices, faces, plane_point, plane_normal


def test_shared_edge_welding_tetrahedron():
    vertices, faces, plane_point, plane_normal = _tetrahedron()
    result = clip_mesh_f64(vertices, faces, plane_point, plane_normal)

    assert len(result.intersection_points) == 3, (
        f"expected 3 welded intersection points, got {len(result.intersection_points)}"
    )

    pts = result.intersection_points
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert not np.allclose(pts[i], pts[j]), (
                f"intersection points {i} and {j} are duplicates: {pts[i]} vs {pts[j]}"
            )

    assert len(result.boundary_edges) == 3, (
        f"expected 3 boundary edges, got {len(result.boundary_edges)}"
    )

    loops = _extract_loops(result.boundary_edges)
    assert len(loops) == 1, f"expected 1 closed loop, got {len(loops)}"
    assert len(loops[0]) == 3, f"expected loop of length 3, got {len(loops[0])}"


def test_shared_edge_welding_two_triangles():
    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 1.0, 0.0])

    result = clip_mesh_f64(vertices, faces, plane_point, plane_normal)

    assert len(result.intersection_points) == 3, (
        f"expected 3 unique source edges to weld to 3 points, got "
        f"{len(result.intersection_points)}"
    )
    assert len(result.faces) == 2, f"expected 2 output faces, got {len(result.faces)}"

    n_original = len(vertices)
    face_a = {int(v) for v in result.faces[0]}
    face_b = {int(v) for v in result.faces[1]}
    new_a = {v for v in face_a if v >= n_original}
    new_b = {v for v in face_b if v >= n_original}
    shared = new_a & new_b

    assert len(shared) == 1, (
        f"expected exactly one welded vertex shared between the two faces, got {shared}"
    )

    shared_index = next(iter(shared))
    assert np.allclose(result.vertices[shared_index], np.array([0.0, 0.0, 0.0])), (
        f"welded vertex on shared edge (0,1) should sit at the plane origin, "
        f"got {result.vertices[shared_index]}"
    )


def test_cap_winding_consistency():
    vertices, faces, plane_point, plane_normal = _tetrahedron()
    clip_result = clip_mesh_f64(vertices, faces, plane_point, plane_normal)
    cap_result = extract_cap_f64(clip_result)

    assert len(cap_result.cap_faces) > 0, "expected a non-empty cap"
    assert cap_result.winding_consistent, (
        "cap triangulation should have consistent winding"
    )


def test_ear_clip_non_convex_hexagon():
    loop = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    vertices = np.zeros((6, 3), dtype=np.float64)
    vertices[0] = [0, 0, 0]
    vertices[1] = [2, 0, 0]
    vertices[2] = [2, 1, 0]
    vertices[3] = [1, 1, 0]
    vertices[4] = [1, 2, 0]
    vertices[5] = [0, 2, 0]

    faces = _ear_clip_loop(loop, vertices)

    assert len(faces) == 4, f"expected n-2 = 4 triangles, got {len(faces)}"

    reference_normal = np.array([0.0, 0.0, 1.0])
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        assert np.dot(normal, reference_normal) > 0, (
            f"triangle {face} has winding inconsistent with the polygon orientation"
        )


def test_all_positive_no_clip():
    vertices, faces, _, plane_normal = _tetrahedron()
    plane_point = np.array([0.0, -10.0, 0.0])

    result = clip_mesh_f64(vertices, faces, plane_point, plane_normal)

    assert len(result.intersection_points) == 0, "no vertex crosses the plane"
    assert len(result.boundary_edges) == 0, "no vertex crosses the plane"
    assert np.array_equal(result.faces, faces), "all faces should be kept unchanged"


def test_all_negative_no_output():
    vertices, faces, _, plane_normal = _tetrahedron()
    plane_point = np.array([0.0, 10.0, 0.0])

    result = clip_mesh_f64(vertices, faces, plane_point, plane_normal)

    assert len(result.faces) == 0, "every vertex is below the plane, no faces kept"
    assert len(result.intersection_points) == 0, "no vertex crosses the plane"
    assert len(result.boundary_edges) == 0, "no vertex crosses the plane"


def test_f32_f64_welding_parity():
    vertices, faces, plane_point, plane_normal = _tetrahedron()

    result_f64 = clip_mesh_f64(vertices, faces, plane_point, plane_normal)
    result_f32 = clip_mesh_f32(
        vertices, faces, plane_point, plane_normal, DEFAULT_POLICY
    )

    assert len(result_f64.intersection_points) == len(result_f32.intersection_points), (
        "f32 and f64 clip should weld to the same number of intersection points"
    )
    assert len(result_f64.boundary_edges) == len(result_f32.boundary_edges), (
        "f32 and f64 clip should produce the same number of boundary edges"
    )

    loops_f64 = sorted(len(loop) for loop in _extract_loops(result_f64.boundary_edges))
    loops_f32 = sorted(len(loop) for loop in _extract_loops(result_f32.boundary_edges))
    assert loops_f64 == loops_f32, (
        f"loop structure mismatch: f64={loops_f64} f32={loops_f32}"
    )
