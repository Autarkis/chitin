# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from pathlib import Path

import coacd
import numpy as np
import open3d as o3d
import trimesh

from chitin.config import Config
from chitin.result import ExtractionResult, Hull


def extract(
    path: str | Path,
    config: Config | None = None,
) -> ExtractionResult:
    path = Path(path)
    config = config or Config()
    suffix = path.suffix.lower()

    if suffix == ".ply":
        return _extract_from_ply(path, config)
    if suffix in (".obj", ".stl", ".off", ".glb", ".gltf", ".fbx"):
        mesh = trimesh.load(str(path), force="mesh")
        return extract_from_mesh(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            config=config,
        )
    if suffix in (".usd", ".usda", ".usdc"):
        return _extract_from_usd(path, config)

    raise ValueError(f"Unsupported input format: {suffix}")


def extract_from_arrays(
    positions: np.ndarray,
    opacity: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    config: Config | None = None,
) -> ExtractionResult:
    config = config or Config()
    positions = np.asarray(positions, dtype=np.float64)
    source_count = len(positions)

    if opacity is not None:
        raw = np.asarray(opacity, dtype=np.float64).ravel()
        if config.opacity_is_logit:
            activated = 1.0 / (1.0 + np.exp(-raw))
        else:
            activated = raw
        mask = activated >= config.opacity_threshold
        positions = positions[mask]
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64)[mask]

    if len(positions) < 100:
        return ExtractionResult(
            hulls=[], source_vertex_count=source_count, mesh_vertex_count=0
        )

    mesh = _poisson_reconstruct(positions, normals, config)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    if len(triangles) < 4:
        return ExtractionResult(
            hulls=[], source_vertex_count=source_count, mesh_vertex_count=len(vertices)
        )

    return _decompose_and_build(
        vertices, triangles, source_count, len(vertices), config
    )


def extract_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: Config | None = None,
) -> ExtractionResult:
    config = config or Config()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    return _decompose_and_build(vertices, faces, len(vertices), len(vertices), config)


def _extract_from_ply(path: Path, config: Config) -> ExtractionResult:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
        np.float64
    )

    opacity = None
    is_logit = False
    if "opacity" in vertex.data.dtype.names:
        opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        raw_range = opacity.max() - opacity.min()
        is_logit = raw_range > 1.0 or opacity.min() < 0.0

    normals = None
    if all(n in vertex.data.dtype.names for n in ("nx", "ny", "nz")):
        normals = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]]).astype(
            np.float64
        )

    cfg = config
    if is_logit and not config.opacity_is_logit:
        cfg = Config(**{**vars(config), "opacity_is_logit": True})
    return extract_from_arrays(positions, opacity, normals, cfg)


def _extract_from_usd(path: Path, config: Config) -> ExtractionResult:
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ImportError(
            "USD input requires usd-core. Install with: pip install chitin[usd]"
        )

    stage = Usd.Stage.Open(str(path))
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

        if len(points) == 0:
            continue

        tris = []
        idx = 0
        for count in face_counts:
            if count == 3:
                tris.append(face_indices[idx : idx + 3] + vertex_offset)
            elif count > 3:
                for j in range(1, count - 1):
                    tris.append(
                        np.array(
                            [
                                face_indices[idx] + vertex_offset,
                                face_indices[idx + j] + vertex_offset,
                                face_indices[idx + j + 1] + vertex_offset,
                            ],
                            dtype=np.int32,
                        )
                    )
            idx += count

        all_vertices.append(points)
        if tris:
            all_faces.append(np.array(tris, dtype=np.int32))
        vertex_offset += len(points)

    if not all_vertices or not all_faces:
        return ExtractionResult(hulls=[], source_vertex_count=0, mesh_vertex_count=0)

    vertices = np.concatenate(all_vertices)
    faces = np.concatenate(all_faces)
    return extract_from_mesh(vertices, faces, config)


def _poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray | None,
    config: Config,
) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    has_valid_normals = normals is not None and not np.allclose(normals, 0)
    if has_valid_normals:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=config.poisson_depth
    )

    densities = np.asarray(densities)
    if len(densities) == 0:
        return mesh

    density_threshold = np.quantile(densities, 0.1)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def _maybe_decimate(
    vertices: np.ndarray, faces: np.ndarray, max_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) <= max_vertices:
        return vertices, faces
    ratio = max_vertices / len(vertices)
    target_faces = max(4, int(len(faces) * ratio))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int32),
    )


def _decompose_and_build(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    mesh_count: int,
    config: Config,
) -> ExtractionResult:
    vertices, faces = _maybe_decimate(vertices, faces, config.max_decompose_vertices)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    coacd_mesh = coacd.Mesh(tm.vertices, tm.faces)

    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=config.concavity,
        preprocess_mode=config.coacd_preprocess_mode,
        preprocess_resolution=config.coacd_preprocess_resolution,
        max_convex_hull=config.max_hulls,
    )

    hulls = []
    for verts, tris in parts:
        verts = np.asarray(verts, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32).ravel()
        if len(verts) >= config.min_hull_vertices:
            hulls.append(Hull(vertices=verts, indices=tris))

    return ExtractionResult(
        hulls=hulls,
        source_vertex_count=source_count,
        mesh_vertex_count=mesh_count,
    )
