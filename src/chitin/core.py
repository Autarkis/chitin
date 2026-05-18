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
    if suffix in (".obj", ".stl", ".off"):
        mesh = trimesh.load(str(path), force="mesh")
        return extract_from_mesh(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            config=config,
        )

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

    mesh = _poisson_reconstruct(positions, normals, config)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

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


def _poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray | None,
    config: Config,
) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=config.poisson_depth
    )

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.1)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def _decompose_and_build(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    mesh_count: int,
    config: Config,
) -> ExtractionResult:
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    coacd_mesh = coacd.Mesh(tm.vertices, tm.faces)

    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=config.concavity,
        preprocess=config.coacd_preprocess,
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
