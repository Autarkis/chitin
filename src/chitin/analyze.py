# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InputAnalysis:
    format: str
    has_opacity: bool
    has_covariance: bool
    is_environment_likely: bool
    is_skinned: bool
    is_manifold: bool | None
    point_count: int
    face_count: int | None
    opacity_is_logit: bool
    bbox_volume: float
    inner_density_ratio: float

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "has_opacity": self.has_opacity,
            "has_covariance": self.has_covariance,
            "is_environment_likely": self.is_environment_likely,
            "is_skinned": self.is_skinned,
            "is_manifold": self.is_manifold,
            "point_count": self.point_count,
            "face_count": self.face_count,
            "opacity_is_logit": self.opacity_is_logit,
            "bbox_volume": self.bbox_volume,
            "inner_density_ratio": self.inner_density_ratio,
        }


def _compute_inner_density(positions: np.ndarray) -> tuple[float, float]:
    if len(positions) < 1000:
        return 1.0, 0.0

    scene_min = positions.min(axis=0)
    scene_max = positions.max(axis=0)
    extent = scene_max - scene_min
    vol = float(np.prod(np.where(extent == 0, 1.0, extent)))
    if vol < 10.0:
        return 1.0, vol

    center = (scene_min + scene_max) / 2
    inner_extent = extent * 0.5
    inner_min = center - inner_extent / 2
    inner_max = center + inner_extent / 2

    mask = (
        (positions[:, 0] >= inner_min[0])
        & (positions[:, 0] <= inner_max[0])
        & (positions[:, 1] >= inner_min[1])
        & (positions[:, 1] <= inner_max[1])
        & (positions[:, 2] >= inner_min[2])
        & (positions[:, 2] <= inner_max[2])
    )
    ratio = float(mask.sum() / len(positions))
    return ratio, vol


def analyze_arrays(
    positions: np.ndarray,
    opacity: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    rots: np.ndarray | None = None,
    *,
    format: str = "arrays",
    face_count: int | None = None,
    is_skinned: bool = False,
    is_manifold: bool | None = None,
) -> InputAnalysis:
    positions = np.asarray(positions, dtype=np.float64)
    has_covariance = scales is not None and rots is not None
    inner_ratio, bbox_vol = _compute_inner_density(positions)

    has_opacity = opacity is not None
    opacity_is_logit = False
    if has_opacity:
        raw = np.asarray(opacity, dtype=np.float64).ravel()
        raw_range = raw.max() - raw.min()
        opacity_is_logit = bool(raw_range > 1.0 or raw.min() < 0.0)

    return InputAnalysis(
        format=format,
        has_opacity=has_opacity,
        has_covariance=has_covariance,
        is_environment_likely=inner_ratio < 0.05,
        is_skinned=is_skinned,
        is_manifold=is_manifold,
        point_count=len(positions),
        face_count=face_count,
        opacity_is_logit=opacity_is_logit,
        bbox_volume=bbox_vol,
        inner_density_ratio=inner_ratio,
    )


def analyze_input(path: str | Path) -> InputAnalysis:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".ply":
        return _analyze_ply(path)
    if suffix in (".obj", ".stl", ".off"):
        return _analyze_mesh(path)
    if suffix in (".glb", ".gltf", ".fbx"):
        return _analyze_gltf(path)
    if suffix in (".usd", ".usda", ".usdc"):
        return _analyze_usd(path)

    raise ValueError(f"Unsupported input format: {suffix}")


def _analyze_ply(path: Path) -> InputAnalysis:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
        np.float64
    )

    has_opacity = "opacity" in vertex.data.dtype.names
    opacity_is_logit = False
    if has_opacity:
        opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        raw_range = opacity.max() - opacity.min()
        opacity_is_logit = bool(raw_range > 1.0 or opacity.min() < 0.0)

    has_scales = all(f"scale_{i}" in vertex.data.dtype.names for i in range(3))
    has_rots = all(f"rot_{i}" in vertex.data.dtype.names for i in range(4))

    inner_ratio, bbox_vol = _compute_inner_density(positions)

    return InputAnalysis(
        format="ply",
        has_opacity=has_opacity,
        has_covariance=has_scales and has_rots,
        is_environment_likely=inner_ratio < 0.05,
        is_skinned=False,
        is_manifold=None,
        point_count=len(positions),
        face_count=None,
        opacity_is_logit=opacity_is_logit,
        bbox_volume=bbox_vol,
        inner_density_ratio=inner_ratio,
    )


def _analyze_mesh(path: Path) -> InputAnalysis:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    inner_ratio, bbox_vol = _compute_inner_density(positions)

    return InputAnalysis(
        format=path.suffix.lower().lstrip("."),
        has_opacity=False,
        has_covariance=False,
        is_environment_likely=inner_ratio < 0.05,
        is_skinned=False,
        is_manifold=bool(mesh.is_watertight),
        point_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        opacity_is_logit=False,
        bbox_volume=bbox_vol,
        inner_density_ratio=inner_ratio,
    )


def _analyze_gltf(path: Path) -> InputAnalysis:
    from chitin.gltf_skin import parse_skin

    import trimesh

    skin_data = parse_skin(path)
    loaded = trimesh.load(str(path))

    is_skinned = (
        skin_data is not None
        and skin_data.joint_indices is not None
        and skin_data.joint_weights is not None
    )

    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded

    if isinstance(mesh, trimesh.Trimesh):
        positions = np.asarray(mesh.vertices, dtype=np.float64)
        inner_ratio, bbox_vol = _compute_inner_density(positions)
        return InputAnalysis(
            format=path.suffix.lower().lstrip("."),
            has_opacity=False,
            has_covariance=False,
            is_environment_likely=inner_ratio < 0.05,
            is_skinned=is_skinned,
            is_manifold=bool(mesh.is_watertight),
            point_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            opacity_is_logit=False,
            bbox_volume=bbox_vol,
            inner_density_ratio=inner_ratio,
        )

    return InputAnalysis(
        format=path.suffix.lower().lstrip("."),
        has_opacity=False,
        has_covariance=False,
        is_environment_likely=False,
        is_skinned=is_skinned,
        is_manifold=None,
        point_count=0,
        face_count=0,
        opacity_is_logit=False,
        bbox_volume=0.0,
        inner_density_ratio=1.0,
    )


def _analyze_usd(path: Path) -> InputAnalysis:
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ImportError(
            "USD input requires usd-core. Install with: pip install chitin[usd]"
        )

    stage = Usd.Stage.Open(str(path))
    total_points = 0
    total_faces = 0
    all_positions = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points and len(points) > 0:
            total_points += len(points)
            all_positions.append(np.array(points, dtype=np.float64))
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        if face_counts:
            total_faces += len(face_counts)

    if all_positions:
        positions = np.concatenate(all_positions)
        inner_ratio, bbox_vol = _compute_inner_density(positions)
    else:
        inner_ratio, bbox_vol = 1.0, 0.0

    return InputAnalysis(
        format=path.suffix.lower().lstrip("."),
        has_opacity=False,
        has_covariance=False,
        is_environment_likely=inner_ratio < 0.05,
        is_skinned=False,
        is_manifold=None,
        point_count=total_points,
        face_count=total_faces,
        opacity_is_logit=False,
        bbox_volume=bbox_vol,
        inner_density_ratio=inner_ratio,
    )
