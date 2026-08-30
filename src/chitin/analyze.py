from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ENV_INNER_DENSITY_MAX = 0.05
_ENV_AMBIGUOUS_MAX = 0.20
_MIN_ENV_POINTS = 1000
_MIN_ENV_VOLUME = 10.0

# Wall and floor detection. A slab this deep is taken off an AABB face; its
# points count as a plane when most of them hug one depth and that thin band
# covers enough of the face.
_SLAB_FRACTION = 0.1
_BAND_FRACTION = 0.25
_BAND_CONCENTRATION = 0.6
_FACE_GRID = 16
_FACE_COVERAGE_MIN = 0.35
_MIN_FACE_POINTS = 200


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
    wall_faces: int = 0
    floor_coverage: float = 0.0
    is_environment_ambiguous: bool = False

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "has_opacity": self.has_opacity,
            "has_covariance": self.has_covariance,
            "is_environment_likely": self.is_environment_likely,
            "is_environment_ambiguous": self.is_environment_ambiguous,
            "is_skinned": self.is_skinned,
            "is_manifold": self.is_manifold,
            "point_count": self.point_count,
            "face_count": self.face_count,
            "opacity_is_logit": self.opacity_is_logit,
            "bbox_volume": self.bbox_volume,
            "inner_density_ratio": self.inner_density_ratio,
            "wall_faces": self.wall_faces,
            "floor_coverage": self.floor_coverage,
        }


@dataclass(frozen=True)
class EnvironmentSignals:
    inner_density_ratio: float
    bbox_volume: float
    wall_faces: int
    floor_coverage: float
    is_environment_likely: bool
    is_environment_ambiguous: bool


_UNMEASURED_INNER_DENSITY = float("nan")

_EMPTY_SIGNALS = EnvironmentSignals(
    _UNMEASURED_INNER_DENSITY, 0.0, 0, 0.0, False, False
)


def _compute_inner_density(positions: np.ndarray) -> tuple[float, float]:
    if len(positions) == 0:
        return _UNMEASURED_INNER_DENSITY, 0.0

    scene_min = positions.min(axis=0)
    scene_max = positions.max(axis=0)
    extent = scene_max - scene_min
    vol = float(np.prod(np.where(extent == 0, 1.0, extent)))

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


def _face_coverage(
    positions: np.ndarray,
    normal_axis: int,
    low_face: bool,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
) -> float:
    """How much of one AABB face is covered by a plane of points against it.

    Returns the occupied fraction of a grid over the face, or 0.0 when the slab
    taken off that face is not planar -- a solid volume fills its slab evenly
    instead of hugging one depth, which is what separates a wall from a block.
    """
    extent = scene_max - scene_min
    depth = float(extent[normal_axis]) * _SLAB_FRACTION
    if depth <= 0:
        return 0.0

    coord = positions[:, normal_axis]
    if low_face:
        slab = positions[coord <= scene_min[normal_axis] + depth]
    else:
        slab = positions[coord >= scene_max[normal_axis] - depth]
    if len(slab) < _MIN_FACE_POINTS:
        return 0.0

    offset = np.abs(slab[:, normal_axis] - np.median(slab[:, normal_axis]))
    band = offset <= depth * _BAND_FRACTION
    if band.sum() / len(slab) < _BAND_CONCENTRATION:
        return 0.0

    axes = [i for i in range(3) if i != normal_axis]
    spans = np.where(extent[axes] == 0, 1.0, extent[axes])
    cells = (slab[band][:, axes] - scene_min[axes]) / spans * _FACE_GRID
    cells = np.clip(cells.astype(int), 0, _FACE_GRID - 1)
    occupied = len(np.unique(cells[:, 0] * _FACE_GRID + cells[:, 1]))
    return occupied / (_FACE_GRID * _FACE_GRID)


def _shell_signature(positions: np.ndarray) -> tuple[int, float]:
    """Count wall planes against the vertical AABB faces, and cover the floor.

    A room keeps its walls and floor however cluttered the middle gets, which is
    exactly what inner-AABB density alone stops seeing.
    """
    scene_min = positions.min(axis=0)
    scene_max = positions.max(axis=0)

    walls = 0
    for axis in (0, 2):
        for low_face in (True, False):
            coverage = _face_coverage(positions, axis, low_face, scene_min, scene_max)
            if coverage >= _FACE_COVERAGE_MIN:
                walls += 1

    floor_coverage = _face_coverage(positions, 1, True, scene_min, scene_max)
    return walls, floor_coverage


def _environment_signals(positions: np.ndarray) -> EnvironmentSignals:
    """Decide whether an input is a scanned environment rather than an object.

    A hollow middle still counts on its own, for shells with no axis-aligned
    walls. Otherwise the shell signature carries it, so a room with a pillar or
    a mid-floor row of shelves is not read as a solid object and filled in.
    """
    inner_ratio, bbox_volume = _compute_inner_density(positions)
    if len(positions) < _MIN_ENV_POINTS or bbox_volume < _MIN_ENV_VOLUME:
        return EnvironmentSignals(inner_ratio, bbox_volume, 0, 0.0, False, False)

    walls, floor_coverage = _shell_signature(positions)
    hollow = inner_ratio < _ENV_INNER_DENSITY_MAX
    enclosed = walls >= 2 and floor_coverage >= _FACE_COVERAGE_MIN
    is_environment = hollow or enclosed

    return EnvironmentSignals(
        inner_density_ratio=inner_ratio,
        bbox_volume=bbox_volume,
        wall_faces=walls,
        floor_coverage=floor_coverage,
        is_environment_likely=is_environment,
        is_environment_ambiguous=(
            not is_environment
            and _ENV_INNER_DENSITY_MAX <= inner_ratio < _ENV_AMBIGUOUS_MAX
        ),
    )


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
    env = _environment_signals(positions)

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
        is_environment_likely=env.is_environment_likely,
        is_environment_ambiguous=env.is_environment_ambiguous,
        is_skinned=is_skinned,
        is_manifold=is_manifold,
        point_count=len(positions),
        face_count=face_count,
        opacity_is_logit=opacity_is_logit,
        bbox_volume=env.bbox_volume,
        inner_density_ratio=env.inner_density_ratio,
        wall_faces=env.wall_faces,
        floor_coverage=env.floor_coverage,
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
    from chitin.adapters.ply_reader import read_ply_mesh

    vertex, faces = read_ply_mesh(path)
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

    env = _environment_signals(positions)

    return InputAnalysis(
        format="ply",
        has_opacity=has_opacity,
        has_covariance=has_scales and has_rots,
        is_environment_likely=env.is_environment_likely,
        is_environment_ambiguous=env.is_environment_ambiguous,
        is_skinned=False,
        is_manifold=None,
        point_count=len(positions),
        face_count=len(faces) if faces is not None else None,
        opacity_is_logit=opacity_is_logit,
        bbox_volume=env.bbox_volume,
        inner_density_ratio=env.inner_density_ratio,
        wall_faces=env.wall_faces,
        floor_coverage=env.floor_coverage,
    )


def _analyze_mesh(path: Path) -> InputAnalysis:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    env = _environment_signals(positions)

    return InputAnalysis(
        format=path.suffix.lower().lstrip("."),
        has_opacity=False,
        has_covariance=False,
        is_environment_likely=env.is_environment_likely,
        is_environment_ambiguous=env.is_environment_ambiguous,
        is_skinned=False,
        is_manifold=bool(mesh.is_watertight),
        point_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        opacity_is_logit=False,
        bbox_volume=env.bbox_volume,
        inner_density_ratio=env.inner_density_ratio,
        wall_faces=env.wall_faces,
        floor_coverage=env.floor_coverage,
    )


def _analyze_gltf(path: Path) -> InputAnalysis:
    import trimesh

    from chitin.gltf_skin import parse_skin

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
        env = _environment_signals(positions)
        return InputAnalysis(
            format=path.suffix.lower().lstrip("."),
            has_opacity=False,
            has_covariance=False,
            is_environment_likely=env.is_environment_likely,
            is_environment_ambiguous=env.is_environment_ambiguous,
            is_skinned=is_skinned,
            is_manifold=bool(mesh.is_watertight),
            point_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            opacity_is_logit=False,
            bbox_volume=env.bbox_volume,
            inner_density_ratio=env.inner_density_ratio,
            wall_faces=env.wall_faces,
            floor_coverage=env.floor_coverage,
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
        inner_density_ratio=_UNMEASURED_INNER_DENSITY,
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
        env = _environment_signals(positions)
    else:
        env = _EMPTY_SIGNALS

    return InputAnalysis(
        format=path.suffix.lower().lstrip("."),
        has_opacity=False,
        has_covariance=False,
        is_environment_likely=env.is_environment_likely,
        is_environment_ambiguous=env.is_environment_ambiguous,
        is_skinned=False,
        is_manifold=None,
        point_count=total_points,
        face_count=total_faces,
        opacity_is_logit=False,
        bbox_volume=env.bbox_volume,
        inner_density_ratio=env.inner_density_ratio,
        wall_faces=env.wall_faces,
        floor_coverage=env.floor_coverage,
    )
