from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from chitin.config import DEFAULT_COACD_TIMEOUT_SECONDS
from chitin.errors import CompilationError
from chitin.plan import BuildPlan
from chitin.resolve import ResolvedConfig
from chitin.result import ExtractionResult, Hull, LodHulls
from chitin.stages.flatness import make_planar_box
from chitin.verify.convex import outward_face_planes as _outward_face_planes
from chitin.verify.convex import points_inside as _per_vertex_inside

_COACD_WORKER_SCRIPT = Path(__file__).parent / "_coacd_worker.py"
# Environment that pins CoACD's native thread pool to one thread. CoACD's search
# is OpenMP-parallel and its scheduling decides which decomposition it settles
# on, so an unpinned run returns a different hull count and different bytes each
# time -- the same failure Poisson had (see the single-threaded note in
# docs/architecture.md). Set on the worker subprocess rather than the parent so
# only the decomposition is pinned.
_DETERMINISTIC_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
# A non-watertight mesh forces CoACD's voxel-remesh; at high resolution that
# explodes the triangle count (~150k+) and stalls MCTS. Capping the resolution
# keeps the remesh small enough to decompose. Verified: open patches complete
# in ~1s at res 6-8 but hang at >=10.
NONWATERTIGHT_MAX_RESOLUTION = 8


def is_closed_surface(faces: np.ndarray) -> bool:
    """Cheap watertight proxy: is every edge shared by exactly two triangles?

    Pure numpy so it stays out of open3d. A closed mesh lets CoACD skip its
    explosive voxel-remesh; an open one forces it.
    """
    if len(faces) == 0:
        return False
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def aabb(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of ``vertices`` as (min_corner, max_corner)."""
    return vertices.min(axis=0), vertices.max(axis=0)


def split_mesh_components(
    vertices: np.ndarray, faces: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a triangle mesh into compact vertex-connected components.

    Source formats commonly concatenate several solids into one vertex/face
    array. Passing that whole array to CoACD is both wasteful and, when the
    solids overlap, pathological: the combined triangle soup is not one valid
    solid even though every individual component is watertight. Decomposing the
    components independently preserves the same collision union and avoids
    making CoACD discover boundaries the source already provides.

    The implementation is pure NumPy/Python so base installs do not acquire a
    SciPy dependency through trimesh's graph helpers. Components retain source
    face order; vertices are compacted and faces reindexed per component.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if len(faces) == 0:
        return []

    parent = np.arange(len(vertices), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for a, b, c in faces:
        union(int(a), int(b))
        union(int(a), int(c))

    grouped: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        grouped.setdefault(find(int(face[0])), []).append(face_index)

    components: list[tuple[np.ndarray, np.ndarray]] = []
    for face_indices in grouped.values():
        component_faces = faces[np.asarray(face_indices, dtype=np.int64)]
        used, inverse = np.unique(component_faces.ravel(), return_inverse=True)
        components.append(
            (
                vertices[used],
                inverse.reshape((-1, 3)).astype(np.int32, copy=False),
            )
        )
    return components


class CoACDTimeoutError(RuntimeError):
    """CoACD exceeded its time budget or the worker crashed."""


def run_coacd_bounded(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    threshold: float,
    preprocess_mode: str,
    preprocess_resolution: int,
    max_convex_hull: int,
    timeout: float = DEFAULT_COACD_TIMEOUT_SECONDS,
    deterministic: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run CoACD in a subprocess so a native stall can be hard-killed on timeout.

    Returns a list of (vertices, triangles) hulls, or raises CoACDTimeoutError
    when CoACD does not finish in ``timeout`` seconds (or the worker crashes) --
    e.g. the manifold-remesh explosion on non-watertight input.

    With ``deterministic`` (the default) the worker runs single-threaded, so the
    same mesh and config always produce the same hulls; it costs 2-4x the wall
    time on concave assets. Turning it off restores CoACD's own parallelism and
    with it a decomposition that varies run to run.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.npz"
        out_path = Path(tmpdir) / "out.npz"
        np.savez(
            in_path,
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int32),
            threshold=np.array([threshold], dtype=np.float64),
            preprocess_mode=np.array([preprocess_mode]),
            preprocess_resolution=np.array([preprocess_resolution], dtype=np.int64),
            max_convex_hull=np.array([max_convex_hull], dtype=np.int64),
        )
        env = None
        if deterministic:
            env = {**os.environ, **_DETERMINISTIC_ENV}
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(_COACD_WORKER_SCRIPT),
                    str(in_path),
                    str(out_path),
                ],
                capture_output=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CoACDTimeoutError(f"coacd timeout after {timeout}s") from exc
        if result.returncode != 0 or not out_path.exists():
            raise CoACDTimeoutError(f"coacd worker exit {result.returncode}")
        with np.load(out_path) as data:
            n = int(data["n"][0])
            return [(data[f"v{i}"], data[f"t{i}"]) for i in range(n)]


logger = logging.getLogger("chitin")


# CoACD's manifold preprocessing voxel-remeshes non-watertight input on a
# `resolution`^3 grid, emitting ~O(resolution^2) surface triangles regardless of
# how simple the source is. For low-poly catalog assets (a few hundred faces) the
# default resolution of 50 inflates the mesh ~150x (a 676-face panel becomes
# ~100k triangles), and the MCTS decomposition then crawls on the bloated mesh.
# Scale the grid to input complexity: catalog assets get a light grid; dense
# scans keep the full configured resolution (where it is genuinely warranted).
ADAPTIVE_MIN_RESOLUTION = 30
ADAPTIVE_LOW_FACES = 1_000
ADAPTIVE_HIGH_FACES = 50_000


def adaptive_preprocess_resolution(face_count: int, configured: int) -> int:
    """Pick a preprocess resolution scaled to mesh complexity, capped at the
    configured value (never coarser than ``ADAPTIVE_MIN_RESOLUTION``).

    Ramps linearly from the floor at ``ADAPTIVE_LOW_FACES`` to the configured
    resolution at ``ADAPTIVE_HIGH_FACES``, so simple meshes are cheap and dense
    meshes are untouched.
    """
    floor = min(ADAPTIVE_MIN_RESOLUTION, configured)
    if face_count <= ADAPTIVE_LOW_FACES:
        return floor
    if face_count >= ADAPTIVE_HIGH_FACES:
        return configured
    span = ADAPTIVE_HIGH_FACES - ADAPTIVE_LOW_FACES
    frac = (face_count - ADAPTIVE_LOW_FACES) / span
    return round(floor + (configured - floor) * frac)


def aabb_overlaps_bounds(
    hull: Hull, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> bool:
    h_min, h_max = aabb(hull.vertices)
    return bool(
        h_max[0] >= bounds_min[0]
        and h_min[0] <= bounds_max[0]
        and h_max[1] >= bounds_min[1]
        and h_min[1] <= bounds_max[1]
        and h_max[2] >= bounds_min[2]
        and h_min[2] <= bounds_max[2]
    )


def aabb_iou(a: Hull, b: Hull) -> float:
    a_min, a_max = aabb(a.vertices)
    b_min, b_max = aabb(b.vertices)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])
    a_vol = float(np.prod(a_max - a_min))
    b_vol = float(np.prod(b_max - b_min))
    union_vol = a_vol + b_vol - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


def dedup_overlapping_hulls(
    hulls: list[Hull], iou_threshold: float = 0.5
) -> list[Hull]:
    if len(hulls) <= 1:
        return hulls
    order = sorted(
        range(len(hulls)), key=lambda i: len(hulls[i].vertices), reverse=True
    )
    discarded: set[int] = set()
    kept: list[Hull] = []
    for pos, i in enumerate(order):
        if i in discarded:
            continue
        kept.append(hulls[i])
        for j in order[pos + 1 :]:
            if j in discarded:
                continue
            if aabb_iou(hulls[i], hulls[j]) >= iou_threshold:
                discarded.add(j)
    return kept


def _aabb_overlaps(a_min, a_max, b_min, b_max) -> bool:
    return bool(np.all(a_min <= b_max + 1e-6) and np.all(b_min <= a_max + 1e-6))


def cull_contained_hulls(hulls: list[Hull]) -> list[Hull]:
    if len(hulls) <= 1:
        return hulls
    volumes = []
    for h in hulls:
        mn, mx = aabb(h.vertices)
        volumes.append(float(np.prod(np.maximum(mx - mn, 1e-10))))
    order = sorted(range(len(hulls)), key=lambda i: volumes[i], reverse=True)
    contained: set[int] = set()
    for pos, i in enumerate(order):
        if i in contained:
            continue
        normals, d = _outward_face_planes(hulls[i])
        for j in order[pos + 1 :]:
            if j in contained:
                continue
            inside = _per_vertex_inside(normals, d, hulls[j].vertices)
            if np.all(inside):
                contained.add(j)
    return [h for idx, h in enumerate(hulls) if idx not in contained]


def consolidate_near_contained_hulls(
    hulls: list[Hull],
    volume_pct_threshold: float = 0.005,
    containment_ratio: float = 0.7,
    shallow_protrusion: float = 0.15,
    moderate_protrusion: float = 0.30,
) -> list[Hull]:
    if len(hulls) <= 1:
        return hulls

    aabb_data = []
    volumes = []
    for h in hulls:
        mn, mx = aabb(h.vertices)
        aabb_data.append((mn, mx))
        volumes.append(float(np.prod(np.maximum(mx - mn, 1e-10))))

    scene_min = np.min([a[0] for a in aabb_data], axis=0)
    scene_max = np.max([a[1] for a in aabb_data], axis=0)
    scene_vol = float(np.prod(scene_max - scene_min))
    if scene_vol <= 0:
        return hulls

    vol_threshold = volume_pct_threshold * scene_vol
    order = sorted(range(len(hulls)), key=lambda i: volumes[i], reverse=True)

    planes_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    absorbed: set[int] = set()

    for small_idx in order:
        if small_idx in absorbed or volumes[small_idx] >= vol_threshold:
            continue
        s_mn, s_mx = aabb_data[small_idx]
        small_extent = float(np.linalg.norm(s_mx - s_mn))
        if small_extent < 1e-10:
            absorbed.add(small_idx)
            continue

        for large_idx in order:
            if large_idx == small_idx or large_idx in absorbed:
                continue
            if volumes[large_idx] <= volumes[small_idx]:
                break

            l_mn, l_mx = aabb_data[large_idx]
            if not _aabb_overlaps(s_mn, s_mx, l_mn, l_mx):
                continue

            if large_idx not in planes_cache:
                planes_cache[large_idx] = _outward_face_planes(hulls[large_idx])
            normals, d = planes_cache[large_idx]

            small_verts = hulls[small_idx].vertices.astype(np.float64)
            inside = _per_vertex_inside(normals, d, small_verts)
            frac = inside.sum() / len(inside)
            if frac < containment_ratio:
                continue

            outside_pts = small_verts[~inside]
            if len(outside_pts) == 0:
                absorbed.add(small_idx)
                break

            violations = outside_pts @ normals.T - d[np.newaxis, :]
            max_protrusion = float(violations.max())
            prot_ratio = max_protrusion / small_extent

            if prot_ratio < shallow_protrusion:
                absorbed.add(small_idx)
                break

            if prot_ratio < moderate_protrusion:
                large_centroid = hulls[large_idx].vertices.mean(axis=0)
                small_centroid = hulls[small_idx].vertices.mean(axis=0)
                sep = small_centroid - large_centroid
                sep_len = float(np.linalg.norm(sep))
                if sep_len < 1e-10:
                    absorbed.add(small_idx)
                    break
                sep_norm = sep / sep_len
                max_face = int(violations.max(axis=0).argmax())
                alignment = abs(float(np.dot(normals[max_face], sep_norm)))
                if alignment < 0.7:
                    absorbed.add(small_idx)
                    break

    return [h for idx, h in enumerate(hulls) if idx not in absorbed]


def maybe_decimate(
    vertices: np.ndarray, faces: np.ndarray, max_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) <= max_vertices:
        return vertices, faces
    try:
        import open3d as o3d
    except ImportError:
        logger.warning(
            "mesh has %d vertices (over max_decompose_vertices=%d) but decimation "
            "was skipped because Open3D is not installed; the full mesh is passed "
            "to CoACD. Install the 'splat' extra (pip install chitin[splat]) to "
            "enable large-mesh decimation.",
            len(vertices),
            max_vertices,
        )
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


def extract_walkable_hulls(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[list[Hull], np.ndarray]:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(face_normals, axis=1, keepdims=True)
    areas = np.where(areas == 0, 1e-12, areas)
    face_normals /= areas

    hist = np.abs(face_normals).sum(axis=0)
    up_axis_idx = np.argmax(hist)
    up_axis = np.zeros(3)
    up_axis[up_axis_idx] = 1.0
    if (face_normals @ up_axis).sum() < 0:
        up_axis = -up_axis

    dot = face_normals @ up_axis
    walkable_mask = dot >= np.cos(np.radians(35.0))

    if not np.any(walkable_mask):
        return [], faces

    walkable_faces = faces[walkable_mask]
    unwalkable_faces = faces[~walkable_mask]

    tm = trimesh.Trimesh(vertices=vertices, faces=walkable_faces, process=False)
    try:
        components = trimesh.graph.connected_components(tm.face_adjacency)
    except Exception:  # noqa: BLE001
        return [], faces

    hulls = []
    keep_unwalkable = [unwalkable_faces]

    for comp in components:
        if len(comp) < 200:
            keep_unwalkable.append(walkable_faces[comp])
            continue

        comp_faces = walkable_faces[comp]
        comp_verts = vertices[np.unique(comp_faces)]

        cv0, cv1, cv2 = (
            vertices[comp_faces[:, 0]],
            vertices[comp_faces[:, 1]],
            vertices[comp_faces[:, 2]],
        )
        comp_normals = np.cross(cv1 - cv0, cv2 - cv0)
        avg_normal = comp_normals.mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 0:
            avg_normal /= norm
        else:
            avg_normal = up_axis

        box_hull = make_planar_box(comp_verts, avg_normal)
        hulls.append(box_hull)

    final_unwalkable = (
        np.vstack(keep_unwalkable)
        if keep_unwalkable
        else np.array([], dtype=np.int32).reshape(0, 3)
    )
    return hulls, final_unwalkable


def decompose_and_build(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    mesh_count: int,
    config: ResolvedConfig,
    _plan: BuildPlan | None = None,
) -> ExtractionResult:
    pre_decimate_count = len(vertices)
    vertices, faces = maybe_decimate(vertices, faces, config.max_decompose_vertices)

    if _plan is not None:
        if len(vertices) < pre_decimate_count:
            _plan.decimated = True
            _plan.step("decimate")
            _plan.detected["decimated_from"] = pre_decimate_count
            _plan.detected["decimated_to"] = len(vertices)
        elif pre_decimate_count > config.max_decompose_vertices:
            # Over the threshold but unchanged => decimation was skipped (Open3D
            # absent). Surface it so the caller knows the full mesh hit CoACD.
            _plan.detected["decimation_skipped"] = pre_decimate_count
        _plan.step("decompose")
        _plan.processed_vertices = _plan.processed_vertices or len(vertices)

    is_env = _plan is not None and _plan.detected.get("is_environment", False)
    env_hulls = []
    if is_env:
        env_hulls, faces = extract_walkable_hulls(vertices, faces)
        if env_hulls and _plan is not None:
            _plan.detected["walkable_hulls"] = len(env_hulls)
            _plan.detected["planar_substitute_hulls"] = _plan.detected.get(
                "planar_substitute_hulls", 0
            ) + len(env_hulls)

    if len(faces) < 4:
        return ExtractionResult(
            hulls=env_hulls,
            source_vertex_count=source_count,
            mesh_vertex_count=mesh_count,
            build_plan=_plan,
            lod_tiers=None,
        )

    tm = trimesh.Trimesh(vertices=vertices, faces=faces)

    preprocess_resolution = config.coacd_preprocess_resolution
    if config.coacd_adaptive_preprocess and config.coacd_preprocess_mode != "off":
        preprocess_resolution = adaptive_preprocess_resolution(
            len(faces), config.coacd_preprocess_resolution
        )
        if (
            _plan is not None
            and preprocess_resolution != config.coacd_preprocess_resolution
        ):
            _plan.detected["preprocess_resolution"] = preprocess_resolution

    closed = is_closed_surface(faces)
    if (
        config.coacd_preprocess_mode != "off"
        and not closed
        and preprocess_resolution > NONWATERTIGHT_MAX_RESOLUTION
    ):
        preprocess_resolution = NONWATERTIGHT_MAX_RESOLUTION
        if _plan is not None:
            _plan.detected["nonwatertight_preprocess_cap"] = preprocess_resolution

    if _plan is not None:
        # Stamped where the decomposition actually happens, so a build that
        # never reached CoACD (a planar box, an all-environment shell) carries
        # no claim either way. The acceptance gate reads this to decide whether
        # these hulls can be reproduced.
        _plan.detected["coacd_deterministic"] = bool(config.coacd_deterministic)

    def _decompose(threshold: float) -> list[Hull]:
        parts = run_coacd_bounded(
            tm.vertices,
            tm.faces,
            threshold=threshold,
            preprocess_mode=config.coacd_preprocess_mode,
            preprocess_resolution=preprocess_resolution,
            max_convex_hull=config.max_hulls,
            timeout=config.coacd_timeout,
            deterministic=config.coacd_deterministic,
        )
        out = env_hulls.copy()
        for verts, tris in parts:
            verts = np.asarray(verts, dtype=np.float32)
            tris = np.asarray(tris, dtype=np.uint32).ravel()
            if len(verts) >= config.min_hull_vertices:
                out.append(Hull(vertices=verts, indices=tris))
        return out

    try:
        hulls = _decompose(config.concavity)

        lod_tiers = None
        if config.lod_concavities:
            lod_tiers = [
                LodHulls(concavity=c, hulls=_decompose(c))
                for c in sorted(config.lod_concavities)
            ]
    except CoACDTimeoutError as exc:
        raise CompilationError(
            code="COACD_TIMEOUT",
            evidence={
                "source_vertex_count": source_count,
                "mesh_vertex_count": mesh_count,
                "timeout_seconds": config.coacd_timeout,
                "preprocess_mode": config.coacd_preprocess_mode,
                "preprocess_resolution": preprocess_resolution,
                "concavity": config.concavity,
                "deterministic": config.coacd_deterministic,
            },
            message=(
                f"CoACD timed out after {config.coacd_timeout}s on mesh "
                f"with {mesh_count} vertices"
            ),
        ) from exc

    return ExtractionResult(
        hulls=hulls,
        source_vertex_count=source_count,
        mesh_vertex_count=mesh_count,
        build_plan=_plan,
        lod_tiers=lod_tiers,
    )


def decompose_source_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    config: ResolvedConfig,
    _plan: BuildPlan | None = None,
) -> ExtractionResult:
    """Decompose each topological source-mesh component independently.

    Reconstructed point clouds and octree cells have their own partitioning
    policies. This wrapper is intentionally for direct source meshes (GLB,
    OBJ, array input), where disconnected primitives are author-provided solid
    boundaries and should remain separate decomposition units.
    """
    components = split_mesh_components(vertices, faces)
    if len(components) <= 1:
        return decompose_and_build(
            vertices,
            faces,
            source_count,
            len(vertices),
            config,
            _plan=_plan,
        )

    if _plan is not None:
        _plan.step("split_components")
        _plan.step("decompose")
        _plan.detected["mesh_component_count"] = len(components)
        _plan.detected["mesh_component_face_counts"] = [
            len(component_faces) for _, component_faces in components
        ]

    hulls: list[Hull] = []
    lod_by_concavity: dict[float, list[Hull]] = {}
    processed_vertices = 0
    decimated_components = 0

    for component_vertices, component_faces in components:
        child_plan = None
        if _plan is not None:
            child_plan = BuildPlan(
                input_kind=_plan.input_kind,
                collider_kind=_plan.collider_kind,
            )
        result = decompose_and_build(
            component_vertices,
            component_faces,
            len(component_vertices),
            len(component_vertices),
            config,
            _plan=child_plan,
        )
        hulls.extend(result.hulls)
        processed_vertices += (
            child_plan.processed_vertices
            if child_plan is not None
            else result.mesh_vertex_count
        )
        if child_plan is not None:
            _plan.merge_signals(child_plan.child_signals())
            if child_plan.decimated:
                decimated_components += 1
        for tier in result.lod_tiers or []:
            lod_by_concavity.setdefault(tier.concavity, []).extend(tier.hulls)

    if _plan is not None:
        _plan.processed_vertices = processed_vertices
        if decimated_components:
            _plan.decimated = True
            _plan.detected["decimated_components"] = decimated_components

    lod_tiers = None
    if lod_by_concavity:
        lod_tiers = [
            LodHulls(concavity=concavity, hulls=lod_by_concavity[concavity])
            for concavity in sorted(lod_by_concavity)
        ]

    return ExtractionResult(
        hulls=hulls,
        source_vertex_count=source_count,
        mesh_vertex_count=len(vertices),
        build_plan=_plan,
        lod_tiers=lod_tiers,
    )
