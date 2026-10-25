"""GPU clip pipeline types and input validation.

Status codes, per-operation metrics, and the unified pipeline result
type for ``dispatch_clip_pipeline``.  Input validation rejects NaN,
Inf, out-of-range indices, and degenerate plane normals before any
GPU work.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class ClipStatus(enum.IntEnum):
    OK = 0
    ALL_POSITIVE = 1
    ALL_NEGATIVE = 2
    ALL_COPLANAR = 3
    CAPACITY_EXCEEDED = 4
    INVALID_INPUT = 5
    TOPOLOGY_ERROR = 6
    UNSUPPORTED_TOPOLOGY = 7

    @property
    def is_success(self) -> bool:
        return self in (
            ClipStatus.OK,
            ClipStatus.ALL_POSITIVE,
            ClipStatus.ALL_NEGATIVE,
            ClipStatus.ALL_COPLANAR,
        )


@dataclass(frozen=True, slots=True)
class ClipMetrics:
    classify_fast_count: int = 0
    classify_fallback_count: int = 0
    classify_on_plane_count: int = 0
    input_face_count: int = 0
    positive_face_count: int = 0
    negative_face_count: int = 0
    coplanar_face_count: int = 0
    intersection_count_raw: int = 0
    intersection_count_welded: int = 0
    boundary_loop_count: int = 0
    cap_face_count: int = 0


@dataclass(frozen=True, slots=True)
class ClipDiagnostics:
    """Per-clip stage intermediates for divergence attribution."""

    signs: np.ndarray  # (V,) int32 — per-vertex classification
    weld_result: object | None = None  # WeldResult — avoid circular import
    loop_result: object | None = None  # LoopExtractionResult
    negative_loop_result: object | None = None
    cap_result: object | None = None  # CapResult
    boundary_edges_raw: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.uint32)
    )


@dataclass(frozen=True, slots=True)
class ClipPipelineResult:
    status: ClipStatus
    positive_vertices: np.ndarray
    positive_faces: np.ndarray
    negative_vertices: np.ndarray
    negative_faces: np.ndarray
    positive_ancestry: np.ndarray
    negative_ancestry: np.ndarray
    intersection_records: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 5), dtype=np.float32)
    )
    metrics: ClipMetrics = field(default_factory=ClipMetrics)
    diagnostics: ClipDiagnostics | None = None

    @staticmethod
    def failure(status: ClipStatus) -> ClipPipelineResult:
        return ClipPipelineResult(
            status=status,
            positive_vertices=np.zeros((0, 3), dtype=np.float32),
            positive_faces=np.zeros((0, 3), dtype=np.uint32),
            negative_vertices=np.zeros((0, 3), dtype=np.float32),
            negative_faces=np.zeros((0, 3), dtype=np.uint32),
            positive_ancestry=np.array([], dtype=np.uint32),
            negative_ancestry=np.array([], dtype=np.uint32),
        )


def validate_clip_inputs(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> ClipStatus | None:
    """Return ``INVALID_INPUT`` if any precondition fails, else ``None``."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    plane_point = np.asarray(plane_point, dtype=np.float32)
    plane_normal = np.asarray(plane_normal, dtype=np.float32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return ClipStatus.INVALID_INPUT
    if faces.ndim != 2 or faces.shape[1] != 3:
        return ClipStatus.INVALID_INPUT
    if plane_point.shape != (3,) and plane_point.shape != (1, 3):
        return ClipStatus.INVALID_INPUT
    if plane_normal.shape != (3,) and plane_normal.shape != (1, 3):
        return ClipStatus.INVALID_INPUT

    if not np.all(np.isfinite(vertices)):
        return ClipStatus.INVALID_INPUT
    if not np.all(np.isfinite(plane_point)):
        return ClipStatus.INVALID_INPUT
    if not np.all(np.isfinite(plane_normal)):
        return ClipStatus.INVALID_INPUT

    normal_sq = float(np.sum(plane_normal.ravel() ** 2))
    if normal_sq == 0.0:
        return ClipStatus.INVALID_INPUT

    v_count = len(vertices)
    if v_count > 0 and len(faces) > 0 and np.any(faces >= v_count):
        return ClipStatus.INVALID_INPUT

    return None


def compute_arena_bounds(vertex_count: int, face_count: int) -> dict[str, int]:
    """Worst-case arena sizes (element counts) per contract Capacity section."""
    return {
        "signs": vertex_count,
        "per_tri_counts": face_count,
        "scan_workspace": face_count,
        "output_vertices_per_side": vertex_count + 2 * face_count,
        "output_faces_per_side": 2 * face_count,
        "intersection_records": 2 * face_count,
        "boundary_edges": face_count,
        "cap_faces_per_side": 2 * face_count,
    }
