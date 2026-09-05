"""Intersection welding — deduplicate by EdgeKey, rewrite indices.

CPU-side for now; typical intersection counts (hundreds, not millions)
make GPU sort overkill. The vectorized numpy remap handles the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chitin.gpu.clip import ClipEmitResult


@dataclass
class WeldResult:
    positive_faces: np.ndarray  # (Fp, 3) uint32, ixn refs welded
    negative_faces: np.ndarray  # (Fn, 3) uint32, ixn refs welded
    welded_positions: np.ndarray  # (W, 3) float32
    welded_edge_keys: np.ndarray  # (W, 2) uint32
    boundary_edges: np.ndarray  # (B, 2) uint32, welded
    positive_ancestry: np.ndarray  # (Fp,) uint32
    negative_ancestry: np.ndarray  # (Fn,) uint32
    intersection_count_raw: int
    intersection_count_welded: int


def _remap_ixn(arr: np.ndarray, v_count: int, weld_map: np.ndarray) -> np.ndarray:
    """Remap intersection indices (>= v_count) from raw to welded."""
    out = arr.astype(np.uint32, copy=True)
    mask = out >= v_count
    out[mask] = v_count + weld_map[out[mask] - v_count]
    return out


def weld_intersections_reference(
    emit_result: ClipEmitResult, vertex_count: int
) -> WeldResult:
    """Deduplicate intersection vertices by EdgeKey and rewrite indices.

    Two triangles that cut the same source edge produce two raw
    intersection records for it (one per emitting triangle); welding
    collapses those into a single shared vertex so the clipped mesh
    stays watertight.
    """
    if emit_result.total_intersections == 0:
        return WeldResult(
            positive_faces=emit_result.positive_faces.copy(),
            negative_faces=emit_result.negative_faces.copy(),
            welded_positions=np.zeros((0, 3), dtype=np.float32),
            welded_edge_keys=np.zeros((0, 2), dtype=np.uint32),
            boundary_edges=emit_result.boundary_edges.copy(),
            positive_ancestry=emit_result.positive_ancestry.copy(),
            negative_ancestry=emit_result.negative_ancestry.copy(),
            intersection_count_raw=0,
            intersection_count_welded=0,
        )

    edge_keys = emit_result.intersection_edges
    positions = emit_result.intersection_positions
    raw_count = emit_result.total_intersections

    key_to_welded: dict[tuple[int, int], int] = {}
    weld_map = np.zeros(raw_count, dtype=np.uint32)
    pos_list: list[np.ndarray] = []
    key_list: list[tuple[int, int]] = []

    for i in range(raw_count):
        key = (int(edge_keys[i, 0]), int(edge_keys[i, 1]))
        welded_idx = key_to_welded.get(key)
        if welded_idx is None:
            welded_idx = len(pos_list)
            key_to_welded[key] = welded_idx
            pos_list.append(positions[i])
            key_list.append(key)
        weld_map[i] = welded_idx

    welded_count = len(pos_list)
    if welded_count > 0:
        welded_positions = np.array(pos_list, dtype=np.float32)
        welded_edge_keys = np.array(key_list, dtype=np.uint32)
    else:
        welded_positions = np.zeros((0, 3), dtype=np.float32)
        welded_edge_keys = np.zeros((0, 2), dtype=np.uint32)

    pos_faces = _remap_ixn(emit_result.positive_faces, vertex_count, weld_map)
    neg_faces = _remap_ixn(emit_result.negative_faces, vertex_count, weld_map)
    boundary = _remap_ixn(emit_result.boundary_edges, vertex_count, weld_map)

    return WeldResult(
        positive_faces=pos_faces,
        negative_faces=neg_faces,
        welded_positions=welded_positions,
        welded_edge_keys=welded_edge_keys,
        boundary_edges=boundary,
        positive_ancestry=emit_result.positive_ancestry.copy(),
        negative_ancestry=emit_result.negative_ancestry.copy(),
        intersection_count_raw=raw_count,
        intersection_count_welded=welded_count,
    )
