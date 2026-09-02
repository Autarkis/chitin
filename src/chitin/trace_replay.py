"""Replay and diff two traces to find the first divergent stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from chitin.trace import TraceRecorder


@dataclass
class Divergence:
    """The first point where two traces disagree."""

    stage_index: int
    stage_name: str
    kind: str  # "input_mismatch", "output_mismatch", "missing_stage", "extra_stage"
    reference_digest: str
    candidate_digest: str
    detail: dict[str, Any]


@dataclass
class HullDivergence:
    """Semantic description of how two hull sets differ."""

    hull_count_reference: int
    hull_count_candidate: int
    first_divergent_hull: int | None  # index, or None if count differs
    vertex_count_delta: int | None  # at first_divergent_hull
    max_vertex_displacement: float | None  # L2 norm, at first_divergent_hull
    face_count_delta: int | None
    winding_consistent_reference: bool | None
    winding_consistent_candidate: bool | None


def _check_winding_consistency(vertices: np.ndarray, indices: np.ndarray) -> bool:
    """Check if all triangle face normals point consistently outward."""
    indices_2d = np.asarray(indices).reshape(-1, 3)
    vertices = np.asarray(vertices, dtype=np.float64)
    v0 = vertices[indices_2d[:, 0]]
    v1 = vertices[indices_2d[:, 1]]
    v2 = vertices[indices_2d[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    # Signed volume via divergence theorem
    centroids = (v0 + v1 + v2) / 3
    dots = np.sum(normals * centroids, axis=1)
    # All same sign = consistent winding
    return bool(np.all(dots > 0) or np.all(dots < 0))


def _compare_hulls_semantically(
    ref_recorder: TraceRecorder,
    cand_recorder: TraceRecorder,
    stage_index: int,
) -> HullDivergence | None:
    """Compare hull blobs recorded for a decompose stage at `stage_index`.

    Returns None if either recorder has no hull blobs for that stage (i.e.
    the stage wasn't a decompose stage or blobs weren't retained).
    """
    ref_stage = ref_recorder.events[stage_index].stage
    cand_stage = cand_recorder.events[stage_index].stage
    ref_prefix = f"{stage_index}_{ref_stage}_hull_"
    cand_prefix = f"{stage_index}_{cand_stage}_hull_"

    def _hull_count(recorder: TraceRecorder, prefix: str) -> int:
        count = 0
        while f"{prefix}{count}_v" in recorder._blobs:
            count += 1
        return count

    def _winding_or_none(verts: np.ndarray, indices: np.ndarray) -> bool | None:
        # Hull index buffers are only guaranteed valid against their own
        # vertex buffer; malformed/synthetic traces can carry indices that
        # exceed it. Winding is a diagnostic extra, so failing to compute it
        # degrades to "unknown" rather than aborting the whole comparison.
        try:
            return _check_winding_consistency(verts, indices)
        except (IndexError, ValueError):
            return None

    ref_count = _hull_count(ref_recorder, ref_prefix)
    cand_count = _hull_count(cand_recorder, cand_prefix)

    if ref_count == 0 and cand_count == 0:
        return None

    first_divergent: int | None = None
    vertex_count_delta: int | None = None
    max_vertex_displacement: float | None = None
    face_count_delta: int | None = None
    winding_ref: bool | None = None
    winding_cand: bool | None = None

    if ref_count != cand_count:
        first_divergent = None
    else:
        for i in range(ref_count):
            ref_v = ref_recorder._blobs[f"{ref_prefix}{i}_v"]
            ref_i = ref_recorder._blobs[f"{ref_prefix}{i}_i"]
            cand_v = cand_recorder._blobs[f"{cand_prefix}{i}_v"]
            cand_i = cand_recorder._blobs[f"{cand_prefix}{i}_i"]

            if len(ref_v) != len(cand_v) or len(ref_i) != len(cand_i):
                first_divergent = i
                vertex_count_delta = len(cand_v) - len(ref_v)
                face_count_delta = (len(cand_i) - len(ref_i)) // 3
                winding_ref = _winding_or_none(ref_v, ref_i)
                winding_cand = _winding_or_none(cand_v, cand_i)
                break

            if not np.array_equal(ref_v, cand_v) or not np.array_equal(ref_i, cand_i):
                first_divergent = i
                vertex_count_delta = 0
                face_count_delta = 0
                if len(ref_v) == len(cand_v):
                    max_vertex_displacement = float(
                        np.max(np.linalg.norm(cand_v - ref_v, axis=1))
                    )
                winding_ref = _winding_or_none(ref_v, ref_i)
                winding_cand = _winding_or_none(cand_v, cand_i)
                break

    return HullDivergence(
        hull_count_reference=ref_count,
        hull_count_candidate=cand_count,
        first_divergent_hull=first_divergent,
        vertex_count_delta=vertex_count_delta,
        max_vertex_displacement=max_vertex_displacement,
        face_count_delta=face_count_delta,
        winding_consistent_reference=winding_ref,
        winding_consistent_candidate=winding_cand,
    )


@dataclass
class ReplayResult:
    """Outcome of replaying a candidate trace against a reference."""

    identical: bool
    divergence: Divergence | None
    stages_compared: int


def replay_and_diff(
    reference: TraceRecorder,
    candidate: TraceRecorder,
) -> ReplayResult:
    """Compare two traces stage-by-stage, report the first divergence.

    Checks:
    1. Same number of stages
    2. Same stage names in order
    3. Same input digest at each stage
    4. Same output digest at each stage
    """
    ref_events = reference.events
    cand_events = candidate.events

    min_len = min(len(ref_events), len(cand_events))

    for i in range(min_len):
        ref_e = ref_events[i]
        cand_e = cand_events[i]

        if ref_e.stage != cand_e.stage:
            return ReplayResult(
                identical=False,
                divergence=Divergence(
                    stage_index=i,
                    stage_name=ref_e.stage,
                    kind="stage_name_mismatch",
                    reference_digest="",
                    candidate_digest="",
                    detail={
                        "reference_stage": ref_e.stage,
                        "candidate_stage": cand_e.stage,
                    },
                ),
                stages_compared=i,
            )

        if ref_e.input_digest != cand_e.input_digest:
            return ReplayResult(
                identical=False,
                divergence=Divergence(
                    stage_index=i,
                    stage_name=ref_e.stage,
                    kind="input_mismatch",
                    reference_digest=ref_e.input_digest,
                    candidate_digest=cand_e.input_digest,
                    detail={
                        "reference_shape": ref_e.input_shape,
                        "candidate_shape": cand_e.input_shape,
                    },
                ),
                stages_compared=i,
            )

        if ref_e.output_digest != cand_e.output_digest:
            detail: dict[str, Any] = {
                "reference_shape": ref_e.output_shape,
                "candidate_shape": cand_e.output_shape,
                "reference_metadata": ref_e.metadata,
                "candidate_metadata": cand_e.metadata,
            }
            hull_div = _compare_hulls_semantically(reference, candidate, i)
            if hull_div is not None:
                detail["hull_divergence"] = asdict(hull_div)
            return ReplayResult(
                identical=False,
                divergence=Divergence(
                    stage_index=i,
                    stage_name=ref_e.stage,
                    kind="output_mismatch",
                    reference_digest=ref_e.output_digest,
                    candidate_digest=cand_e.output_digest,
                    detail=detail,
                ),
                stages_compared=i,
            )

    if len(ref_events) != len(cand_events):
        longer = "reference" if len(ref_events) > len(cand_events) else "candidate"
        extra = (ref_events if longer == "reference" else cand_events)[min_len]
        return ReplayResult(
            identical=False,
            divergence=Divergence(
                stage_index=min_len,
                stage_name=extra.stage,
                kind="extra_stage",
                reference_digest="",
                candidate_digest="",
                detail={
                    "extra_in": longer,
                    "reference_count": len(ref_events),
                    "candidate_count": len(cand_events),
                },
            ),
            stages_compared=min_len,
        )

    return ReplayResult(
        identical=True,
        divergence=None,
        stages_compared=len(ref_events),
    )
