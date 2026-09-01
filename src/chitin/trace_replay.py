"""Replay and diff two traces to find the first divergent stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
            return ReplayResult(
                identical=False,
                divergence=Divergence(
                    stage_index=i,
                    stage_name=ref_e.stage,
                    kind="output_mismatch",
                    reference_digest=ref_e.output_digest,
                    candidate_digest=cand_e.output_digest,
                    detail={
                        "reference_shape": ref_e.output_shape,
                        "candidate_shape": cand_e.output_shape,
                        "reference_metadata": ref_e.metadata,
                        "candidate_metadata": cand_e.metadata,
                    },
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
