"""Trace-backed f32 predicate replay (#108).

Replays CoACD's exact internal planes and component meshes through
the f32 predicate gate, proving that f32 quantization preserves
topology for the actual operations the decomposer performs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chitin.coacd_trace import CoACDTrace, TracedClip
from chitin.f32_policy import DEFAULT_POLICY, QuantizationPolicy
from chitin.f32_predicates import (
    classify_plane_f32,
    classify_plane_f64,
    clip_mesh_f32,
    clip_mesh_f64,
    diff_classifications,
    diff_clips,
    extract_cap_f32,
    extract_cap_f64,
    diff_caps,
)


@dataclass
class TraceReplayReport:
    """Result of replaying one traced clip through f32 predicates."""

    clip_index: int
    component_id: int
    plane_a: float
    plane_b: float
    plane_c: float
    plane_d: float
    num_vertices: int
    num_faces: int
    classification_agrees: bool
    classification_detail: str
    clip_agrees: bool
    clip_detail: str
    cap_agrees: bool
    cap_detail: str


@dataclass
class TraceCorpusReport:
    """Aggregate result of replaying an entire trace corpus."""

    trace_source: str
    num_clips_replayed: int
    num_classification_agree: int
    num_clip_agree: int
    num_cap_agree: int
    reports: list[TraceReplayReport] = field(default_factory=list)
    skipped: int = 0

    @property
    def classification_rate(self) -> float:
        return self.num_classification_agree / max(1, self.num_clips_replayed)

    @property
    def clip_rate(self) -> float:
        return self.num_clip_agree / max(1, self.num_clips_replayed)

    @property
    def cap_rate(self) -> float:
        return self.num_cap_agree / max(1, self.num_clips_replayed)

    @property
    def all_agree(self) -> bool:
        return (
            self.num_classification_agree == self.num_clips_replayed
            and self.num_clip_agree == self.num_clips_replayed
            and self.num_cap_agree == self.num_clips_replayed
        )

    def summary(self) -> str:
        return (
            f"Trace replay: {self.num_clips_replayed} clips, "
            f"classify={self.classification_rate:.1%}, "
            f"clip={self.clip_rate:.1%}, "
            f"cap={self.cap_rate:.1%}, "
            f"skipped={self.skipped}"
        )

    def first_disagreement(self) -> TraceReplayReport | None:
        for r in self.reports:
            if not r.classification_agrees or not r.clip_agrees or not r.cap_agrees:
                return r
        return None


def replay_clip(
    clip: TracedClip,
    clip_index: int,
    policy: QuantizationPolicy = DEFAULT_POLICY,
) -> TraceReplayReport | None:
    """Replay one traced clip through f32 vs f64 predicates.

    Returns None if the clip has no mesh data to replay.
    """
    # Use recorded input mesh if available (trace version 2+);
    # fall back to reconstruction from outputs for legacy traces.
    if clip.input_vertices is not None and clip.input_faces is not None:
        vertices = clip.input_vertices.astype(np.float64)
        faces = clip.input_faces.astype(np.int64)
    else:
        # Legacy: reconstruct from pos/neg outputs (duplicates cut vertices,
        # loses vertex identity — known limitation of trace version 1)
        pos_v = clip.pos_vertices
        neg_v = clip.neg_vertices
        pos_f = clip.pos_triangles
        neg_f = clip.neg_triangles
        if pos_v is None or neg_v is None or pos_f is None or neg_f is None:
            return None
        vertices = np.vstack([pos_v, neg_v]).astype(np.float64)
        neg_f_offset = neg_f + len(pos_v)
        faces = np.vstack([pos_f, neg_f_offset]).astype(np.int64)

    # Plane as point + normal
    plane = clip.plane
    n = np.array([plane.a, plane.b, plane.c], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-15:
        return None
    normal = n / norm
    # Plane eq: n.p + d = 0 => p = -d * n / |n|^2
    point = -(plane.d / norm) * normal

    # 1. Classification
    ref_cls = classify_plane_f64(vertices, point, normal)
    cand_cls = classify_plane_f32(vertices, point, normal, policy)
    cls_diff = diff_classifications(ref_cls, cand_cls)

    # 2. Clip
    ref_clip = clip_mesh_f64(vertices, faces, point, normal)
    cand_clip = clip_mesh_f32(vertices, faces, point, normal, policy)
    clip_diff = diff_clips(ref_clip, cand_clip, policy=policy)

    # 3. Cap
    cap_ref = extract_cap_f64(ref_clip)
    cap_cand = extract_cap_f32(cand_clip, policy)
    cap_diff = diff_caps(cap_ref, cap_cand)

    return TraceReplayReport(
        clip_index=clip_index,
        component_id=clip.component_id,
        plane_a=plane.a,
        plane_b=plane.b,
        plane_c=plane.c,
        plane_d=plane.d,
        num_vertices=len(vertices),
        num_faces=len(faces),
        classification_agrees=cls_diff.agrees,
        classification_detail=cls_diff.first_divergence or "",
        clip_agrees=clip_diff.agrees,
        clip_detail=clip_diff.first_divergence or "",
        cap_agrees=cap_diff.agrees,
        cap_detail=cap_diff.first_divergence or "",
    )


@dataclass
class OracleComparison:
    """Result of comparing f32 classification against C++ oracle decisions."""

    clip_index: int
    num_vertices: int
    num_agree: int
    num_disagree: int
    near_plane_disagree: int  # disagreements where |dot| < threshold
    far_plane_disagree: int  # disagreements where |dot| >= threshold
    max_dot_at_disagree: float  # largest |dot product| at a disagreement

    @property
    def agreement_rate(self) -> float:
        return self.num_agree / self.num_vertices if self.num_vertices > 0 else 1.0


def compare_oracle(
    clip: TracedClip,
    clip_index: int,
    policy: QuantizationPolicy = DEFAULT_POLICY,
    near_plane_threshold: float = 1e-3,
) -> OracleComparison | None:
    """Compare f32 classification directly against recorded C++ oracle decisions.

    Returns None if the trace lacks oracle data.
    """
    if clip.oracle_sides is None or clip.input_vertices is None:
        return None

    vertices = clip.input_vertices.astype(np.float64)
    normal = np.array([clip.plane.a, clip.plane.b, clip.plane.c], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-15:
        return None
    normal = normal / norm
    point = normal * (-clip.plane.d / np.dot(normal, normal))

    f32_sides = classify_plane_f32(vertices, point, normal, policy)
    oracle_sides = clip.oracle_sides.astype(np.int8)

    agree_mask = f32_sides == oracle_sides
    num_agree = int(np.sum(agree_mask))
    num_disagree = len(vertices) - num_agree

    # Compute dot products to characterize disagreements
    dots = np.dot(vertices - point, normal)
    disagree_dots = np.abs(dots[~agree_mask]) if num_disagree > 0 else np.array([])

    near_plane = (
        int(np.sum(disagree_dots < near_plane_threshold)) if num_disagree > 0 else 0
    )
    far_plane = num_disagree - near_plane
    max_dot = float(np.max(disagree_dots)) if num_disagree > 0 else 0.0

    return OracleComparison(
        clip_index=clip_index,
        num_vertices=len(vertices),
        num_agree=num_agree,
        num_disagree=num_disagree,
        near_plane_disagree=near_plane,
        far_plane_disagree=far_plane,
        max_dot_at_disagree=max_dot,
    )


def replay_trace(
    trace: CoACDTrace,
    policy: QuantizationPolicy = DEFAULT_POLICY,
    max_clips: int | None = None,
) -> TraceCorpusReport:
    """Replay all clips in a trace through f32 predicates."""
    report = TraceCorpusReport(
        trace_source=f"call_{trace.call_id}",
        num_clips_replayed=0,
        num_classification_agree=0,
        num_clip_agree=0,
        num_cap_agree=0,
    )

    clips = trace.clips
    if max_clips is not None:
        clips = clips[:max_clips]

    for i, clip in enumerate(clips):
        result = replay_clip(clip, i, policy)
        if result is None:
            report.skipped += 1
            continue
        report.num_clips_replayed += 1
        report.reports.append(result)
        if result.classification_agrees:
            report.num_classification_agree += 1
        if result.clip_agrees:
            report.num_clip_agree += 1
        if result.cap_agrees:
            report.num_cap_agree += 1

    return report


def replay_classifications(
    trace: CoACDTrace,
    policy: QuantizationPolicy = DEFAULT_POLICY,
    max_clips: int | None = None,
) -> TraceCorpusReport:
    """Fast path: only replay vertex classification (O(n) per clip)."""
    report = TraceCorpusReport(
        trace_source=f"call_{trace.call_id}",
        num_clips_replayed=0,
        num_classification_agree=0,
        num_clip_agree=0,
        num_cap_agree=0,
    )

    clips = trace.clips
    if max_clips is not None:
        clips = clips[:max_clips]

    for i, clip in enumerate(clips):
        if clip.pos_vertices is None or clip.neg_vertices is None:
            report.skipped += 1
            continue
        if clip.pos_triangles is None or clip.neg_triangles is None:
            report.skipped += 1
            continue

        if clip.input_vertices is not None and clip.input_faces is not None:
            vertices = clip.input_vertices.astype(np.float64)
        else:
            pos_v = clip.pos_vertices.astype(np.float64)
            neg_v = clip.neg_vertices.astype(np.float64)
            vertices = np.vstack([pos_v, neg_v])

        plane = clip.plane
        n = np.array([plane.a, plane.b, plane.c], dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            report.skipped += 1
            continue
        normal = n / norm
        point = -(plane.d / norm) * normal

        ref_cls = classify_plane_f64(vertices, point, normal)
        cand_cls = classify_plane_f32(vertices, point, normal, policy)
        cls_diff = diff_classifications(ref_cls, cand_cls)

        report.num_clips_replayed += 1
        if cls_diff.agrees:
            report.num_classification_agree += 1
            report.num_clip_agree += 1
            report.num_cap_agree += 1
        else:
            report.reports.append(
                TraceReplayReport(
                    clip_index=i,
                    component_id=clip.component_id,
                    plane_a=plane.a,
                    plane_b=plane.b,
                    plane_c=plane.c,
                    plane_d=plane.d,
                    num_vertices=len(vertices),
                    num_faces=len(clip.pos_triangles) + len(clip.neg_triangles),
                    classification_agrees=False,
                    classification_detail=cls_diff.first_divergence or "",
                    clip_agrees=False,
                    clip_detail="classification disagrees",
                    cap_agrees=False,
                    cap_detail="classification disagrees",
                )
            )

    return report


def replay_traces(
    traces: list[CoACDTrace],
    policy: QuantizationPolicy = DEFAULT_POLICY,
    max_clips_per_trace: int | None = None,
) -> TraceCorpusReport:
    """Replay multiple traces, aggregate into one report."""
    combined = TraceCorpusReport(
        trace_source="corpus",
        num_clips_replayed=0,
        num_classification_agree=0,
        num_clip_agree=0,
        num_cap_agree=0,
    )

    for trace in traces:
        r = replay_trace(trace, policy, max_clips_per_trace)
        combined.num_clips_replayed += r.num_clips_replayed
        combined.num_classification_agree += r.num_classification_agree
        combined.num_clip_agree += r.num_clip_agree
        combined.num_cap_agree += r.num_cap_agree
        combined.reports.extend(r.reports)
        combined.skipped += r.skipped

    return combined
