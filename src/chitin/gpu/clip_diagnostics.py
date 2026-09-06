"""Stage-level divergence attribution for CPU vs GPU clip comparison.

Compares pipeline intermediates using canonical vertex identities
(original index for source vertices, EdgeKey for intersection vertices)
rather than positions, so intersection-coordinate differences don't
mask or create false topology divergences.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass

import numpy as np


class DivergenceStage(enum.Enum):
    CLASSIFY = "classify"
    EMIT = "emit"
    WELD = "weld"
    BOUNDARY = "boundary"
    LOOPS = "loops"
    CAPS = "caps"
    AGREE = "agree"


@dataclass
class DivergenceReport:
    """Result of stage-by-stage comparison."""

    stage: DivergenceStage
    details: str
    # Per-stage pass/fail for the full report
    classify_ok: bool = True
    emit_ok: bool = True
    weld_ok: bool = True
    boundary_ok: bool = True
    loops_ok: bool = True
    caps_ok: bool = True


@dataclass
class CpuDiagnostics:
    """CPU oracle intermediates for stage comparison."""

    signs: np.ndarray
    pos_faces: np.ndarray  # clip faces only, no caps
    neg_faces: np.ndarray
    pos_edge_key_map: np.ndarray  # (K,2) uint32 for positive clip
    neg_edge_key_map: np.ndarray  # (K,2) uint32 for negative clip
    pos_boundary_edges: np.ndarray  # already parity-deduped
    neg_boundary_edges: np.ndarray
    pos_loops: list[np.ndarray]
    neg_loops: list[np.ndarray]
    pos_cap_faces: np.ndarray
    neg_cap_faces: np.ndarray
    pos_winding_consistent: bool
    neg_winding_consistent: bool
    pos_vertex_count: int  # number of original (source) vertices in pos clip
    neg_vertex_count: int


def _canonicalize_vertex(v: int, vertex_count: int, edge_keys: np.ndarray) -> tuple:
    """Convert a vertex index to its canonical identity.

    Source vertex (v < vertex_count): identity is (0, v)
    Intersection vertex (v >= vertex_count): identity is (1, v_lo, v_hi)
    """
    if v < vertex_count:
        return (0, int(v))
    else:
        ek_idx = v - vertex_count
        if ek_idx >= len(edge_keys):
            return (-1, int(v))  # unmapped — signals a problem
        return (1, int(edge_keys[ek_idx, 0]), int(edge_keys[ek_idx, 1]))


def _canonicalize_face(
    face: np.ndarray, vertex_count: int, edge_keys: np.ndarray
) -> tuple | None:
    """Convert face to sorted canonical identity triple."""
    canonical = []
    for v in face:
        cid = _canonicalize_vertex(int(v), vertex_count, edge_keys)
        if cid[0] == -1:
            return None
        canonical.append(cid)
    return tuple(sorted(canonical))


def _canonicalize_edge(
    edge: np.ndarray, vertex_count: int, edge_keys: np.ndarray
) -> tuple | None:
    """Convert edge to sorted canonical identity pair."""
    a = _canonicalize_vertex(int(edge[0]), vertex_count, edge_keys)
    b = _canonicalize_vertex(int(edge[1]), vertex_count, edge_keys)
    if a[0] == -1 or b[0] == -1:
        return None
    return tuple(sorted([a, b]))


def _face_multiset(
    faces: np.ndarray, vertex_count: int, edge_keys: np.ndarray
) -> Counter:
    """Build canonical face multiset."""
    counter: Counter = Counter()
    for f in faces:
        cf = _canonicalize_face(f, vertex_count, edge_keys)
        if cf is not None:
            counter[cf] += 1
    return counter


def _edge_multiset(
    edges: np.ndarray, vertex_count: int, edge_keys: np.ndarray
) -> Counter:
    """Build canonical edge multiset."""
    counter: Counter = Counter()
    for e in edges:
        ce = _canonicalize_edge(e, vertex_count, edge_keys)
        if ce is not None:
            counter[ce] += 1
    return counter


def _parity_dedup(edge_counter: Counter) -> Counter:
    """Keep only odd-multiplicity edges (parity cancellation)."""
    return Counter({k: v for k, v in edge_counter.items() if v % 2 == 1})


def _compare_signs(cpu_signs: np.ndarray, gpu_signs: np.ndarray) -> tuple[bool, str]:
    """Compare per-vertex classification signs."""
    if len(cpu_signs) != len(gpu_signs):
        return (
            False,
            f"sign array length mismatch: cpu={len(cpu_signs)} gpu={len(gpu_signs)}",
        )
    diff_mask = cpu_signs != gpu_signs
    n_diff = int(diff_mask.sum())
    if n_diff == 0:
        return True, "signs agree"
    # Report first few disagreements
    diff_idx = np.nonzero(diff_mask)[0][:5]
    details = "; ".join(
        f"v{i}: cpu={cpu_signs[i]} gpu={gpu_signs[i]}" for i in diff_idx
    )
    return False, f"{n_diff} sign disagreements: {details}"


def _compare_emit(
    cpu_faces: np.ndarray,
    gpu_faces: np.ndarray,
    cpu_vertex_count: int,
    gpu_vertex_count: int,
    cpu_edge_keys: np.ndarray,
    gpu_edge_keys: np.ndarray,
    label: str,
) -> tuple[bool, str]:
    """Compare emitted faces using canonical identities."""
    cpu_ms = _face_multiset(cpu_faces, cpu_vertex_count, cpu_edge_keys)
    gpu_ms = _face_multiset(gpu_faces, gpu_vertex_count, gpu_edge_keys)

    if cpu_ms == gpu_ms:
        return True, f"{label} faces agree ({len(cpu_faces)} faces)"

    only_cpu = cpu_ms - gpu_ms
    only_gpu = gpu_ms - cpu_ms
    details_parts = [f"{label}: cpu={len(cpu_faces)} gpu={len(gpu_faces)}"]
    if only_cpu:
        details_parts.append(f"only_cpu={len(only_cpu)} faces")
    if only_gpu:
        details_parts.append(f"only_gpu={len(only_gpu)} faces")
    return False, "; ".join(details_parts)


def _compare_boundary_parity(
    cpu_edges: np.ndarray,
    gpu_edges: np.ndarray,
    cpu_vertex_count: int,
    gpu_vertex_count: int,
    cpu_edge_keys: np.ndarray,
    gpu_edge_keys: np.ndarray,
    label: str,
) -> tuple[bool, str]:
    """Compare boundary edges after parity cancellation.

    CPU boundary edges are already parity-deduped.
    GPU boundary edges are NOT — apply parity dedup here.
    """
    cpu_ms = _edge_multiset(cpu_edges, cpu_vertex_count, cpu_edge_keys)
    gpu_raw = _edge_multiset(gpu_edges, gpu_vertex_count, gpu_edge_keys)
    gpu_deduped = _parity_dedup(gpu_raw)

    if cpu_ms == gpu_deduped:
        return (
            True,
            f"{label} boundary edges agree after parity ({len(cpu_edges)} edges)",
        )

    only_cpu = cpu_ms - gpu_deduped
    only_gpu = gpu_deduped - cpu_ms
    # Also report if GPU has even-multiplicity edges that should cancel
    gpu_even = {k: v for k, v in gpu_raw.items() if v % 2 == 0 and v > 0}

    details = [
        (
            f"{label}: cpu={len(cpu_edges)} gpu_raw={len(gpu_edges)} "
            f"gpu_deduped={sum(gpu_deduped.values())}"
        ),
    ]
    if only_cpu:
        details.append(f"only_cpu={len(only_cpu)}")
    if only_gpu:
        details.append(f"only_gpu={len(only_gpu)}")
    if gpu_even:
        details.append(f"gpu_even_parity_edges={len(gpu_even)}")
    return False, "; ".join(details)


def _canonicalize_loop(
    loop: np.ndarray,
    vertex_count: int,
    edge_keys: np.ndarray,
) -> frozenset:
    """Convert a loop's vertex indices to canonical identity set."""
    canonical = set()
    for v in loop:
        cid = _canonicalize_vertex(int(v), vertex_count, edge_keys)
        canonical.add(cid)
    return frozenset(canonical)


def _compare_loops(
    cpu_loops: list[np.ndarray],
    gpu_loops: list[np.ndarray],
    gpu_valid: bool,
    label: str,
    cpu_vertex_count: int = 0,
    gpu_vertex_count: int = 0,
    cpu_edge_keys: np.ndarray | None = None,
    gpu_edge_keys: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Compare extracted boundary loops using canonical vertex identities.

    Loops are compared as canonical identity sets (not raw vertex indices),
    since CPU and GPU use different intersection vertex index assignments.
    """
    if not gpu_valid:
        return False, f"{label} GPU loop extraction failed (valid=False)"

    if len(cpu_loops) != len(gpu_loops):
        return False, f"{label} loop count: cpu={len(cpu_loops)} gpu={len(gpu_loops)}"

    if cpu_edge_keys is None or gpu_edge_keys is None:
        cpu_sets = sorted(
            [frozenset(int(v) for v in loop) for loop in cpu_loops], key=len
        )
        gpu_sets = sorted(
            [frozenset(int(v) for v in loop) for loop in gpu_loops], key=len
        )
    else:
        cpu_sets = sorted(
            [
                _canonicalize_loop(lp, cpu_vertex_count, cpu_edge_keys)
                for lp in cpu_loops
            ],
            key=len,
        )
        gpu_sets = sorted(
            [
                _canonicalize_loop(lp, gpu_vertex_count, gpu_edge_keys)
                for lp in gpu_loops
            ],
            key=len,
        )

    if cpu_sets == gpu_sets:
        return True, f"{label} loops agree ({len(cpu_loops)} loops)"

    return False, f"{label} loop vertex sets differ"


def attribute_divergence(
    gpu_diagnostics,
    cpu_diagnostics: CpuDiagnostics,
    source_vertex_count: int,
) -> DivergenceReport:
    """Compare CPU and GPU clip pipeline at each stage, report earliest divergence.

    Parameters
    ----------
    gpu_diagnostics : ClipDiagnostics
        From ClipPipelineResult.diagnostics
    cpu_diagnostics : CpuDiagnostics
        From run_cpu_oracle_diagnostic
    source_vertex_count : int
        Number of vertices in the original input mesh
    """
    report = DivergenceReport(stage=DivergenceStage.AGREE, details="all stages agree")

    # Stage 1: classify
    classify_ok, classify_detail = _compare_signs(
        cpu_diagnostics.signs, gpu_diagnostics.signs
    )
    report.classify_ok = classify_ok
    if not classify_ok:
        report.stage = DivergenceStage.CLASSIFY
        report.details = classify_detail
        return report

    # Stage 2: emit — compare clip faces using canonical identities
    # GPU: weld_result has faces with welded intersection indices
    # CPU: clip result has faces with CPU intersection indices
    weld_result = gpu_diagnostics.weld_result
    if weld_result is None:
        # No weld result means early exit (all same sign) — no emit to compare
        report.stage = DivergenceStage.AGREE
        report.details = "no clip geometry (all same sign)"
        return report

    # GPU vertex count = source_vertex_count (intersection verts offset from there)
    gpu_vcount = source_vertex_count

    # Filter out cap faces from GPU (ancestry != 0xFFFFFFFF)
    pos_clip_mask = weld_result.positive_ancestry != 0xFFFFFFFF
    neg_clip_mask = weld_result.negative_ancestry != 0xFFFFFFFF
    gpu_pos_clip = weld_result.positive_faces[pos_clip_mask]
    gpu_neg_clip = weld_result.negative_faces[neg_clip_mask]

    pos_emit_ok, pos_emit_detail = _compare_emit(
        cpu_diagnostics.pos_faces,
        gpu_pos_clip,
        cpu_diagnostics.pos_vertex_count,
        gpu_vcount,
        cpu_diagnostics.pos_edge_key_map,
        weld_result.welded_edge_keys,
        "pos",
    )
    neg_emit_ok, neg_emit_detail = _compare_emit(
        cpu_diagnostics.neg_faces,
        gpu_neg_clip,
        cpu_diagnostics.neg_vertex_count,
        gpu_vcount,
        cpu_diagnostics.neg_edge_key_map,
        weld_result.welded_edge_keys,
        "neg",
    )
    report.emit_ok = pos_emit_ok and neg_emit_ok
    if not report.emit_ok:
        report.stage = DivergenceStage.EMIT
        report.details = f"pos: {pos_emit_detail}; neg: {neg_emit_detail}"
        return report

    # Stage 3: weld — compare welded EdgeKey sets
    # Filter out (v, v) self-loop sentinels: GPU weld chain processes
    # zero-padded arena slots, producing a spurious (0,0) unique entry.
    gpu_ek_set = {
        tuple(int(x) for x in ek)
        for ek in weld_result.welded_edge_keys
        if ek[0] != ek[1]
    }
    cpu_ek_set: set[tuple[int, ...]] = set()
    for ekm in [cpu_diagnostics.pos_edge_key_map, cpu_diagnostics.neg_edge_key_map]:
        for ek in ekm:
            cpu_ek_set.add((int(ek[0]), int(ek[1])))

    gpu_sentinels = sum(1 for ek in weld_result.welded_edge_keys if ek[0] == ek[1])
    if gpu_ek_set != cpu_ek_set:
        report.weld_ok = False
        only_gpu_ek = gpu_ek_set - cpu_ek_set
        only_cpu_ek = cpu_ek_set - gpu_ek_set
        report.stage = DivergenceStage.WELD
        report.details = (
            f"EdgeKey set mismatch: only_gpu={len(only_gpu_ek)} "
            f"only_cpu={len(only_cpu_ek)}"
            f"{f' (gpu_sentinels={gpu_sentinels})' if gpu_sentinels else ''}"
        )
        return report
    report.weld_ok = True

    # Stage 4: boundary — compare boundary edges after parity dedup
    pos_bnd_ok, pos_bnd_detail = _compare_boundary_parity(
        cpu_diagnostics.pos_boundary_edges,
        weld_result.boundary_edges,
        cpu_diagnostics.pos_vertex_count,
        gpu_vcount,
        cpu_diagnostics.pos_edge_key_map,
        weld_result.welded_edge_keys,
        "pos",
    )
    report.boundary_ok = pos_bnd_ok
    if not pos_bnd_ok:
        report.stage = DivergenceStage.BOUNDARY
        report.details = pos_bnd_detail
        return report

    # Stage 5: loops
    loop_result = gpu_diagnostics.loop_result
    if loop_result is not None:
        loops_ok, loops_detail = _compare_loops(
            cpu_diagnostics.pos_loops,
            loop_result.loops,
            loop_result.valid,
            "pos",
            cpu_vertex_count=source_vertex_count,
            gpu_vertex_count=source_vertex_count,
            cpu_edge_keys=cpu_diagnostics.pos_edge_key_map,
            gpu_edge_keys=weld_result.welded_edge_keys,
        )
        report.loops_ok = loops_ok
        if not loops_ok:
            report.stage = DivergenceStage.LOOPS
            report.details = loops_detail
            return report

    # Stage 6: caps
    cap_result = gpu_diagnostics.cap_result
    if cap_result is not None and cap_result.valid:
        # Compare cap face count and winding
        gpu_pos_caps = len(cap_result.positive_cap_faces)
        cpu_pos_caps = len(cpu_diagnostics.pos_cap_faces)
        gpu_neg_caps = len(cap_result.negative_cap_faces)
        cpu_neg_caps = len(cpu_diagnostics.neg_cap_faces)

        caps_count_ok = gpu_pos_caps == cpu_pos_caps and gpu_neg_caps == cpu_neg_caps
        caps_winding_ok = (
            cpu_diagnostics.pos_winding_consistent
            and cpu_diagnostics.neg_winding_consistent
        )

        report.caps_ok = caps_count_ok and caps_winding_ok
        if not report.caps_ok:
            report.stage = DivergenceStage.CAPS
            details = []
            if not caps_count_ok:
                details.append(
                    f"count: pos cpu={cpu_pos_caps} gpu={gpu_pos_caps}, "
                    f"neg cpu={cpu_neg_caps} gpu={gpu_neg_caps}"
                )
            if not caps_winding_ok:
                details.append("winding inconsistent")
            report.details = "; ".join(details)
            return report

    return report
