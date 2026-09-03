"""Classify the first divergence stage for the #119 regression corpus.

Loads the 114 CoACD clip fixtures under ``tests/fixtures/regression/`` and,
for each one, runs three f32 predicate variants against the f64 reference
(classification -> clip -> cap): ``raw_f32`` (plain float32 arithmetic, no
grid frame at all), ``grid_no_snap`` (grid-frame quantization with
intersection snapping effectively disabled), and ``policy_0_1_0`` (the
shipped default policy, grid quantization plus intersection snapping).

For each variant and each clip this records the earliest stage at which it
disagrees with the f64 reference, and for classification disagreements the
individual vertices whose sign flipped, with their f64 and quantized dot
products. The result is written to ``docs/divergence-report.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.f32_policy import DEFAULT_POLICY, QuantizationPolicy
from chitin.f32_predicates import (
    PlaneClassification,
    _clip_mesh_generic,
    _count_signs,
    _to_grid_frame,
    classify_plane_f32,
    classify_plane_f64,
    clip_mesh_f32,
    clip_mesh_f64,
    diff_caps,
    diff_classifications,
    diff_clips,
    extract_cap_f32,
    extract_cap_f64,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "regression"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
REPORT_PATH = REPO_ROOT / "docs" / "divergence-report.json"

NO_SNAP_POLICY = QuantizationPolicy(
    version="diagnostic",
    grid_bits=20,
    classification_ulp_margin=0,
    intersection_snap_bits=60,
)

STAGE_ORDER = ("classification", "clip", "cap")


def _plane_point_normal(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = plane[:3].astype(np.float64)
    norm = np.linalg.norm(n)
    normal = n / norm
    point = -(plane[3] / norm) * normal
    return point, normal


def _load_clip(
    fixture: str, clip_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = FIXTURES_DIR / fixture / f"clip_{clip_index}.npz"
    data = np.load(path)
    vertices = data["input_vertices"].astype(np.float64)
    faces = data["input_faces"].astype(np.int64)
    point, normal = _plane_point_normal(data["plane"])
    return vertices, faces, point, normal


def _classify_raw_f32(
    vertices: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> PlaneClassification:
    dot = np.sum((vertices - plane_point) * plane_normal, axis=1)
    signs = np.sign(dot).astype(np.int8)
    positive_count, negative_count, on_plane_count = _count_signs(signs)
    return PlaneClassification(signs, positive_count, negative_count, on_plane_count)


def _raw_f32_dot(
    vertices: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    return np.sum((vertices - plane_point) * plane_normal, axis=1)


def _grid_dot(
    vertices: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    policy: QuantizationPolicy,
) -> np.ndarray:
    grid_vertices, grid_plane_point, _centroid, scale_factor = _to_grid_frame(
        vertices, plane_point, policy
    )
    grid_normal_f32 = (plane_normal * scale_factor).astype(np.float32)
    return np.sum(
        (grid_vertices.astype(np.float32) - grid_plane_point.astype(np.float32))
        * grid_normal_f32,
        axis=1,
    )


def _disagreeing_vertices(
    ref_signs: np.ndarray,
    ref_dot: np.ndarray,
    cand_signs: np.ndarray,
    cand_dot: np.ndarray,
    dot_label: str,
) -> list[dict]:
    diff_indices = np.nonzero(ref_signs != cand_signs)[0]
    return [
        {
            "index": int(i),
            "f64_dot": float(ref_dot[i]),
            dot_label: float(cand_dot[i]),
            "plane_distance": float(abs(ref_dot[i])),
        }
        for i in diff_indices
    ]


def _variant_report(
    ref_cls: PlaneClassification,
    ref_clip,
    ref_cap,
    ref_dot: np.ndarray,
    cand_cls: PlaneClassification,
    cand_clip,
    cand_cap,
    cand_dot: np.ndarray,
    dot_label: str,
    diff_policy: QuantizationPolicy | None,
) -> dict:
    cls_diff = diff_classifications(ref_cls, cand_cls)
    clip_diff = diff_clips(ref_clip, cand_clip, policy=diff_policy)
    cap_diff = diff_caps(ref_cap, cand_cap)

    first_divergence = None
    if not cls_diff.agrees:
        first_divergence = "classification"
    elif not clip_diff.agrees:
        first_divergence = "clip"
    elif not cap_diff.agrees:
        first_divergence = "cap"

    report = {
        "first_divergence": first_divergence,
        "classification_agrees": cls_diff.agrees,
        "clip_face_set_agrees": clip_diff.details.get("face_set_agrees", True),
        "cap_agrees": cap_diff.agrees,
    }
    if not cls_diff.agrees:
        disagreeing = _disagreeing_vertices(
            ref_cls.signs, ref_dot, cand_cls.signs, cand_dot, dot_label
        )
        report["classification_disagree_count"] = len(disagreeing)
        report["disagreeing_vertices"] = disagreeing
    return report


def _evaluate_raw_f32(
    vertices: np.ndarray,
    faces: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    ref_cls: PlaneClassification,
    ref_clip,
    ref_cap,
    ref_dot: np.ndarray,
) -> dict:
    vertices_f32 = vertices.astype(np.float32)
    point_f32 = point.astype(np.float32)
    normal_f32 = normal.astype(np.float32)

    cand_cls = _classify_raw_f32(vertices_f32, point_f32, normal_f32)
    cand_dot = _raw_f32_dot(vertices_f32, point_f32, normal_f32)
    cand_clip = _clip_mesh_generic(
        vertices_f32, faces, point_f32, normal_f32, _classify_raw_f32
    )
    cand_cap = extract_cap_f64(cand_clip)

    return _variant_report(
        ref_cls,
        ref_clip,
        ref_cap,
        ref_dot,
        cand_cls,
        cand_clip,
        cand_cap,
        cand_dot,
        "f32_dot",
        None,
    )


def _evaluate_grid_variant(
    vertices: np.ndarray,
    faces: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    policy: QuantizationPolicy,
    ref_cls: PlaneClassification,
    ref_clip,
    ref_cap,
    ref_dot: np.ndarray,
) -> dict:
    cand_cls = classify_plane_f32(vertices, point, normal, policy)
    cand_dot = _grid_dot(vertices, point, normal, policy)
    cand_clip = clip_mesh_f32(vertices, faces, point, normal, policy)
    cand_cap = extract_cap_f32(cand_clip, policy)

    return _variant_report(
        ref_cls,
        ref_clip,
        ref_cap,
        ref_dot,
        cand_cls,
        cand_clip,
        cand_cap,
        cand_dot,
        "grid_dot",
        policy,
    )


def _process_clip(fixture: str, clip_index: int) -> dict:
    vertices, faces, point, normal = _load_clip(fixture, clip_index)

    ref_cls = classify_plane_f64(vertices, point, normal)
    ref_clip = clip_mesh_f64(vertices, faces, point, normal)
    ref_cap = extract_cap_f64(ref_clip)
    ref_dot = np.dot(vertices - point, normal)

    variants = {
        "raw_f32": _evaluate_raw_f32(
            vertices, faces, point, normal, ref_cls, ref_clip, ref_cap, ref_dot
        ),
        "grid_no_snap": _evaluate_grid_variant(
            vertices,
            faces,
            point,
            normal,
            NO_SNAP_POLICY,
            ref_cls,
            ref_clip,
            ref_cap,
            ref_dot,
        ),
        "policy_0_1_0": _evaluate_grid_variant(
            vertices,
            faces,
            point,
            normal,
            DEFAULT_POLICY,
            ref_cls,
            ref_clip,
            ref_cap,
            ref_dot,
        ),
    }
    return {"fixture": fixture, "clip_index": clip_index, "variants": variants}


def _summarize(variant_name: str, clips: list[dict]) -> dict:
    classification_disagree = 0
    clip_disagree = 0
    cap_disagree = 0
    for clip in clips:
        v = clip["variants"][variant_name]
        if not v["classification_agrees"]:
            classification_disagree += 1
        if not v["clip_face_set_agrees"]:
            clip_disagree += 1
        if not v["cap_agrees"]:
            cap_disagree += 1
    return {
        "classification_disagree": classification_disagree,
        "clip_disagree": clip_disagree,
        "cap_disagree": cap_disagree,
    }


def _primary_divergence_class(clips: list[dict]) -> str:
    first_divergences = [
        clip["variants"]["policy_0_1_0"]["first_divergence"] for clip in clips
    ]
    disagreeing = [fd for fd in first_divergences if fd is not None]
    if disagreeing and all(fd == "classification" for fd in disagreeing):
        return "grid_quantization_classification"
    if not disagreeing:
        return "none"
    return "mixed"


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    clip_refs = manifest["clips"]

    clips: list[dict] = []
    current_fixture = None
    fixture_count = 0
    for entry in clip_refs:
        fixture = entry["fixture"]
        clip_index = entry["clip_index"]
        if fixture != current_fixture:
            if current_fixture is not None:
                print(f"{current_fixture}: {fixture_count} clips processed")
            current_fixture = fixture
            fixture_count = 0
        clips.append(_process_clip(fixture, clip_index))
        fixture_count += 1
    if current_fixture is not None:
        print(f"{current_fixture}: {fixture_count} clips processed")

    summary = {
        "raw_f32": _summarize("raw_f32", clips),
        "grid_no_snap": _summarize("grid_no_snap", clips),
        "policy_0_1_0": _summarize("policy_0_1_0", clips),
    }

    report = {
        "version": "1.0",
        "date": datetime.now(UTC).isoformat(),
        "policy": DEFAULT_POLICY.version,
        "total_clips": len(clips),
        "summary": summary,
        "primary_divergence_class": _primary_divergence_class(clips),
        "clips": clips,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print()
    print(f"total_clips={report['total_clips']}")
    for variant_name, counts in summary.items():
        print(f"{variant_name}: {counts}")
    print(f"primary_divergence_class={report['primary_divergence_class']}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
