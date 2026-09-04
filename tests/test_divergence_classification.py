"""Regression test for #119: first-divergence classification of the 114 f32 failures.

Asserts that every regression clip's first divergence under Policy 0.1.0
is at the classification stage (grid quantization flips 1-2 near-plane
vertices), and that raw f32 (no grid) has zero divergence at any stage.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from chitin.f32_policy import POLICY_0_1_0
from chitin.f32_predicates import (
    PlaneClassification,
    _clip_mesh_generic,
    _count_signs,
    classify_plane_f32,
    classify_plane_f64,
    clip_mesh_f64,
    diff_classifications,
    diff_clips,
)

REGRESSION_DIR = Path(__file__).parent / "fixtures" / "regression"

pytestmark = pytest.mark.skipif(
    not (REGRESSION_DIR / "manifest.json").exists(),
    reason="Regression corpus not extracted (run scripts/extract_regression_corpus.py)",
)


def _load_manifest():
    with open(REGRESSION_DIR / "manifest.json") as f:
        return json.load(f)


def _plane_point_normal(plane):
    n = plane[:3].astype(np.float64)
    norm = np.linalg.norm(n)
    normal = n / norm
    point = -(plane[3] / norm) * normal
    return point, normal


def _classify_raw_f32(vertices, plane_point, plane_normal):
    dot = np.sum((vertices - plane_point) * plane_normal, axis=1)
    signs = np.sign(dot).astype(np.int8)
    pc, nc, oc = _count_signs(signs)
    return PlaneClassification(signs, pc, nc, oc)


def _build_clip_params():
    if not (REGRESSION_DIR / "manifest.json").exists():
        return []
    manifest = _load_manifest()
    return [(c["fixture"], c["clip_index"]) for c in manifest["clips"]]


REGRESSION_CLIPS = _build_clip_params()
REGRESSION_IDS = [f"{fix}-clip{idx}" for fix, idx in REGRESSION_CLIPS]


@pytest.mark.parametrize(
    "fixture,clip_index",
    REGRESSION_CLIPS,
    ids=REGRESSION_IDS,
)
def test_first_divergence_is_classification(fixture, clip_index):
    npz = np.load(REGRESSION_DIR / fixture / f"clip_{clip_index}.npz")
    verts = npz["input_vertices"].astype(np.float64)
    point, normal = _plane_point_normal(npz["plane"])

    ref_cls = classify_plane_f64(verts, point, normal)
    cand_cls = classify_plane_f32(verts, point, normal, POLICY_0_1_0)
    cls_diff = diff_classifications(ref_cls, cand_cls)

    assert not cls_diff.agrees, (
        f"{fixture}/clip_{clip_index}: grid classification agrees with f64 "
        f"(expected divergence from grid quantization)"
    )
    assert cls_diff.first_divergence is not None


@pytest.mark.parametrize(
    "fixture,clip_index",
    REGRESSION_CLIPS,
    ids=REGRESSION_IDS,
)
def test_raw_f32_has_no_divergence(fixture, clip_index):
    npz = np.load(REGRESSION_DIR / fixture / f"clip_{clip_index}.npz")
    verts = npz["input_vertices"].astype(np.float64)
    faces = npz["input_faces"].astype(np.int64)
    point, normal = _plane_point_normal(npz["plane"])

    ref_cls = classify_plane_f64(verts, point, normal)
    v32 = verts.astype(np.float32)
    p32 = point.astype(np.float32)
    n32 = normal.astype(np.float32)
    raw_cls = _classify_raw_f32(v32, p32, n32)

    cls_diff = diff_classifications(ref_cls, raw_cls)
    assert cls_diff.agrees, (
        f"{fixture}/clip_{clip_index}: raw f32 classification disagrees with f64 "
        f"(unexpected — grid quantization should be the sole cause)"
    )

    ref_clip = clip_mesh_f64(verts, faces, point, normal)
    raw_clip = _clip_mesh_generic(v32, faces, p32, n32, _classify_raw_f32)
    clip_diff = diff_clips(ref_clip, raw_clip)
    assert clip_diff.details["face_set_agrees"], (
        f"{fixture}/clip_{clip_index}: raw f32 face set disagrees with f64"
    )


def test_divergence_report_consistency():
    report_path = Path(__file__).parent.parent / "docs" / "divergence-report.json"
    if not report_path.exists():
        pytest.skip("divergence-report.json not generated")

    with open(report_path) as f:
        report = json.load(f)

    assert report["total_clips"] == 114
    assert report["primary_divergence_class"] == "grid_quantization_classification"
    assert report["summary"]["raw_f32"]["classification_disagree"] == 0
    assert report["summary"]["raw_f32"]["clip_disagree"] == 0
    assert report["summary"]["policy_0_1_0"]["classification_disagree"] == 114
    assert report["summary"]["policy_0_1_0"]["clip_disagree"] == 114

    for clip in report["clips"]:
        p = clip["variants"]["policy_0_1_0"]
        assert p["first_divergence"] == "classification", (
            f"{clip['fixture']}/clip_{clip['clip_index']}: "
            f"expected first_divergence='classification', got '{p['first_divergence']}'"
        )
        r = clip["variants"]["raw_f32"]
        assert r["first_divergence"] is None
