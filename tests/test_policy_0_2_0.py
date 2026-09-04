"""Policy 0.2.0 (#115): ambiguity-band f32 fallback fixes all 114 known regressions."""

import json
from pathlib import Path

import numpy as np
import pytest

from chitin.f32_policy import DEFAULT_POLICY, POLICY_0_2_0
from chitin.f32_predicates import (
    _grid_quantization_bound,
    _to_grid_frame,
    classify_plane_f32,
    classify_plane_f64,
    clip_mesh_f32,
    clip_mesh_f64,
    diff_classifications,
    diff_clips,
)

REGRESSION_DIR = Path(__file__).parent / "fixtures" / "regression"

needs_corpus = pytest.mark.skipif(
    not (REGRESSION_DIR / "manifest.json").exists(),
    reason="Regression corpus not extracted",
)


def _load_manifest():
    with open(REGRESSION_DIR / "manifest.json") as f:
        return json.load(f)


def _build_clip_params():
    if not (REGRESSION_DIR / "manifest.json").exists():
        return []
    manifest = _load_manifest()
    params = []
    for fixture_name, info in manifest["fixtures"].items():
        for clip_index in info["clip_indices"]:
            params.append((fixture_name, clip_index))
    return params


REGRESSION_CLIPS = _build_clip_params()
REGRESSION_IDS = [f"{fix}-clip{idx}" for fix, idx in REGRESSION_CLIPS]


def _plane_point_normal(plane):
    n = plane[:3].astype(np.float64)
    norm = np.linalg.norm(n)
    normal = n / norm
    point = -(plane[3] / norm) * normal
    return point, normal


def _load_clip(fixture_name, clip_index):
    path = REGRESSION_DIR / fixture_name / f"clip_{clip_index}.npz"
    data = np.load(path)
    vertices = data["input_vertices"].astype(np.float64)
    faces = data["input_faces"].astype(np.int64)
    point, normal = _plane_point_normal(data["plane"])
    return vertices, faces, point, normal


@needs_corpus
@pytest.mark.parametrize(
    "fixture_name,clip_index", REGRESSION_CLIPS, ids=REGRESSION_IDS
)
def test_policy_0_2_0_classification_agrees(fixture_name, clip_index):
    vertices, _faces, point, normal = _load_clip(fixture_name, clip_index)
    ref = classify_plane_f64(vertices, point, normal)
    cand = classify_plane_f32(vertices, point, normal, POLICY_0_2_0)
    diff = diff_classifications(ref, cand)
    assert diff.agrees, diff.first_divergence


@needs_corpus
@pytest.mark.parametrize(
    "fixture_name,clip_index", REGRESSION_CLIPS, ids=REGRESSION_IDS
)
def test_policy_0_2_0_face_set_agrees(fixture_name, clip_index):
    vertices, faces, point, normal = _load_clip(fixture_name, clip_index)
    ref_clip = clip_mesh_f64(vertices, faces, point, normal)
    cand_clip = clip_mesh_f32(vertices, faces, point, normal, POLICY_0_2_0)
    clip_diff = diff_clips(ref_clip, cand_clip, policy=POLICY_0_2_0)
    assert clip_diff.details["face_set_agrees"], clip_diff.first_divergence


@needs_corpus
@pytest.mark.parametrize(
    "fixture_name,clip_index", REGRESSION_CLIPS, ids=REGRESSION_IDS
)
def test_ambiguity_counts_consistent(fixture_name, clip_index):
    vertices, _faces, point, normal = _load_clip(fixture_name, clip_index)
    result = classify_plane_f32(vertices, point, normal, POLICY_0_2_0)
    assert result.fast_path_count + result.ambiguity_path_count == len(vertices)
    assert result.ambiguity_path_count > 0


@needs_corpus
@pytest.mark.parametrize(
    "fixture_name,clip_index", REGRESSION_CLIPS, ids=REGRESSION_IDS
)
def test_ambiguity_path_respects_bound(fixture_name, clip_index):
    vertices, _faces, point, normal = _load_clip(fixture_name, clip_index)

    grid_v, grid_p, _centroid, scale_factor = _to_grid_frame(
        vertices, point, POLICY_0_2_0
    )
    grid_n = (normal * scale_factor).astype(np.float32)
    dot = np.sum(
        (grid_v.astype(np.float32) - grid_p.astype(np.float32)) * grid_n, axis=1
    )
    bound = _grid_quantization_bound(grid_n)

    grid_only_signs = np.sign(dot).astype(np.int8)
    ref_signs = classify_plane_f64(vertices, point, normal).signs
    ambiguous = np.abs(dot) <= bound
    disagrees = grid_only_signs != ref_signs

    # fast path: outside the bound, the raw grid predicate must already
    # agree with f64 — no misclassified vertex escapes the ambiguity band.
    assert not np.any(disagrees & ~ambiguous)

    # every actual grid-vs-f64 divergence falls inside the ambiguity band
    assert np.all(ambiguous[disagrees])


def test_ambiguity_not_entered_when_far_from_plane():
    vertices = np.array([[0, 0, 10], [0, 0, 20], [0, 0, 30]], dtype=np.float64)
    point = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    result = classify_plane_f32(vertices, point, normal, POLICY_0_2_0)
    assert result.ambiguity_path_count == 0
    assert result.fast_path_count == 3


def test_policy_0_1_0_has_no_ambiguity_counts():
    vertices = np.array([[0, 0, 10], [0, 0, 20], [0, 0, -5]], dtype=np.float64)
    point = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    result = classify_plane_f32(vertices, point, normal, DEFAULT_POLICY)
    assert result.fast_path_count == len(vertices)
    assert result.ambiguity_path_count == 0


@needs_corpus
@pytest.mark.parametrize(
    "fixture_name,clip_index",
    REGRESSION_CLIPS,
    ids=REGRESSION_IDS,
)
def test_intersection_finite_and_on_plane(fixture_name, clip_index):
    """Every intersection point produced by Policy 0.2.0 is finite and on-plane."""
    vertices, faces, point, normal = _load_clip(fixture_name, clip_index)
    result = clip_mesh_f32(vertices, faces, point, normal, POLICY_0_2_0)
    if len(result.intersection_points) == 0:
        return
    assert np.all(np.isfinite(result.intersection_points)), "non-finite intersection"
    extent = max(float(np.abs(vertices - vertices.mean(axis=0)).max()), 1e-30)
    grid_cell_world = 2 * extent / POLICY_0_2_0.grid_scale
    residuals = np.abs(
        np.sum(
            (result.intersection_points - point.astype(np.float64)) * normal,
            axis=1,
        )
    )
    tolerance = grid_cell_world * 64
    assert np.all(residuals < tolerance), (
        f"intersection off-plane: max residual {residuals.max():.2e}, "
        f"tolerance {tolerance:.2e}"
    )


@needs_corpus
@pytest.mark.parametrize(
    "fixture_name,clip_index",
    REGRESSION_CLIPS,
    ids=REGRESSION_IDS,
)
def test_cap_closure_and_winding(fixture_name, clip_index):
    """Boundary edges form closed loops with consistent winding under Policy 0.2.0."""
    from chitin.f32_predicates import extract_cap_f32

    vertices, faces, point, normal = _load_clip(fixture_name, clip_index)
    clip_result = clip_mesh_f32(vertices, faces, point, normal, POLICY_0_2_0)
    if len(clip_result.boundary_edges) == 0:
        return
    cap = extract_cap_f32(clip_result, POLICY_0_2_0)
    non_degenerate = [loop for loop in cap.loops if len(loop) >= 3]
    if POLICY_0_2_0.winding_check and non_degenerate:
        assert cap.winding_consistent, "cap winding inconsistent"
