"""Regression tests for Policy 0.2.0 holdout classifier-failure clips."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chitin.f32_policy import QuantizationPolicy
from chitin.f32_predicates import classify_plane_f32

FAILURES_DIR = Path(__file__).parent / "fixtures" / "traces" / "holdout_failures_0_2_0"
MANIFEST_PATH = FAILURES_DIR / "manifest.json"

POLICY_0_2_0 = QuantizationPolicy(
    grid_bits=20,
    classification_ulp_margin=0,
    intersection_snap_bits=20,
    winding_check=True,
    ambiguity_fallback=True,
)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _clip_ids():
    return [c["file"] for c in _load_manifest()["clips"]]


def _clip_entry(file_name):
    return next(c for c in _load_manifest()["clips"] if c["file"] == file_name)


def _point_and_normal(npz_data):
    raw_normal = npz_data["plane_normal"].astype(np.float64)
    offset = float(npz_data["plane_offset"])
    normal = raw_normal / np.linalg.norm(raw_normal)
    point = normal * (-offset / np.dot(normal, normal))
    return point, normal


@pytest.fixture(scope="module")
def manifest():
    return _load_manifest()


@pytest.fixture(scope="module")
def clip_entries(manifest):
    return manifest["clips"]


class TestIntegrity:
    def test_manifest_exists(self):
        assert MANIFEST_PATH.exists()

    def test_clip_count(self, clip_entries):
        assert len(clip_entries) == 14

    def test_all_files_present(self, clip_entries):
        for clip in clip_entries:
            assert (FAILURES_DIR / clip["file"]).exists()

    @pytest.mark.parametrize("clip_file", _clip_ids())
    def test_npz_digests(self, clip_file):
        clip = _clip_entry(clip_file)
        digest = _sha256_file(FAILURES_DIR / clip_file)
        assert digest == clip["npz_sha256"]

    @pytest.mark.parametrize("clip_file", _clip_ids())
    def test_npz_contents(self, clip_file):
        clip = _clip_entry(clip_file)
        data = np.load(FAILURES_DIR / clip_file)
        assert set(data.files) >= {
            "vertices",
            "faces",
            "oracle_sides",
            "plane_normal",
            "plane_offset",
        }

        vertices = data["vertices"]
        faces = data["faces"]
        oracle_sides = data["oracle_sides"]

        n = vertices.shape[0]
        assert vertices.shape == (n, 3)
        assert vertices.dtype == np.float64
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert faces.dtype == np.int32
        assert oracle_sides.shape == (n,)
        assert oracle_sides.dtype == np.int8
        assert n == clip["num_vertices"]

    def test_results_digest(self, manifest):
        results_path = (
            FAILURES_DIR.parent.parent.parent.parent
            / "docs"
            / "holdout-results-0.2.0.json"
        )
        digest = _sha256_file(results_path)
        assert digest == manifest["results_digest"]


class TestReproduction:
    @pytest.mark.parametrize("clip_file", _clip_ids())
    def test_f32_f64_divergence(self, clip_file):
        data = np.load(FAILURES_DIR / clip_file)
        vertices = data["vertices"].astype(np.float64)
        point, normal = _point_and_normal(data)

        result = classify_plane_f32(vertices, point, normal, POLICY_0_2_0)
        f32_signs = result.signs

        raw = np.dot(vertices - point, normal)
        f64_signs = np.sign(raw).astype(np.int8)
        f64_signs[f64_signs == 0] = 1

        assert np.any(f32_signs != f64_signs)

    @pytest.mark.parametrize("clip_file", _clip_ids())
    def test_divergence_is_near_plane(self, clip_file):
        data = np.load(FAILURES_DIR / clip_file)
        vertices = data["vertices"].astype(np.float64)
        point, normal = _point_and_normal(data)

        result = classify_plane_f32(vertices, point, normal, POLICY_0_2_0)
        f32_signs = result.signs

        raw = np.dot(vertices - point, normal)
        f64_signs = np.sign(raw).astype(np.int8)
        f64_signs[f64_signs == 0] = 1

        diverge_mask = f32_signs != f64_signs
        assert np.any(diverge_mask)
        assert np.all(np.abs(raw[diverge_mask]) < 1e-3)
