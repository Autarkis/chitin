"""Per-clip regression gate for the 114 known-failing clips (#118).

Every clip in this corpus changed clip connectivity under Policy 0.1.0
(f32 classification disagreement → face-set topology mismatch).

The current default must recover every clip: both oracle classification and
face-set topology are asserted per clip, with no aggregate acceptance floor.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chitin.coacd_trace import TracedClip, TracedPlane
from chitin.coacd_trace_replay import compare_oracle, replay_clip
from chitin.f32_policy import DEFAULT_POLICY

REGRESSION_DIR = Path(__file__).parent / "fixtures" / "regression"

pytestmark = pytest.mark.skipif(
    not (REGRESSION_DIR / "manifest.json").exists(),
    reason="Regression corpus not extracted (run scripts/extract_regression_corpus.py)",
)


def _load_manifest() -> dict:
    with open(REGRESSION_DIR / "manifest.json") as f:
        return json.load(f)


def _load_clip(fixture: str, clip_index: int) -> TracedClip:
    npz_path = REGRESSION_DIR / fixture / f"clip_{clip_index}.npz"
    npz = np.load(str(npz_path))
    plane_arr = npz["plane"]
    return TracedClip(
        component_id=int(npz["component_id"]),
        plane=TracedPlane(
            a=float(plane_arr[0]),
            b=float(plane_arr[1]),
            c=float(plane_arr[2]),
            d=float(plane_arr[3]),
            method="clip",
            index=-1,
        ),
        pos_verts=len(npz["pos_vertices"]),
        pos_faces=len(npz["pos_triangles"]),
        neg_verts=len(npz["neg_vertices"]),
        neg_faces=len(npz["neg_triangles"]),
        intersection_count=len(npz["cut_points"]),
        pos_vertices=npz["pos_vertices"],
        pos_triangles=npz["pos_triangles"],
        neg_vertices=npz["neg_vertices"],
        neg_triangles=npz["neg_triangles"],
        input_vertices=npz["input_vertices"],
        input_faces=npz["input_faces"],
        oracle_sides=npz["oracle_sides"],
        cut_points=npz["cut_points"],
    )


def _build_clip_params() -> list[tuple[str, int]]:
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
def test_regression_classification(fixture, clip_index):
    clip = _load_clip(fixture, clip_index)
    result = compare_oracle(clip, clip_index, DEFAULT_POLICY)
    assert result is not None, (
        f"{fixture} clip {clip_index}: compare_oracle returned None"
    )
    assert result.num_disagree == 0, (
        f"{fixture} clip {clip_index}: {result.num_disagree}/{result.num_vertices} "
        f"oracle classification disagreements"
    )


@pytest.mark.parametrize(
    "fixture,clip_index",
    REGRESSION_CLIPS,
    ids=REGRESSION_IDS,
)
def test_regression_topology(fixture, clip_index):
    clip = _load_clip(fixture, clip_index)
    result = replay_clip(clip, clip_index, DEFAULT_POLICY)
    assert result is not None, f"{fixture} clip {clip_index}: replay_clip returned None"
    assert result.clip_face_set_agrees, (
        f"{fixture} clip {clip_index}: face-set topology diverged"
    )


def test_regression_manifest_integrity():
    manifest = _load_manifest()
    assert manifest["total_clips"] == 114
    for entry in manifest["clips"]:
        npz_path = REGRESSION_DIR / entry["fixture"] / f"clip_{entry['clip_index']}.npz"
        assert npz_path.exists(), f"Missing: {npz_path}"
        h = hashlib.sha256()
        with open(npz_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        assert h.hexdigest() == entry["npz_sha256"], (
            f"{entry['fixture']}/clip_{entry['clip_index']}.npz: "
            f"digest mismatch (manifest: {entry['npz_sha256'][:16]}..., "
            f"actual: {h.hexdigest()[:16]}...)"
        )
