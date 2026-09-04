"""Arithmetic diagnosis invariants for the 14 Policy 0.2 holdout failures."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "diagnose_grid_boundary.py"
SPEC = importlib.util.spec_from_file_location("diagnose_grid_boundary", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSIS)


def test_all_holdout_failures_are_f32_input_precision_loss():
    report = DIAGNOSIS.diagnose_corpus(DIAGNOSIS.DEFAULT_CORPUS)

    assert report["clip_count"] == 14
    assert report["mismatching_vertex_count"] == 42
    assert report["causes"] == {"source_delta_below_f32_input_precision": 42}
    assert report["canonical_f32_clip_agree_count"] == 14


def test_every_mismatch_entered_policy_0_2_ambiguity_path():
    report = DIAGNOSIS.diagnose_corpus(DIAGNOSIS.DEFAULT_CORPUS)
    mismatches = [
        mismatch for clip in report["clips"] for mismatch in clip["mismatches"]
    ]

    assert mismatches
    assert all(mismatch["entered_ambiguity_path"] for mismatch in mismatches)
    assert all(
        mismatch["policy_sign"] == mismatch["f32_input_exact_sign"]
        for mismatch in mismatches
    )
