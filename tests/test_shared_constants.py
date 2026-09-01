import json
from pathlib import Path

from chitin._shared_constants import (
    ACCEPTANCE_THRESHOLDS,
    COACD_CONCAVITY_THRESHOLD,
    COACD_PREPROCESS_RESOLUTION,
    INTERACTIVE_MIN_HULL_VERTICES,
    NATIVE_MIN_HULL_VERTICES,
    PROFILE_NAMES,
)


def test_generated_constants_match_shared_contract():
    contract_path = Path(__file__).parents[1] / "docs" / "shared-constants.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert COACD_CONCAVITY_THRESHOLD == contract["coacd"]["concavity_threshold"]
    assert COACD_PREPROCESS_RESOLUTION == contract["coacd"]["preprocess_resolution"]
    assert NATIVE_MIN_HULL_VERTICES == contract["hull"]["native_min_vertices"]
    assert INTERACTIVE_MIN_HULL_VERTICES == contract["hull"]["interactive_min_vertices"]
    assert PROFILE_NAMES == tuple(contract["profiles"])
    assert ACCEPTANCE_THRESHOLDS == contract["acceptance_thresholds"]
