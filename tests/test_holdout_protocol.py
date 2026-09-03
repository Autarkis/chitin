"""Tests for holdout protocol enforcement (#121).

Verifies that evaluate_holdout.py:
- rejects known corpus digests
- selects the correct policy from --policy flag
- versions output paths
- preserves 0.1.0 backward compatibility
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the evaluator module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
evaluate_holdout = importlib.import_module("evaluate_holdout")


class TestKnownDigestRejection:
    def test_known_digests_populated(self):
        assert len(evaluate_holdout.KNOWN_CORPUS_DIGESTS) >= 4

    def test_spent_external_tier_rejected(self):
        for digest in [
            "293790274a89a0c7549f6d86394017a2620fa95ccb71dbe7e52a26c85d10b202",
            "dce6de15b4b3560df0cb799803e84beaead93b1eafb8f55dc288a8e49c41ef14",
            "b42e20807a3cf4fc2b6d8048dfc434e83f13f137c4658951de5471a290fa6972",
        ]:
            assert digest in evaluate_holdout.KNOWN_CORPUS_DIGESTS

    def test_regression_manifest_rejected(self):
        assert (
            "c2311cfc0c026ee3e870c35ed8b295b1a455c1b1525e24e4b348db511d0ee92b"
            in evaluate_holdout.KNOWN_CORPUS_DIGESTS
        )

    def test_evaluate_fixture_rejects_known_digest(self, tmp_path):
        import numpy as np

        fixture_dir = tmp_path / "fake_fixture"
        fixture_dir.mkdir()
        npz_path = fixture_dir / "arrays.npz"
        np.savez(npz_path, dummy=np.array([1, 2, 3]))
        real_digest = evaluate_holdout._sha256_file(npz_path)

        with patch.object(
            evaluate_holdout,
            "KNOWN_CORPUS_DIGESTS",
            {real_digest},
        ):
            from chitin.f32_policy import DEFAULT_POLICY

            with pytest.raises(SystemExit, match="REJECTED"):
                evaluate_holdout._evaluate_fixture(
                    "fake_fixture", tmp_path, DEFAULT_POLICY
                )


class TestPolicyFlag:
    def _parse_args(self, *args):
        with patch("sys.argv", ["evaluate_holdout.py", *args]):
            return evaluate_holdout._parse_args()

    def test_default_policy_is_0_1_0(self):
        _args, policy = self._parse_args()
        assert policy.version == "0.1.0"
        assert not policy.ambiguity_fallback

    def test_policy_0_2_0_selected(self):
        _args, policy = self._parse_args("--policy", "0.2.0")
        assert policy.version == "0.2.0"
        assert policy.ambiguity_fallback

    def test_invalid_policy_rejected(self):
        with pytest.raises(SystemExit):
            self._parse_args("--policy", "0.3.0")


class TestOutputPath:
    def _parse_args(self, *args):
        with patch("sys.argv", ["evaluate_holdout.py", *args]):
            return evaluate_holdout._parse_args()

    def test_0_1_0_default_path(self):
        args, _ = self._parse_args()
        assert args.output == Path("docs/holdout-results.json")

    def test_0_2_0_versioned_path(self):
        args, _ = self._parse_args("--policy", "0.2.0")
        assert args.output == Path("docs/holdout-results-0.2.0.json")

    def test_explicit_output_overrides(self):
        args, _ = self._parse_args("--output", "custom.json")
        assert args.output == Path("custom.json")

    def test_explicit_output_with_policy(self):
        args, _ = self._parse_args("--policy", "0.2.0", "--output", "custom.json")
        assert args.output == Path("custom.json")


class TestPolicyRecordFields:
    def test_policy_fields_complete(self):
        from chitin.f32_policy import POLICY_0_2_0

        expected = {
            "version",
            "grid_bits",
            "grid_scale",
            "classification_ulp_margin",
            "intersection_snap_bits",
            "winding_check",
            "ambiguity_fallback",
        }
        record = {
            "version": POLICY_0_2_0.version,
            "grid_bits": POLICY_0_2_0.grid_bits,
            "grid_scale": POLICY_0_2_0.grid_scale,
            "classification_ulp_margin": POLICY_0_2_0.classification_ulp_margin,
            "intersection_snap_bits": POLICY_0_2_0.intersection_snap_bits,
            "winding_check": POLICY_0_2_0.winding_check,
            "ambiguity_fallback": POLICY_0_2_0.ambiguity_fallback,
        }
        assert set(record.keys()) == expected
        assert record["version"] == "0.2.0"
        assert record["ambiguity_fallback"] is True
