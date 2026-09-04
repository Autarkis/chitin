"""Tests for holdout protocol enforcement (#121).

Verifies that evaluate_holdout.py:
- rejects known corpus digests
- selects the correct policy from --policy flag
- versions output paths
- preserves 0.1.0 backward compatibility
"""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the evaluator module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
evaluate_holdout = importlib.import_module("evaluate_holdout")


class TestKnownDigestRejection:
    def test_known_digests_populated(self):
        assert len(evaluate_holdout.KNOWN_CORPUS_DIGESTS) >= 11

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

    def test_ci_tier_rejected(self):
        for digest in [
            "f2778d3f5ddb58e309bf903667899940b9cbed192b9102791194d889697f125c",  # box
            "5ff57d55f916ed43e9c54c359f7bb2bde9545248e426625d26bfc853025e0e87",  # icosphere
            "874302dd2e001fb74d1235ba9302100464a78f246e4d48f831d013a9fddf57c5",  # thin_panel
            "48a45262c932e01d278544522f15d45d25b59a9f3bc920f5ddcbbdd0e8f68420",  # l_shape
            "1b1223a253fedc2a86e6c000d6a5d873bf586aab0bbdf04f5175bcae9bb36e40",  # thin_u_channel
            "fa2e275d0da0bc8c539b0b772e0795d3818f883af9a91c3b2a810544f174593e",  # cross_bracket
            "9f9be026aecacccb40891d0c60b1874b70fbcdbc9c8d972c4d9300fb4b0760c5",  # staircase
        ]:
            assert digest in evaluate_holdout.KNOWN_CORPUS_DIGESTS

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
        _args, policy = self._parse_args(
            "--policy", "0.2.0", "--corpus-manifest", "manifest.json"
        )
        assert policy.version == "0.2.0"
        assert policy.ambiguity_fallback

    def test_invalid_policy_rejected(self):
        with pytest.raises(SystemExit):
            self._parse_args("--policy", "0.99.0")

    def test_manifest_required_for_0_2_0(self):
        with pytest.raises(SystemExit, match="--corpus-manifest is required"):
            self._parse_args("--policy", "0.2.0")

    def test_manifest_required_for_0_3_0(self):
        with pytest.raises(SystemExit, match="--corpus-manifest is required"):
            self._parse_args("--policy", "0.3.0")


class TestOutputPath:
    def _parse_args(self, *args):
        with patch("sys.argv", ["evaluate_holdout.py", *args]):
            return evaluate_holdout._parse_args()

    def test_0_1_0_default_path(self):
        args, _ = self._parse_args()
        assert args.output == Path("docs/holdout-results.json")

    def test_0_2_0_versioned_path(self):
        args, _ = self._parse_args(
            "--policy", "0.2.0", "--corpus-manifest", "manifest.json"
        )
        assert args.output == Path("docs/holdout-results-0.2.0.json")

    def test_explicit_output_overrides(self):
        args, _ = self._parse_args("--output", "custom.json")
        assert args.output == Path("custom.json")

    def test_explicit_output_with_policy(self):
        args, _ = self._parse_args(
            "--policy",
            "0.2.0",
            "--output",
            "custom.json",
            "--corpus-manifest",
            "manifest.json",
        )
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


class TestOverwriteProtection:
    def test_rejects_existing_output(self, tmp_path):
        output = tmp_path / "existing.json"
        output.write_text("{}")
        with (
            patch("sys.argv", ["evaluate_holdout.py", "--output", str(output)]),
            pytest.raises(SystemExit, match="already exists"),
        ):
            evaluate_holdout.main()

    def test_force_allows_overwrite(self, tmp_path):
        output = tmp_path / "existing.json"
        output.write_text("{}")
        with patch(
            "sys.argv", ["evaluate_holdout.py", "--output", str(output), "--force"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                evaluate_holdout.main()
            assert "already exists" not in str(exc_info.value)

    def test_canonical_path_immutable(self):
        """Canonical output path cannot be overwritten even with --force."""
        canonical = Path("docs/holdout-results-0.2.0.json")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        pre_existing = canonical.read_bytes() if canonical.exists() else None
        canonical.write_text("{}")
        try:
            with (
                patch(
                    "sys.argv",
                    [
                        "evaluate_holdout.py",
                        "--policy",
                        "0.2.0",
                        "--corpus-manifest",
                        "manifest.json",
                        "--force",
                    ],
                ),
                pytest.raises(SystemExit, match="permanently immutable"),
            ):
                evaluate_holdout.main()
        finally:
            if pre_existing is not None:
                canonical.write_bytes(pre_existing)
            else:
                canonical.unlink(missing_ok=True)


class TestManifestLoading:
    def test_loads_fixture_names(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "fixtures": [
                        {"name": "fix_a", "source_digest": "abc"},
                        {"name": "fix_b", "source_digest": "def"},
                    ],
                }
            )
        )
        result = evaluate_holdout._load_corpus_manifest(manifest)
        names = [f["name"] for f in result["fixtures"]]
        assert names == ["fix_a", "fix_b"]

    def test_rejects_empty_fixtures(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "fixtures": [],
                }
            )
        )
        with pytest.raises(SystemExit, match="no fixtures"):
            evaluate_holdout._load_corpus_manifest(manifest)

    def test_rejects_missing_name(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "fixtures": [{"source_digest": "abc"}],
                }
            )
        )
        with pytest.raises(SystemExit, match="missing 'name'"):
            evaluate_holdout._load_corpus_manifest(manifest)

    def test_rejects_bad_schema_version(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "fixtures": [{"name": "x"}],
                }
            )
        )
        with pytest.raises(SystemExit, match="Unsupported manifest schema"):
            evaluate_holdout._load_corpus_manifest(manifest)


class TestVerdictComputation:
    def _make_results(
        self,
        cls_rate=1.0,
        oracle_rate=1.0,
        face_set_agree=100,
        face_set_total=100,
        disagree_clips=None,
    ):
        """Build a minimal results dict for verdict testing."""
        fixtures = [
            {
                "classification": {"agree": int(cls_rate * 1000), "total": 1000},
                "oracle": {"agree": int(oracle_rate * 10000), "total": 10000},
                "clip_topology": {
                    "face_set_agree": face_set_agree,
                    "total": face_set_total,
                },
                "classification_disagreements": disagree_clips or [],
                "intersection_error": {
                    "nan_residual_count": 0,
                    "inf_residual_count": 0,
                },
            }
        ]
        return {
            "fixtures": fixtures,
            "aggregate": {
                "classification_rate": cls_rate,
                "oracle_rate": oracle_rate,
            },
        }

    def test_pass_all_criteria(self):
        results = self._make_results()
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["verdict"] == "PASS"
        assert all(c["pass"] for c in verdict["checks"].values())

    def test_fail_classification(self):
        results = self._make_results(cls_rate=0.98)
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["verdict"] == "FAIL"
        assert not verdict["checks"]["classification_gte_99pct"]["pass"]

    def test_fail_oracle(self):
        results = self._make_results(oracle_rate=0.999)
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["verdict"] == "FAIL"
        assert not verdict["checks"]["oracle_gte_99_99pct"]["pass"]

    def test_fail_disagree_topology(self):
        bad_disagree = [
            {"clip_face_set_agrees": False},
            {"clip_face_set_agrees": True},
        ]
        results = self._make_results(disagree_clips=bad_disagree)
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["verdict"] == "FAIL"
        assert not verdict["checks"]["disagree_topology_100pct_face_set"]["pass"]

    def test_fail_stratified_topology(self):
        results = self._make_results(face_set_agree=90, face_set_total=100)
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["verdict"] == "FAIL"
        assert not verdict["checks"]["stratified_topology_gte_99pct"]["pass"]

    def test_pass_no_disagreements(self):
        results = self._make_results(disagree_clips=[])
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["checks"]["disagree_topology_100pct_face_set"]["pass"]

    def test_fail_invalid_geometry(self):
        results = self._make_results()
        # Inject NaN/Inf counts
        results["fixtures"][0]["intersection_error"] = {
            "nan_residual_count": 2,
            "inf_residual_count": 1,
        }
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["verdict"] == "FAIL"
        assert not verdict["checks"]["zero_invalid_geometry"]["pass"]
        assert verdict["checks"]["zero_invalid_geometry"]["nan_count"] == 2
        assert verdict["checks"]["zero_invalid_geometry"]["inf_count"] == 1

    def test_pass_zero_invalid_geometry(self):
        results = self._make_results()
        results["fixtures"][0]["intersection_error"] = {
            "nan_residual_count": 0,
            "inf_residual_count": 0,
        }
        verdict = evaluate_holdout._compute_verdict(results)
        assert verdict["checks"]["zero_invalid_geometry"]["pass"]


class TestCaptureRecordVerification:
    """Tests for evaluate_holdout._load_capture_record()."""

    def test_missing_record(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"schema_version": "1.0", "fixtures": [{"name": "x"}]}')
        with pytest.raises(SystemExit, match="Capture record not found"):
            evaluate_holdout._load_capture_record(tmp_path, manifest)

    def test_manifest_digest_mismatch(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"schema_version": "1.0", "fixtures": [{"name": "x"}]}')
        record = tmp_path / "capture-record.json"
        record.write_text(
            json.dumps({"manifest_digest": "wrong_digest", "fixtures": []})
        )
        with pytest.raises(SystemExit, match="manifest digest mismatch"):
            evaluate_holdout._load_capture_record(tmp_path, manifest)

    def test_valid_record(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"schema_version": "1.0", "fixtures": [{"name": "x"}]}')
        manifest_digest = evaluate_holdout._sha256_file(manifest)
        record = tmp_path / "capture-record.json"
        record.write_text(
            json.dumps(
                {
                    "manifest_digest": manifest_digest,
                    "fixtures": [{"name": "x", "trace_digest": "abc123"}],
                }
            )
        )
        result = evaluate_holdout._load_capture_record(tmp_path, manifest)
        assert result["manifest_digest"] == manifest_digest
