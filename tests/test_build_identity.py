"""Build identity: logical identity is stable and distinct from artifact digest."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from chitin import Config, extract_from_mesh
from chitin.provenance import (
    ALGORITHM_VERSION,
    REALIZATION_MANIFEST_VERSION,
    build_logical_identity,
    build_realization,
    build_realization_manifest,
    effective_input_digest,
    output_affecting_config_digest,
    realization_address,
)
from chitin.report import build_compilation_report


def _box_mesh():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


class TestEffectiveInputDigest:
    def test_same_geometry_same_digest(self):
        v, f = _box_mesh()
        assert effective_input_digest(v, f) == effective_input_digest(
            v.copy(), f.copy()
        )

    def test_different_geometry_different_digest(self):
        v, f = _box_mesh()
        v2 = v * 2.0
        assert effective_input_digest(v, f) != effective_input_digest(v2, f)

    def test_digest_format(self):
        v, f = _box_mesh()
        d = effective_input_digest(v, f)
        assert d.startswith("sha256:")
        assert len(d) == len("sha256:") + 64


class TestConfigDigest:
    def test_same_config_same_digest(self):
        c1 = Config()
        c2 = Config()
        assert output_affecting_config_digest(c1) == output_affecting_config_digest(c2)

    def test_different_output_config_different_digest(self):
        c1 = Config(concavity=0.01)
        c2 = Config(concavity=0.05)
        assert output_affecting_config_digest(c1) != output_affecting_config_digest(c2)

    def test_timeout_does_not_affect_digest(self):
        c1 = Config(coacd_timeout=10.0)
        c2 = Config(coacd_timeout=300.0)
        assert output_affecting_config_digest(c1) == output_affecting_config_digest(c2)

    def test_deterministic_flag_does_not_affect_digest(self):
        c1 = Config(coacd_deterministic=True)
        c2 = Config(coacd_deterministic=False)
        assert output_affecting_config_digest(c1) == output_affecting_config_digest(c2)


class TestLogicalBuildIdentity:
    def test_contains_all_fields(self):
        v, f = _box_mesh()
        identity = build_logical_identity(v, f, Config())
        assert set(identity.keys()) == {
            "effective_input_digest",
            "algorithm_version",
            "numerical_policy_version",
            "config_digest",
        }

    def test_versions_are_semver(self):
        v, f = _box_mesh()
        identity = build_logical_identity(v, f, Config())
        for key in ("algorithm_version", "numerical_policy_version"):
            parts = identity[key].split(".")
            assert len(parts) == 3
            assert all(p.isdigit() for p in parts)

    def test_stable_across_calls(self):
        v, f = _box_mesh()
        c = Config()
        id1 = build_logical_identity(v, f, c)
        id2 = build_logical_identity(v, f, c)
        assert id1 == id2


class TestRealization:
    def test_wraps_logical_identity(self):
        v, f = _box_mesh()
        logical = build_logical_identity(v, f, Config())
        real = build_realization(logical, "a" * 64)
        assert real["logical_build_identity"] == logical
        assert real["artifact_digest"] == f"sha256:{'a' * 64}"
        assert real["realization"]["runtime"]["kind"] == "python_native"


_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "lingot"
    / "contracts"
    / "realization_manifest.schema.json"
)


def _manifest(**kw):
    v, f = _box_mesh()
    logical = build_logical_identity(v, f, Config())
    return build_realization_manifest(logical, kw.pop("digest", "a" * 64), **kw)


class TestRealizationManifest:
    def test_validates_against_lingot_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        if not _SCHEMA_PATH.exists():
            pytest.skip(f"lingot schema not found at {_SCHEMA_PATH}")
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        full = _manifest(
            conformance={"profile": "walkable", "verdict_status": "pass"},
            platform_evidence={"os": "linux"},
        )
        jsonschema.validate(full, schema, format_checker=jsonschema.FormatChecker())
        jsonschema.validate(_manifest(), schema)
        assert full["realization_manifest_version"] == REALIZATION_MANIFEST_VERSION

    def test_bare_and_prefixed_digest_normalize(self):
        a = _manifest(digest="a" * 64, timestamp="2026-01-01T00:00:00Z")
        b = _manifest(digest="sha256:" + "a" * 64, timestamp="2026-01-01T00:00:00Z")
        assert a == b
        assert a["artifact_digest"] == f"sha256:{'a' * 64}"

    @pytest.mark.parametrize("bad", [None, "", "sha256:", "zz", "sha256:xyz"])
    def test_bad_digest_raises(self, bad):
        with pytest.raises(ValueError):
            _manifest(digest=bad)

    def test_conformance_included_or_omitted(self):
        assert "conformance" not in _manifest()
        c = {"profile": None, "verdict_status": "not_evaluated"}
        assert _manifest(conformance=c)["conformance"] == c
        with pytest.raises(ValueError):
            _manifest(conformance={"profile": "x", "verdict_status": "maybe"})
        with pytest.raises(ValueError):
            _manifest(conformance={"verdict_status": "pass"})

    def test_platform_evidence_only_when_given(self):
        assert "platform_evidence" not in _manifest()["realization"]

    def test_default_timestamp_is_utc_z(self):
        ts = _manifest()["timestamp"]
        assert ts.endswith("Z")

    def test_build_realization_unchanged(self):
        v, f = _box_mesh()
        logical = build_logical_identity(v, f, Config())
        assert build_realization(logical, None)["artifact_digest"] is None

    def test_address_matches_gesso(self, tmp_path):
        gesso_store = pytest.importorskip("gesso.store")
        m = _manifest(timestamp="2026-01-01T00:00:00Z")
        addr = gesso_store.LocalStore(tmp_path).put_json(m)
        assert realization_address(m) == addr


class TestReportBuildIdentity:
    def test_report_includes_build_identity(self):
        v, f = _box_mesh()
        cfg = Config()
        result = extract_from_mesh(v, f, config=cfg)
        identity = build_logical_identity(v, f, cfg)
        report = build_compilation_report(result, build_identity=identity)
        d = report.to_dict()
        assert "build_identity" in d
        assert d["build_identity"]["algorithm_version"] == ALGORITHM_VERSION
        assert d["build_identity"]["effective_input_digest"] is not None

    def test_report_defaults_to_null_digests(self):
        v, f = _box_mesh()
        result = extract_from_mesh(v, f, config=Config())
        report = build_compilation_report(result)
        d = report.to_dict()
        assert d["build_identity"]["effective_input_digest"] is None
        assert d["build_identity"]["config_digest"] is None
        assert d["build_identity"]["algorithm_version"] == ALGORITHM_VERSION
