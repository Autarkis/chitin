"""Build identity: logical identity is stable and distinct from artifact digest."""

import numpy as np
import trimesh

from chitin import Config, extract_from_mesh
from chitin.provenance import (
    ALGORITHM_VERSION,
    build_logical_identity,
    build_realization,
    effective_input_digest,
    output_affecting_config_digest,
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
