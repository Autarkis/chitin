"""Source-only validity tests for Policy 0.3.0 holdout corpus fixtures.

Tests geometry properties (manifold, winding, topology) without touching
Policy 0.3, trace replay, or the evaluator. These run during selection
to prove the fixtures are valid holdout candidates.
"""

import numpy as np
import pytest

from chitin.topology import analyze_topology
from chitin.trace_fixtures import FIXTURES, HOLDOUT_FIXTURES

HOLDOUT_FIXTURE_NAMES = [
    "barbed_helix_prism",
    "fluted_twist_column",
    "ridged_torus",
    "interlocked_frame",
    "barbed_helix_prism_offset",
    "fluted_twist_column_offset",
    "ridged_torus_offset",
]


@pytest.mark.parametrize("name", HOLDOUT_FIXTURE_NAMES)
class TestHoldoutFixtureValidity:
    def test_closed(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        analysis = analyze_topology(v, f)
        assert analysis.closed

    def test_two_manifold(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        analysis = analyze_topology(v, f)
        assert analysis.two_manifold

    def test_consistently_oriented(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        analysis = analyze_topology(v, f)
        assert analysis.consistently_oriented

    def test_no_boundary_edges(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        analysis = analyze_topology(v, f)
        assert analysis.boundary_edge_count == 0

    def test_no_non_manifold_edges(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        analysis = analyze_topology(v, f)
        assert analysis.non_manifold_edge_count == 0

    def test_finite_vertices(self, name):
        v, _ = HOLDOUT_FIXTURES[name]()
        assert np.all(np.isfinite(v))

    def test_deterministic(self, name):
        v1, f1 = HOLDOUT_FIXTURES[name]()
        v2, f2 = HOLDOUT_FIXTURES[name]()
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)

    def test_no_degenerate_faces(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        analysis = analyze_topology(v, f)
        assert analysis.degenerate_face_count == 0

    def test_no_zero_area_faces(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        areas = np.linalg.norm(np.cross(e1, e2), axis=1)
        assert np.all(areas > 1e-12)


class TestCorpusDiversity:
    def test_at_least_three_topology_families(self):
        families = set()
        for name in HOLDOUT_FIXTURE_NAMES:
            v, f = HOLDOUT_FIXTURES[name]()
            analysis = analyze_topology(v, f)
            n_edges = len(f) * 3 // 2
            euler = len(v) - n_edges + len(f)
            families.add((analysis.component_count, euler))
        assert len(families) >= 3

    def test_has_genus_one_fixture(self):
        v, f = HOLDOUT_FIXTURES["ridged_torus"]()
        n_edges = len(f) * 3 // 2
        euler = len(v) - n_edges + len(f)
        assert euler == 0, f"Expected genus-one (Euler=0), got {euler}"

    def test_has_multi_component_fixture(self):
        v, f = HOLDOUT_FIXTURES["interlocked_frame"]()
        analysis = analyze_topology(v, f)
        assert analysis.component_count >= 5

    def test_has_high_concavity_fixture(self):
        _, f = HOLDOUT_FIXTURES["barbed_helix_prism"]()
        assert len(f) >= 100

    def test_has_oblique_nonparallel_fixture(self):
        v, f = HOLDOUT_FIXTURES["fluted_twist_column"]()
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        oblique = np.all(np.abs(normals) > 0.05, axis=1)
        assert np.sum(oblique) > 0

    def test_holdout_not_in_fixtures(self):
        """Normal corpus gate cannot enumerate holdout fixtures."""
        overlap = set(FIXTURES) & set(HOLDOUT_FIXTURES)
        assert not overlap, f"Holdout fixtures leaked into FIXTURES: {overlap}"

    def test_manifest_matches_holdout_registry(self):
        """Manifest fixture names exactly match HOLDOUT_FIXTURES keys."""
        import json
        from pathlib import Path

        manifest_path = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "holdout-corpus-0.3.0.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_names = {entry["name"] for entry in manifest["fixtures"]}
        registry_names = set(HOLDOUT_FIXTURES)
        assert manifest_names == registry_names, (
            f"Manifest/registry mismatch: "
            f"manifest-only={manifest_names - registry_names}, "
            f"registry-only={registry_names - manifest_names}"
        )

    def test_matched_pairs_differ_only_by_translation(self):
        """Ordinary and offset fixtures produce identical faces; vertices differ by translation only."""
        import numpy as np

        from chitin.trace_fixtures import HOLDOUT_FIXTURES

        pairs = [
            ("barbed_helix_prism", "barbed_helix_prism_offset"),
            ("fluted_twist_column", "fluted_twist_column_offset"),
            ("ridged_torus", "ridged_torus_offset"),
        ]
        offset = np.array([1e7, 5e6, 3e6], dtype=np.float64)
        for base_name, offset_name in pairs:
            v_base, f_base = HOLDOUT_FIXTURES[base_name]()
            v_off, f_off = HOLDOUT_FIXTURES[offset_name]()
            np.testing.assert_array_equal(
                f_base, f_off, err_msg=f"{base_name} faces mismatch"
            )
            assert v_off.dtype == np.float64
            assert v_base.dtype == np.float32
            # Vertices should differ by exactly the offset (in float64)
            v_base_f64 = v_base.astype(np.float64)
            v_off_f64 = v_off.astype(np.float64)
            diff = v_off_f64 - v_base_f64
            np.testing.assert_array_equal(
                diff,
                np.broadcast_to(offset, diff.shape),
                err_msg=f"{base_name} offset mismatch",
            )

    def test_strata_coverage(self):
        """Both ordinary and large-offset strata are represented."""
        ordinary = {n for n in HOLDOUT_FIXTURE_NAMES if not n.endswith("_offset")}
        large_offset = {n for n in HOLDOUT_FIXTURE_NAMES if n.endswith("_offset")}
        assert len(ordinary) >= 3, "Need ≥3 ordinary fixtures"
        assert len(large_offset) >= 2, "Need ≥2 large-offset fixtures"


OFFSET_FIXTURE_NAMES = [
    "barbed_helix_prism_offset",
    "fluted_twist_column_offset",
    "ridged_torus_offset",
]


@pytest.mark.parametrize("name", OFFSET_FIXTURE_NAMES)
class TestOffsetCanonPrecisionLoss:
    """Prove f32 canonicalization — not fixture generation — causes precision loss."""

    def test_source_dtype_is_float64(self, name):
        v, _ = HOLDOUT_FIXTURES[name]()
        assert v.dtype == np.float64

    def test_canon_f32_loses_precision(self, name):
        v, _ = HOLDOUT_FIXTURES[name]()
        v_canon = v.astype(np.float32).astype(np.float64)
        assert not np.array_equal(v, v_canon), (
            f"{name}: no precision loss from f32 cast"
        )

    def test_canon_f32_collapses_vertices(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        v_canon = v.astype(np.float32).astype(np.float64)
        e1 = v_canon[f[:, 1]] - v_canon[f[:, 0]]
        e2 = v_canon[f[:, 2]] - v_canon[f[:, 0]]
        areas = np.linalg.norm(np.cross(e1, e2), axis=1)
        degenerate_count = np.sum(areas < 1e-12)
        assert degenerate_count > 0, (
            f"{name}: f32 canon did not produce any degenerate faces"
        )
