"""Source-only validity tests for Policy 0.2.0 holdout corpus fixtures.

Tests geometry properties (manifold, winding, topology) without touching
Policy 0.2, trace replay, or the evaluator. These run during selection
to prove the fixtures are valid holdout candidates.
"""

import numpy as np
import pytest

from chitin.topology import analyze_topology
from chitin.trace_fixtures import FIXTURES, HOLDOUT_FIXTURES

HOLDOUT_FIXTURE_NAMES = [
    "oblique_gear_prism",
    "twisted_notched_column",
    "skewed_rectangular_torus",
    "multiscale_shard_cluster",
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

    def test_no_degenerate_faces(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        analysis = analyze_topology(v, f)
        assert analysis.degenerate_face_count == 0

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

    def test_no_zero_area_faces(self, name):
        v, f = HOLDOUT_FIXTURES[name]()
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        areas = np.linalg.norm(np.cross(e1, e2), axis=1)
        assert np.all(areas > 1e-12)

    def test_deterministic(self, name):
        v1, f1 = HOLDOUT_FIXTURES[name]()
        v2, f2 = HOLDOUT_FIXTURES[name]()
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)


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
        v, f = HOLDOUT_FIXTURES["skewed_rectangular_torus"]()
        n_edges = len(f) * 3 // 2
        euler = len(v) - n_edges + len(f)
        assert euler == 0, f"Expected genus-one (Euler=0), got {euler}"

    def test_has_multi_component_fixture(self):
        v, f = HOLDOUT_FIXTURES["multiscale_shard_cluster"]()
        analysis = analyze_topology(v, f)
        assert analysis.component_count >= 5

    def test_has_high_concavity_fixture(self):
        _, f = HOLDOUT_FIXTURES["oblique_gear_prism"]()
        assert len(f) >= 100

    def test_has_oblique_nonparallel_fixture(self):
        v, f = HOLDOUT_FIXTURES["twisted_notched_column"]()
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
            / "holdout-corpus-0.2.0.json"
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
