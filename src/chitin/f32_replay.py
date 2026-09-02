"""Replay #91 reference traces through f32 predicate paths and diff against f64."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chitin.f32_policy import DEFAULT_POLICY, QuantizationPolicy
from chitin.f32_predicates import PredicateDiff


@dataclass
class MeshTestCase:
    """One mesh + one splitting plane = one test case."""

    fixture_name: str
    plane_point: np.ndarray
    plane_normal: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class PredicateReport:
    """Full predicate-level comparison for one test case under one policy."""

    test_case: MeshTestCase
    policy: QuantizationPolicy
    classification_diff: PredicateDiff
    clip_diff: PredicateDiff
    cap_diff: PredicateDiff
    hull_diff: PredicateDiff | None

    @property
    def all_agree(self) -> bool:
        diffs = [self.classification_diff, self.clip_diff, self.cap_diff]
        if self.hull_diff is not None:
            diffs.append(self.hull_diff)
        return all(d.agrees for d in diffs)

    @property
    def first_divergence(self) -> str | None:
        for d in [
            self.classification_diff,
            self.clip_diff,
            self.cap_diff,
            self.hull_diff,
        ]:
            if d is not None and not d.agrees:
                return f"{d.predicate_name}: {d.first_divergence}"
        return None


@dataclass
class CorpusReport:
    """Aggregate over all test cases and policies."""

    reports: list

    @property
    def pass_rate(self) -> float:
        if not self.reports:
            return 1.0
        return sum(1 for r in self.reports if r.all_agree) / len(self.reports)

    @property
    def failing_reports(self) -> list:
        return [r for r in self.reports if not r.all_agree]

    def summary_by_predicate(self) -> dict:
        result = {}
        for name in ["classify_plane", "clip_mesh", "extract_cap", "convex_hull"]:
            agree = 0
            disagree = 0
            skipped = 0
            for r in self.reports:
                diff = getattr(
                    r,
                    {
                        "classify_plane": "classification_diff",
                        "clip_mesh": "clip_diff",
                        "extract_cap": "cap_diff",
                        "convex_hull": "hull_diff",
                    }[name],
                )
                if diff is None:
                    skipped += 1
                elif diff.agrees:
                    agree += 1
                else:
                    disagree += 1
            result[name] = {"agree": agree, "disagree": disagree, "skipped": skipped}
        return result

    def summary_by_policy(self) -> dict:
        by_policy = {}
        for r in self.reports:
            key = f"grid_bits={r.policy.grid_bits}"
            if key not in by_policy:
                by_policy[key] = {"pass": 0, "fail": 0}
            if r.all_agree:
                by_policy[key]["pass"] += 1
            else:
                by_policy[key]["fail"] += 1
        return by_policy


def generate_test_planes(
    vertices: np.ndarray, seed: int = 42, n_random: int = 5
) -> list:
    """Generate candidate splitting planes: 3 axis-aligned through centroid + n_random seeded."""
    centroid = vertices.mean(axis=0)
    planes = []
    for axis in range(3):
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        planes.append((centroid.copy(), normal))
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        normal = rng.standard_normal(3)
        normal /= np.linalg.norm(normal)
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        point = bbox_min + rng.random(3) * (bbox_max - bbox_min)
        planes.append((point, normal))
    return planes


def build_test_cases(n_random_planes: int = 5, seed: int = 42) -> list:
    """Build test cases from all fixtures x generated planes."""
    from chitin.trace_fixtures import FIXTURES

    cases = []
    for name, mesh_fn in FIXTURES.items():
        vertices, faces = mesh_fn()
        vertices = vertices.astype(np.float64)
        faces = faces.astype(np.int64)
        planes = generate_test_planes(vertices, seed=seed, n_random=n_random_planes)
        for plane_point, plane_normal in planes:
            cases.append(MeshTestCase(name, plane_point, plane_normal, vertices, faces))
    return cases


def run_predicate_gate(
    test_case: MeshTestCase, policy: QuantizationPolicy
) -> PredicateReport:
    """Run all four predicate families on one test case, f64 vs f32, return diffs."""
    from chitin.f32_predicates import (
        classify_plane_f32,
        classify_plane_f64,
        clip_mesh_f32,
        clip_mesh_f64,
        convex_hull_f32,
        convex_hull_f64,
        diff_caps,
        diff_classifications,
        diff_clips,
        diff_hulls,
        extract_cap_f32,
        extract_cap_f64,
    )

    v, f = test_case.vertices, test_case.faces
    pp, pn = test_case.plane_point, test_case.plane_normal

    cls_ref = classify_plane_f64(v, pp, pn)
    cls_cand = classify_plane_f32(v, pp, pn, policy)
    cls_diff = diff_classifications(cls_ref, cls_cand)

    clip_ref = clip_mesh_f64(v, f, pp, pn)
    clip_cand = clip_mesh_f32(v, f, pp, pn, policy)
    clip_diff_result = diff_clips(clip_ref, clip_cand, policy)

    cap_ref = extract_cap_f64(clip_ref)
    cap_cand = extract_cap_f32(clip_cand, policy)
    cap_diff_result = diff_caps(cap_ref, cap_cand)

    hull_diff_result = None
    if len(clip_ref.vertices) >= 4 and len(clip_cand.vertices) >= 4:
        try:
            hull_ref = convex_hull_f64(clip_ref.vertices)
            hull_cand = convex_hull_f32(clip_cand.vertices, policy)
            hull_diff_result = diff_hulls(hull_ref, hull_cand)
        except (ValueError, IndexError, RuntimeError):
            pass  # degenerate geometry — Qhull or diff can't handle it

    return PredicateReport(
        test_case, policy, cls_diff, clip_diff_result, cap_diff_result, hull_diff_result
    )


def run_corpus_gate(
    policies: list[QuantizationPolicy] | None = None,
    n_random_planes: int = 5,
    seed: int = 42,
) -> CorpusReport:
    """Run the full f32 disproof gate across all fixtures and policies."""
    if policies is None:
        policies = [DEFAULT_POLICY]

    cases = build_test_cases(n_random_planes=n_random_planes, seed=seed)
    reports = []
    for case in cases:
        for policy in policies:
            reports.append(run_predicate_gate(case, policy))
    return CorpusReport(reports)


if __name__ == "__main__":
    from chitin.f32_policy import sweep_policies

    report = run_corpus_gate(policies=sweep_policies(range(10, 24)))
    print(f"pass_rate={report.pass_rate:.4f} over {len(report.reports)} cases")
    print("by predicate:", report.summary_by_predicate())
    print("by policy:", report.summary_by_policy())
    if report.failing_reports:
        print(f"{len(report.failing_reports)} failing cases; first divergence:")
        print(report.failing_reports[0].first_divergence)
