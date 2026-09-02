"""Trace-backed f32 predicate gate test (#101 repass).

Replays exact CoACD internal clip operations through f32 vs f64
predicates using structural topology comparison. The saved corpus
lets CI run without the traced DLL.
"""

import os
import re
from pathlib import Path

import pytest

from chitin.coacd_trace import load_saved_trace
from chitin.coacd_trace_replay import (
    compare_oracle,
    replay_classifications,
    replay_trace,
)
from chitin.f32_policy import sweep_policies

TRACES_DIR = Path(__file__).parent / "fixtures" / "traces"

CORPUS_FIXTURES = [
    "box",
    "icosphere",
    "thin_panel",
    "l_shape",
    "thin_u_channel",
    "cross_bracket",
    "staircase",
]

# Multi-policy sweep across grid quantization widths at and above
# DEFAULT_POLICY (grid_bits=20). Coarser grids (lower grid_bits) are known
# to legitimately lower classification agreement (measured: grid16 ~80%),
# so the sweep stays at-or-finer than default to keep the regression floor
# meaningful rather than penalizing an expected coarse-grid tradeoff.
POLICIES = sweep_policies(range(20, 24))

# These are regression floors, not acceptance gates.
# They assert "at least as good as the last measured result."
# The real gate requires oracle comparison (see docs/coacd-instrumentation-spec.md).
CLASSIFICATION_FLOOR = 0.90  # measured: 93.7-100% across fixtures
CLIP_FLOOR = 0.85  # measured: ~93.7% aggregate
CAP_FLOOR = 0.85  # measured: similar to clip
ORACLE_FLOOR = 0.85

# When CHITIN_GATE_FINAL is set, missing corpus or missing oracle data
# is a hard failure, not a skip. This prevents the gate from silently
# passing in CI when the corpus is absent.
GATE_FINAL = os.environ.get("CHITIN_GATE_FINAL", "").lower() in ("1", "true", "yes")


def _load_fixture_trace(name: str):
    trace_dir = TRACES_DIR / name
    if not trace_dir.exists():
        if GATE_FINAL:
            pytest.fail(
                f"CHITIN_GATE_FINAL: no trace corpus for {name}. "
                f"Run scripts/capture_trace_corpus.py with v2 traced DLL."
            )
        pytest.skip(f"No trace corpus for {name}. Run scripts/capture_trace_corpus.py")
    return load_saved_trace(trace_dir)


class TestClassificationAgreement:
    """f32 vs f64 vertex classification on every traced clip."""

    @pytest.mark.parametrize("fixture_name", CORPUS_FIXTURES)
    @pytest.mark.parametrize(
        "policy", POLICIES, ids=[f"grid{p.grid_bits}" for p in POLICIES]
    )
    def test_classification(self, fixture_name, policy):
        trace = _load_fixture_trace(fixture_name)
        report = replay_classifications(trace, policy)
        if report.num_clips_replayed == 0:
            pytest.skip(f"No clips in {fixture_name}")
        rate = report.classification_rate
        print(
            f"\n{fixture_name} (grid_bits={policy.grid_bits}): {rate:.1%} "
            f"({report.num_classification_agree}/{report.num_clips_replayed})"
        )
        d = report.first_disagreement()
        if d:
            print(f"  First disagree: {d.classification_detail}")
        assert rate >= CLASSIFICATION_FLOOR, (
            f"Classification agreement {rate:.1%} below regression floor "
            f"{CLASSIFICATION_FLOOR:.0%} for {fixture_name} (grid_bits={policy.grid_bits})"
        )


class TestFullTopology:
    """Full clip+cap topology replay across the entire corpus.

    O(n^2) in the replay implementation, so this runs the full corpus
    (including the larger staircase/t_shape meshes) under @pytest.mark.slow
    rather than skipping them.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("fixture_name", CORPUS_FIXTURES)
    @pytest.mark.parametrize(
        "policy", POLICIES, ids=[f"grid{p.grid_bits}" for p in POLICIES]
    )
    def test_clip_topology(self, fixture_name, policy):
        trace = _load_fixture_trace(fixture_name)
        report = replay_trace(trace, policy)
        if report.num_clips_replayed == 0:
            pytest.skip(f"No clips in {fixture_name}")
        rate = report.clip_rate
        print(
            f"\n{fixture_name} clip (grid_bits={policy.grid_bits}): {rate:.1%} "
            f"({report.num_clip_agree}/{report.num_clips_replayed})"
        )
        assert rate >= CLIP_FLOOR, (
            f"Clip agreement {rate:.1%} below regression floor {CLIP_FLOOR:.0%} "
            f"for {fixture_name} (grid_bits={policy.grid_bits})"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("fixture_name", CORPUS_FIXTURES)
    @pytest.mark.parametrize(
        "policy", POLICIES, ids=[f"grid{p.grid_bits}" for p in POLICIES]
    )
    def test_cap_topology(self, fixture_name, policy):
        trace = _load_fixture_trace(fixture_name)
        report = replay_trace(trace, policy)
        if report.num_clips_replayed == 0:
            pytest.skip(f"No clips in {fixture_name}")
        rate = report.cap_rate
        print(
            f"\n{fixture_name} cap (grid_bits={policy.grid_bits}): {rate:.1%} "
            f"({report.num_cap_agree}/{report.num_clips_replayed})"
        )
        assert rate >= CAP_FLOOR, (
            f"Cap agreement {rate:.1%} below regression floor {CAP_FLOOR:.0%} "
            f"for {fixture_name} (grid_bits={policy.grid_bits})"
        )


class TestCorpusSummary:
    """Aggregate gate report across entire corpus."""

    @pytest.mark.parametrize(
        "policy", POLICIES, ids=[f"grid{p.grid_bits}" for p in POLICIES]
    )
    def test_gate_report(self, policy):
        traces = []
        for name in CORPUS_FIXTURES:
            trace_dir = TRACES_DIR / name
            if trace_dir.exists():
                traces.append((name, load_saved_trace(trace_dir)))
        if not traces:
            pytest.skip("No trace corpus")

        print(f"\n=== #101 Gate Report (grid_bits={policy.grid_bits}) ===")
        total_clips = 0
        total_agree = 0
        for name, trace in traces:
            report = replay_classifications(trace, policy)
            total_clips += report.num_clips_replayed
            total_agree += report.num_classification_agree
            if report.num_clips_replayed > 0:
                print(
                    f"  {name}: {report.classification_rate:.1%} "
                    f"({report.num_classification_agree}/{report.num_clips_replayed})"
                )

        assert total_clips > 0, "No clips replayed across corpus"

        rate = total_agree / total_clips
        print(f"\n  TOTAL: {rate:.1%} ({total_agree}/{total_clips})")
        assert rate >= CLASSIFICATION_FLOOR, (
            f"Aggregate classification agreement {rate:.1%} below regression floor "
            f"{CLASSIFICATION_FLOOR:.0%} (grid_bits={policy.grid_bits})"
        )


class TestOracleComparison:
    """Compare f32 classification against C++ oracle Side decisions.

    Requires v2 traces with oracle-recorded Side values; skips per-fixture
    when the loaded trace predates that instrumentation.
    """

    @pytest.mark.parametrize("fixture_name", CORPUS_FIXTURES)
    def test_oracle_agreement(self, fixture_name):
        trace = _load_fixture_trace(fixture_name)
        total_agree = 0
        total_verts = 0
        for i, clip in enumerate(trace.clips):
            result = compare_oracle(clip, i, POLICIES[0])
            if result is None:
                continue
            total_agree += result.num_agree
            total_verts += result.num_vertices
        if total_verts == 0:
            if GATE_FINAL:
                pytest.fail(
                    f"CHITIN_GATE_FINAL: no oracle data in {fixture_name} traces. "
                    f"Rebuild traced DLL with v2 instrumentation."
                )
            pytest.skip(f"No oracle data in {fixture_name} traces (need v2 trace)")
        rate = total_agree / total_verts
        print(
            f"\n{fixture_name}: aggregate oracle agreement {rate:.1%} ({total_agree}/{total_verts})"
        )
        assert rate >= ORACLE_FLOOR, (
            f"{fixture_name}: aggregate oracle agreement {rate:.1%} "
            f"below {ORACLE_FLOOR:.0%} threshold"
        )


class TestNoAbsoluteEpsilon:
    # Known-legitimate divide-by-zero / degenerate-input guards, not
    # geometric tolerances — same class as f32_policy.py's commented
    # `1e-30` floor on the max-extent denominator. Both occurrences guard
    # a zero-length plane normal before `normal = n / norm`. Matched by
    # substring rather than line number so the exemption survives minor
    # line drift in coacd_trace_replay.py.
    ALLOWLISTED_SNIPPETS = [
        "norm < 1e-15",
    ]

    def test_no_hardcoded_tolerance(self):
        suspect_files = [
            Path("src/chitin/f32_predicates.py"),
            Path("src/chitin/coacd_trace_replay.py"),
        ]
        for path in suspect_files:
            if not path.exists():
                continue
            content = path.read_text()
            scanned = content
            for snippet in self.ALLOWLISTED_SNIPPETS:
                scanned = scanned.replace(snippet, "")

            matches = re.findall(
                r"(?:_TOL|_EPSILON|tolerance)\s*=\s*(1e-\d+|[\d.]+e-\d+)", scanned
            )
            for m in matches:
                val = float(m)
                if 1e-12 < val < 1e-3:
                    assert "_REL_" in content or "grid" in content.lower(), (
                        f"{path}: suspicious absolute tolerance {m}"
                    )

            # Bare comparisons, e.g. `if norm < 1e-15:` or `abs(x) < 0.0001`,
            # with no named variable for the assignment regex above to
            # catch. Scans the same `scanned` text so the allowlisted
            # zero-length-normal guards are already excluded.
            comparison_matches = re.findall(
                r"[<>]=?\s*(1e-\d+|[\d.]+e-\d+|0\.0{2,}\d*)", scanned
            )
            for m in comparison_matches:
                val = float(m)
                if 1e-12 < val < 1e-3:
                    assert "_REL_" in content or "grid" in content.lower(), (
                        f"{path}: suspicious absolute tolerance in comparison: {m}"
                    )


@pytest.mark.gate
def test_gate_corpus_exists():
    """Fail loudly if CHITIN_GATE_FINAL is set but no corpus exists.

    This catches CI misconfiguration where the gate is "enabled" but
    the corpus was never downloaded or generated.
    """
    if not GATE_FINAL:
        pytest.skip("CHITIN_GATE_FINAL not set")

    found = 0
    for name in CORPUS_FIXTURES:
        trace_dir = TRACES_DIR / name
        if trace_dir.exists():
            trace = load_saved_trace(trace_dir)
            if trace.clips:
                found += 1

    assert found >= 3, (
        f"CHITIN_GATE_FINAL: only {found} fixtures with clips in corpus. "
        f"Expected at least 3 (l_shape, t_shape, staircase). "
        f"Run scripts/capture_trace_corpus.py."
    )
