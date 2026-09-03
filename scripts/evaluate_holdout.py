"""Holdout evaluation: f32 predicate gate on external-tier corpus.

Loads external-tier trace fixtures, runs replay + oracle at DEFAULT_POLICY,
emits holdout-results.json with per-fixture and aggregate statistics.

Usage:
    python scripts/evaluate_holdout.py [--traces-dir DIR] [--output PATH]
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.coacd_trace import load_saved_trace
from chitin.coacd_trace_replay import (
    compare_oracle,
    replay_classifications,
    replay_clip,
)
from chitin.f32_policy import DEFAULT_POLICY, QuantizationPolicy

EXTERNAL_FIXTURES = ["t_shape", "curved_pipe_quarter", "h_shape"]

DLL_DIGEST = "dd295d37ad6579545f1017c7125bfe8daab65b52a9ff1853a104b8a2851853d3"

TOPOLOGY_SAMPLE_RATE = 0.10
TOPOLOGY_SAMPLE_MIN = 500

KNOWN_CORPUS_DIGESTS = {
    # External tier — Policy 0.1.0 holdout (spent)
    "293790274a89a0c7549f6d86394017a2620fa95ccb71dbe7e52a26c85d10b202",  # t_shape
    "dce6de15b4b3560df0cb799803e84beaead93b1eafb8f55dc288a8e49c41ef14",  # curved_pipe_quarter
    "b42e20807a3cf4fc2b6d8048dfc434e83f13f137c4658951de5471a290fa6972",  # h_shape
    # Regression tier — calibration (spent)
    "c2311cfc0c026ee3e870c35ed8b295b1a455c1b1525e24e4b348db511d0ee92b",  # manifest.json
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _evaluate_fixture(name: str, traces_dir: Path, policy: QuantizationPolicy) -> dict:
    trace_dir = traces_dir / name
    if not trace_dir.exists():
        return {"fixture": name, "error": f"directory not found: {trace_dir}"}

    npz = trace_dir / "arrays.npz"
    corpus_digest = _sha256_file(npz) if npz.exists() else "missing"
    if corpus_digest in KNOWN_CORPUS_DIGESTS:
        raise SystemExit(
            f"REJECTED: fixture '{name}' digest {corpus_digest[:16]}… matches known corpus"
            " — holdout requires unseen data"
        )

    trace = load_saved_trace(trace_dir)
    num_clips = len(trace.clips)

    # Pass 1: classification only (fast, O(n) per clip)
    cls_report = replay_classifications(trace, policy)

    # Identify disagreement clip indices
    disagree_indices = set()
    for r in cls_report.reports:
        if not r.classification_agrees:
            disagree_indices.add(r.clip_index)

    # Build stratified sample: all disagreements + risk-weighted agreeing clips.
    # replay_classifications only stores reports for disagreements, so agreeing
    # clip indices come from enumerating all clips minus skipped/disagreed.
    rng = np.random.default_rng(42)
    all_replayed = set(range(num_clips)) - {
        i
        for i, clip in enumerate(trace.clips)
        if clip.pos_vertices is None
        or clip.neg_vertices is None
        or clip.pos_triangles is None
        or clip.neg_triangles is None
    }
    agree_indices = sorted(all_replayed - disagree_indices)
    sample_count = max(
        TOPOLOGY_SAMPLE_MIN, int(len(agree_indices) * TOPOLOGY_SAMPLE_RATE)
    )
    sample_count = min(sample_count, len(agree_indices))
    sampled_agree = set(rng.choice(agree_indices, size=sample_count, replace=False))
    topology_indices = disagree_indices | sampled_agree

    print(
        f"    Pass 1: {cls_report.num_clips_replayed} classified, "
        f"{len(disagree_indices)} disagree. "
        f"Pass 2: {len(topology_indices)} clips for topology "
        f"({len(disagree_indices)} disagree + {len(sampled_agree)} sampled)",
        flush=True,
    )

    # Pass 2: full topology on targeted clips
    topology_reports = []
    for idx in sorted(topology_indices):
        result = replay_clip(trace.clips[idx], idx, policy)
        if result is not None:
            topology_reports.append(result)

    # Oracle comparison (runs on all clips, classification-level cost)
    oracle_agree = 0
    oracle_total = 0
    for i, clip in enumerate(trace.clips):
        result = compare_oracle(clip, i, policy)
        if result is None:
            continue
        oracle_agree += result.num_agree
        oracle_total += result.num_vertices

    # Disagreement clip details
    disagree_clips = []
    for r in topology_reports:
        if r.clip_index in disagree_indices:
            disagree_clips.append(
                {
                    "clip_index": r.clip_index,
                    "component_id": r.component_id,
                    "clip_face_set_agrees": r.clip_face_set_agrees,
                    "clip_points_agree": r.clip_points_agree,
                    "clip_max_residual": r.clip_max_residual,
                    "clip_agrees": r.clip_agrees,
                    "cap_agrees": r.cap_agrees,
                }
            )

    # Topology stats from sampled clips
    n_topo = len(topology_reports)
    n_clip_agree = sum(1 for r in topology_reports if r.clip_agrees)
    n_cap_agree = sum(1 for r in topology_reports if r.cap_agrees)
    n_face_agree = sum(1 for r in topology_reports if r.clip_face_set_agrees)
    n_points_agree = sum(1 for r in topology_reports if r.clip_points_agree)
    face_only_failures = sum(
        1
        for r in topology_reports
        if not r.clip_face_set_agrees and r.clip_points_agree
    )
    coord_only_failures = sum(
        1
        for r in topology_reports
        if r.clip_face_set_agrees and not r.clip_points_agree
    )
    both_failures = sum(
        1
        for r in topology_reports
        if not r.clip_face_set_agrees and not r.clip_points_agree
    )

    # Error distribution
    residuals_raw = [
        r.clip_max_residual
        for r in topology_reports
        if 0 < r.clip_max_residual < float("inf")
    ]

    # Null/non-finite residual accounting
    null_residual_count = sum(
        1
        for r in topology_reports
        if r.clip_max_residual == 0
        or r.clip_max_residual != r.clip_max_residual
        or r.clip_max_residual == float("inf")
        or r.clip_max_residual == float("-inf")
    )
    inf_residual_count = sum(
        1
        for r in topology_reports
        if r.clip_max_residual == float("inf") or r.clip_max_residual == float("-inf")
    )
    zero_residual_count = sum(1 for r in topology_reports if r.clip_max_residual == 0)
    nan_residual_count = sum(
        1 for r in topology_reports if r.clip_max_residual != r.clip_max_residual
    )

    tolerances = [r.clip_tolerance for r in topology_reports if r.clip_tolerance > 0]
    if residuals_raw:
        arr = np.array(residuals_raw)
        absolute = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }
        mean_tol = float(np.mean(tolerances)) if tolerances else 0.0
        scale_relative = {
            k: v / mean_tol if mean_tol > 0 else 0.0 for k, v in absolute.items()
        }
    else:
        absolute = {}
        scale_relative = {}

    return {
        "fixture": name,
        "corpus_digest": corpus_digest,
        "num_clips": num_clips,
        "num_clips_replayed": cls_report.num_clips_replayed,
        "skipped": cls_report.skipped,
        "classification": {
            "agree": cls_report.num_classification_agree,
            "total": cls_report.num_clips_replayed,
            "rate": cls_report.classification_rate,
        },
        "topology_sample": {
            "total_evaluated": n_topo,
            "disagree_clips": len(disagree_indices),
            "sampled_agree_clips": len(sampled_agree),
            "sampling_ids": [int(x) for x in sorted(topology_indices)],
        },
        "clip_topology": {
            "agree": n_clip_agree,
            "total": n_topo,
            "rate": n_clip_agree / max(1, n_topo),
            "face_set_agree": n_face_agree,
            "face_set_rate": n_face_agree / max(1, n_topo),
            "points_agree": n_points_agree,
            "points_rate": n_points_agree / max(1, n_topo),
            "face_only_failures": face_only_failures,
            "coord_only_failures": coord_only_failures,
            "both_failures": both_failures,
        },
        "cap_topology": {
            "agree": n_cap_agree,
            "total": n_topo,
            "rate": n_cap_agree / max(1, n_topo),
        },
        "oracle": {
            "agree": oracle_agree,
            "total": oracle_total,
            "rate": oracle_agree / max(1, oracle_total),
        },
        "intersection_error": {
            "absolute": absolute,
            "scale_relative": scale_relative,
            "finite_residual_count": len(residuals_raw),
            "null_residual_count": null_residual_count,
            "inf_residual_count": inf_residual_count,
            "zero_residual_count": zero_residual_count,
            "nan_residual_count": nan_residual_count,
        },
        "classification_disagreements": disagree_clips,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="f32 holdout evaluation")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("tests/fixtures/traces"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--policy",
        choices=["0.1.0", "0.2.0"],
        default="0.1.0",
    )
    args = parser.parse_args()

    if args.policy == "0.2.0":
        from chitin.f32_policy import POLICY_0_2_0

        policy = POLICY_0_2_0
    else:
        policy = DEFAULT_POLICY

    if args.output is None:
        args.output = (
            Path(f"docs/holdout-results-{args.policy}.json")
            if args.policy != "0.1.0"
            else Path("docs/holdout-results.json")
        )

    return args, policy


def main():
    args, policy = _parse_args()

    evaluator_digest = _sha256_file(Path(__file__).resolve())
    prior_digest = _sha256_file(args.output) if args.output.exists() else None

    results = {
        "evaluation_date": datetime.now(UTC).isoformat(),
        "evaluator_commit": _git_head(),
        "evaluator_digest": evaluator_digest,
        "supersedes_result_digest": prior_digest,
        "dll_digest": DLL_DIGEST,
        "policy": {
            "version": policy.version,
            "grid_bits": policy.grid_bits,
            "grid_scale": policy.grid_scale,
            "classification_ulp_margin": policy.classification_ulp_margin,
            "intersection_snap_bits": policy.intersection_snap_bits,
            "winding_check": policy.winding_check,
            "ambiguity_fallback": policy.ambiguity_fallback,
        },
        "fixtures": [],
    }

    for name in EXTERNAL_FIXTURES:
        print(f"Evaluating {name}...", flush=True)
        fixture_result = _evaluate_fixture(name, args.traces_dir, policy)
        results["fixtures"].append(fixture_result)
        if "error" not in fixture_result:
            cls = fixture_result["classification"]
            clip = fixture_result["clip_topology"]
            print(
                f"  classification={cls['rate']:.1%}  "
                f"clip_face_set={clip['face_set_rate']:.1%}  "
                f"clip_points={clip['points_rate']:.1%}"
            )

    # Aggregate
    total_cls_agree = sum(
        f["classification"]["agree"] for f in results["fixtures"] if "error" not in f
    )
    total_cls = sum(
        f["classification"]["total"] for f in results["fixtures"] if "error" not in f
    )
    total_oracle_agree = sum(
        f["oracle"]["agree"] for f in results["fixtures"] if "error" not in f
    )
    total_oracle = sum(
        f["oracle"]["total"] for f in results["fixtures"] if "error" not in f
    )
    results["aggregate"] = {
        "classification_rate": total_cls_agree / max(1, total_cls),
        "oracle_rate": total_oracle_agree / max(1, total_oracle),
        "total_clips": total_cls,
        "total_oracle_vertices": total_oracle,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    with open(args.output, "w") as f:
        json.dump(_sanitize(results), f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
