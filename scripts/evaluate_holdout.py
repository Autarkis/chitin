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
from collections import defaultdict
from dataclasses import replace as dc_replace
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
from chitin.f32_policy import DEFAULT_POLICY, POLICY_0_3_0, QuantizationPolicy
from chitin.f32_predicates import (
    canonicalize_inputs_f32,
    categorize_disagreements,
    classify_plane_f32,
    classify_plane_f64,
)

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
    # CI tier — calibration (spent)
    "f2778d3f5ddb58e309bf903667899940b9cbed192b9102791194d889697f125c",  # box
    "5ff57d55f916ed43e9c54c359f7bb2bde9545248e426625d26bfc853025e0e87",  # icosphere
    "874302dd2e001fb74d1235ba9302100464a78f246e4d48f831d013a9fddf57c5",  # thin_panel
    "48a45262c932e01d278544522f15d45d25b59a9f3bc920f5ddcbbdd0e8f68420",  # l_shape
    "1b1223a253fedc2a86e6c000d6a5d873bf586aab0bbdf04f5175bcae9bb36e40",  # thin_u_channel
    "fa2e275d0da0bc8c539b0b772e0795d3818f883af9a91c3b2a810544f174593e",  # cross_bracket
    "9f9be026aecacccb40891d0c60b1874b70fbcdbc9c8d972c4d9300fb4b0760c5",  # staircase
    # Holdout tier -- Policy 0.2.0 holdout (spent)
    "6ef89492d200bbccdfbf3b9e60ae00d0d46ad8ff67b029792d49a7ea1650ac32",  # oblique_gear_prism
    "8be6ff2923d1116bce2b6f44f5bd5655d90f59c109a78c1702cb21b24b92ac82",  # twisted_notched_column
    "545e231435cf1d44989b98ba3822ed8640f7d5a49aa1df8cd627981e18d56272",  # skewed_rectangular_torus
    "c8d75526975637539f89748061ba56341240302b224b8321f3ebe911882ceaf0",  # multiscale_shard_cluster
}

KNOWN_MANIFEST_DIGESTS = {
    # Spent corpus manifests — must not be reused for new evaluations
    "cb30015d74f743fbcb86a22d01cf787c7c7548e7659f8104a2528fa570a6bd37",  # holdout-corpus-0.2.0.json
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


def _load_corpus_manifest(path: Path) -> dict:
    """Load and validate a holdout corpus manifest."""
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "1.0":
        raise SystemExit(
            f"Unsupported manifest schema version: {manifest.get('schema_version')}"
        )
    fixtures = manifest.get("fixtures", [])
    if not fixtures:
        raise SystemExit(f"Manifest {path} declares no fixtures")
    for entry in fixtures:
        if "name" not in entry:
            raise SystemExit(f"Manifest fixture entry missing 'name': {entry}")
    manifest_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest_digest in KNOWN_MANIFEST_DIGESTS:
        raise SystemExit(
            f"REJECTED: manifest {path} digest {manifest_digest[:16]}… matches a spent "
            "corpus manifest — holdout requires a fresh corpus"
        )
    return manifest


def _load_capture_record(traces_dir: Path, manifest_path: Path) -> dict:
    """Load and verify the capture record against the manifest."""
    record_path = traces_dir / "capture-record.json"
    if not record_path.exists():
        raise SystemExit(
            f"Capture record not found at {record_path} — "
            "run capture_holdout_corpus.py first"
        )
    with open(record_path) as f:
        record = json.load(f)
    # Verify the capture was made from this manifest
    actual_manifest_digest = _sha256_file(manifest_path)
    recorded_manifest_digest = record.get("manifest_digest")
    if recorded_manifest_digest != actual_manifest_digest:
        raise SystemExit(
            f"Capture record manifest digest mismatch: "
            f"record={recorded_manifest_digest} actual={actual_manifest_digest}"
        )
    return record


def _verify_corpus_adequacy(manifest: dict, capture_record: dict) -> None:
    """Verify corpus adequacy for Policy 0.3.0 stratified evaluation.

    Reads the floor from the committed manifest (source of truth), verifies
    the capture record's floor matches, and directly checks every stratum
    count against the manifest floor.  Never trusts capture_record's
    corpus_floor_met flag.
    """
    manifest_floor = manifest.get("corpus_size_floor", {}).get("min_clips_per_stratum")
    if manifest_floor is None:
        raise SystemExit(
            "Manifest missing corpus_size_floor.min_clips_per_stratum — "
            "cannot verify corpus adequacy"
        )
    strata_counts = capture_record.get("strata_clip_counts")
    if not strata_counts:
        raise SystemExit(
            "Capture record missing strata_clip_counts — "
            "re-capture with updated capture_holdout_corpus.py"
        )
    record_floor = capture_record.get("corpus_floor")
    if record_floor != manifest_floor:
        raise SystemExit(
            f"Capture record floor ({record_floor}) != manifest floor "
            f"({manifest_floor}) — capture and manifest disagree"
        )
    expected_strata = {"ordinary", "large-offset"}
    actual_strata = set(strata_counts.keys())
    if actual_strata != expected_strata:
        raise SystemExit(
            f"Capture record strata mismatch: "
            f"expected {sorted(expected_strata)}, got {sorted(actual_strata)}"
        )
    undersized = {s: c for s, c in strata_counts.items() if c < manifest_floor}
    if undersized:
        raise SystemExit(
            f"Corpus inadequate: {undersized} below manifest floor "
            f"{manifest_floor} — do not evaluate; capture more data"
        )


ACCEPTANCE_CRITERIA_0_2_0 = {
    "classification_rate": 0.99,
    "oracle_rate": 0.9999,
    "disagree_face_set_rate": 1.0,
    "stratified_face_set_rate": 0.99,
    "invalid_geometry": 0,
}


def _compute_verdict(results: dict) -> dict:
    """Compare aggregates against frozen acceptance criteria."""
    fixtures = [f for f in results["fixtures"] if "error" not in f]

    cls_rate = results["aggregate"]["classification_rate"]
    oracle_rate = results["aggregate"]["oracle_rate"]

    disagree_clips = []
    for f in fixtures:
        disagree_clips.extend(f.get("classification_disagreements", []))
    if disagree_clips:
        disagree_face_ok = sum(
            1 for d in disagree_clips if d.get("clip_face_set_agrees")
        )
        disagree_face_rate = disagree_face_ok / len(disagree_clips)
    else:
        disagree_face_rate = 1.0

    topo_face_agree = sum(f["clip_topology"]["face_set_agree"] for f in fixtures)
    topo_total = sum(f["clip_topology"]["total"] for f in fixtures)
    strat_face_rate = topo_face_agree / max(1, topo_total)

    nan_count = sum(
        f.get("intersection_error", {}).get("nan_residual_count", 0) for f in fixtures
    )
    inf_count = sum(
        f.get("intersection_error", {}).get("inf_residual_count", 0) for f in fixtures
    )
    invalid_count = nan_count + inf_count

    checks = {
        "classification_gte_99pct": {
            "value": cls_rate,
            "threshold": 0.99,
            "pass": cls_rate >= 0.99,
        },
        "oracle_gte_99_99pct": {
            "value": oracle_rate,
            "threshold": 0.9999,
            "pass": oracle_rate >= 0.9999,
        },
        "zero_invalid_geometry": {
            "value": invalid_count,
            "threshold": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "pass": invalid_count == 0,
            "note": "Counts NaN/Inf clip residuals; structural cap-loop welding (#120) is out of scope",
        },
        "disagree_topology_100pct_face_set": {
            "value": disagree_face_rate,
            "threshold": 1.0,
            "count": len(disagree_clips),
            "pass": disagree_face_rate >= 1.0 or len(disagree_clips) == 0,
        },
        "stratified_topology_gte_99pct": {
            "value": strat_face_rate,
            "threshold": 0.99,
            "pass": strat_face_rate >= 0.99,
        },
    }

    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    return {"verdict": verdict, "checks": checks}


def _build_stratum_output(stratum_fixtures: list) -> dict:
    """Aggregate, verdict, and precision-loss rollup for one stratum's fixtures."""
    total_cls_agree = sum(
        f["classification"]["agree"] for f in stratum_fixtures if "error" not in f
    )
    total_cls = sum(
        f["classification"]["total"] for f in stratum_fixtures if "error" not in f
    )
    total_oracle_agree = sum(
        f["oracle"]["agree"] for f in stratum_fixtures if "error" not in f
    )
    total_oracle = sum(
        f["oracle"]["total"] for f in stratum_fixtures if "error" not in f
    )
    total_oracle_excused = sum(
        f.get("oracle", {}).get("on_plane_excused", 0)
        for f in stratum_fixtures
        if "error" not in f
    )
    aggregate = {
        "classification_rate": total_cls_agree / max(1, total_cls),
        "oracle_rate": total_oracle_agree / max(1, total_oracle),
        "total_clips": total_cls,
        "total_oracle_vertices": total_oracle,
        "oracle_on_plane_excused": total_oracle_excused,
    }

    precision_loss_clips = sum(
        f.get("precision_loss", {}).get("precision_loss_clips", 0)
        for f in stratum_fixtures
        if "error" not in f
    )
    genuine_arithmetic_clips = sum(
        f.get("precision_loss", {}).get("genuine_arithmetic_disagree_clips", 0)
        for f in stratum_fixtures
        if "error" not in f
    )

    verdict_result = _compute_verdict(
        {"fixtures": stratum_fixtures, "aggregate": aggregate}
    )

    return {
        "fixtures": stratum_fixtures,
        "aggregate": aggregate,
        "verdict": verdict_result,
        "precision_loss_clips": precision_loss_clips,
        "genuine_arithmetic_clips": genuine_arithmetic_clips,
    }


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

    precision_loss_data = None
    if policy.canonical_f32_inputs:
        raw_policy = dc_replace(policy, canonical_f32_inputs=False)
        precision_loss_clips = 0
        genuine_arithmetic_clips = 0
        total_raw_disagree = 0

        for clip in trace.clips:
            if clip.input_vertices is None or clip.input_faces is None:
                continue
            vertices = clip.input_vertices.astype(np.float64)
            plane = clip.plane
            n = np.array([plane.a, plane.b, plane.c], dtype=np.float64)
            norm = np.linalg.norm(n)
            if norm < 1e-15:
                continue
            normal = n / norm
            point = -(plane.d / norm) * normal

            raw_ref = classify_plane_f64(vertices, point, normal)
            cv, cp, cn = canonicalize_inputs_f32(vertices, point, normal)
            canon_ref = classify_plane_f64(cv, cp, cn)
            canon_cand = classify_plane_f32(cv, cp, cn, raw_policy)

            prec_loss, genuine = categorize_disagreements(
                raw_ref.signs, canon_ref.signs, canon_cand.signs
            )

            raw_cand = classify_plane_f32(vertices, point, normal, raw_policy)
            if np.any(raw_ref.signs != raw_cand.signs):
                total_raw_disagree += 1
            if np.any(prec_loss):
                precision_loss_clips += 1
            if np.any(genuine):
                genuine_arithmetic_clips += 1

        precision_loss_data = {
            "precision_loss_clips": precision_loss_clips,
            "genuine_arithmetic_disagree_clips": genuine_arithmetic_clips,
            "total_raw_disagree_clips": total_raw_disagree,
        }

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
    oracle_on_plane_excused = 0
    for i, clip in enumerate(trace.clips):
        result = compare_oracle(clip, i, policy)
        if result is None:
            continue
        oracle_agree += result.num_agree + result.on_plane_excused
        oracle_total += result.num_vertices
        oracle_on_plane_excused += result.on_plane_excused

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

    fixture_result = {
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
            "on_plane_excused": oracle_on_plane_excused,
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
    if precision_loss_data:
        fixture_result["precision_loss"] = precision_loss_data
    return fixture_result


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
        choices=["0.1.0", "0.2.0", "0.3.0"],
        default="0.1.0",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=None,
        help="JSON manifest for holdout corpus (required for --policy 0.2.0+)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output (development only)",
    )
    args = parser.parse_args()

    if args.policy == "0.2.0":
        from chitin.f32_policy import POLICY_0_2_0

        policy = POLICY_0_2_0
    elif args.policy == "0.3.0":
        policy = POLICY_0_3_0
    else:
        policy = DEFAULT_POLICY

    if args.output is None:
        args.output = (
            Path(f"docs/holdout-results-{args.policy}.json")
            if args.policy != "0.1.0"
            else Path("docs/holdout-results.json")
        )

    if args.policy in ("0.2.0", "0.3.0") and args.corpus_manifest is None:
        raise SystemExit(f"--corpus-manifest is required for --policy {args.policy}")

    return args, policy


def main():
    args, policy = _parse_args()

    CANONICAL_OUTPUTS = {
        "0.1.0": Path("docs/holdout-results.json"),
        "0.2.0": Path("docs/holdout-results-0.2.0.json"),
        "0.3.0": Path("docs/holdout-results-0.3.0.json"),
    }
    if args.output.exists():
        canonical = CANONICAL_OUTPUTS.get(policy.version)
        if canonical and args.output.resolve() == canonical.resolve():
            raise SystemExit(
                f"Canonical result {args.output} exists and is permanently immutable."
            )
        if not args.force:
            raise SystemExit(
                f"Output {args.output} already exists — holdout result is immutable. "
                "Use --force to overwrite (development only)."
            )

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
            "canonical_f32_inputs": policy.canonical_f32_inputs,
        },
        "fixtures": [],
    }

    if args.corpus_manifest:
        manifest = _load_corpus_manifest(args.corpus_manifest)
        fixture_names = [entry["name"] for entry in manifest["fixtures"]]
        fixture_strata = {
            entry["name"]: entry.get("stratum", "ordinary")
            for entry in manifest["fixtures"]
        }
        results["corpus_manifest"] = str(args.corpus_manifest)
        results["corpus_manifest_digest"] = _sha256_file(args.corpus_manifest)
        # Verify capture record chain-of-custody
        capture_record = _load_capture_record(args.traces_dir, args.corpus_manifest)
        trace_digests = {
            r["name"]: r["trace_digest"] for r in capture_record.get("fixtures", [])
        }
        for name in fixture_names:
            npz_path = args.traces_dir / name / "arrays.npz"
            if not npz_path.exists():
                raise SystemExit(f"Trace data missing for fixture '{name}': {npz_path}")
            actual_digest = _sha256_file(npz_path)
            expected_digest = trace_digests.get(name)
            if expected_digest is None:
                raise SystemExit(f"Fixture '{name}' not in capture record")
            if actual_digest != expected_digest:
                raise SystemExit(
                    f"Trace digest mismatch for '{name}': "
                    f"record={expected_digest} actual={actual_digest}"
                )
        # Verify compiler identity from capture record
        traced_coacd = capture_record.get("traced_coacd", {})
        if not traced_coacd.get("dll_verified"):
            raise SystemExit(
                "Capture record has dll_verified=false — "
                "holdout evaluation requires verified compiler identity"
            )
        recorded_dll_digest = traced_coacd.get("dll_digest")
        if recorded_dll_digest != DLL_DIGEST:
            raise SystemExit(
                f"Capture record DLL digest mismatch:\n"
                f"  capture record: {recorded_dll_digest}\n"
                f"  evaluator:      {DLL_DIGEST}"
            )
        results["capture_record_verified"] = True

        if policy.version == "0.3.0":
            _verify_corpus_adequacy(manifest, capture_record)
    else:
        fixture_names = EXTERNAL_FIXTURES
        fixture_strata = {name: "ordinary" for name in EXTERNAL_FIXTURES}

    for name in fixture_names:
        print(f"Evaluating {name}...", flush=True)
        fixture_result = _evaluate_fixture(name, args.traces_dir, policy)
        results["fixtures"].append(fixture_result)
        if "error" in fixture_result:
            raise SystemExit(
                f"ABORT: fixture '{name}' failed: {fixture_result['error']}. "
                "No partial results written."
            )
        if "error" not in fixture_result:
            cls = fixture_result["classification"]
            clip = fixture_result["clip_topology"]
            print(
                f"  classification={cls['rate']:.1%}  "
                f"clip_face_set={clip['face_set_rate']:.1%}  "
                f"clip_points={clip['points_rate']:.1%}"
            )

    if policy.version == "0.3.0":
        # Stratum-aware verdict: group fixture results by stratum, compute
        # verdict per stratum independently, and require all strata to pass.
        strata_fixtures = defaultdict(list)
        for fixture_result in results["fixtures"]:
            stratum = fixture_strata.get(fixture_result["fixture"], "ordinary")
            strata_fixtures[stratum].append(fixture_result)

        strata_output = {}
        all_pass = True
        for stratum, stratum_fixtures in strata_fixtures.items():
            stratum_output = _build_stratum_output(stratum_fixtures)
            strata_output[stratum] = stratum_output
            if stratum_output["verdict"]["verdict"] != "PASS":
                all_pass = False

            print(
                f"\nStratum '{stratum}' verdict: {stratum_output['verdict']['verdict']}"
            )
            for check_name, check in stratum_output["verdict"]["checks"].items():
                status = "PASS" if check["pass"] else "FAIL"
                print(f"  {check_name}: {status} ({check['value']})")

        del results["fixtures"]
        results["strata"] = strata_output
        results["overall_verdict"] = "PASS" if all_pass else "FAIL"
        print(f"\nOverall verdict: {results['overall_verdict']}")
    else:
        # Aggregate
        total_cls_agree = sum(
            f["classification"]["agree"]
            for f in results["fixtures"]
            if "error" not in f
        )
        total_cls = sum(
            f["classification"]["total"]
            for f in results["fixtures"]
            if "error" not in f
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

        total_oracle_excused = sum(
            f.get("oracle", {}).get("on_plane_excused", 0)
            for f in results["fixtures"]
            if "error" not in f
        )
        results["aggregate"]["oracle_on_plane_excused"] = total_oracle_excused

        if policy.canonical_f32_inputs:
            results["aggregate"]["precision_loss_clips"] = sum(
                f.get("precision_loss", {}).get("precision_loss_clips", 0)
                for f in results["fixtures"]
                if "error" not in f
            )
            results["aggregate"]["genuine_arithmetic_disagree_clips"] = sum(
                f.get("precision_loss", {}).get("genuine_arithmetic_disagree_clips", 0)
                for f in results["fixtures"]
                if "error" not in f
            )

        if policy.version != "0.1.0":
            verdict_result = _compute_verdict(results)
            results["verdict"] = verdict_result["verdict"]
            results["verdict_checks"] = verdict_result["checks"]
            print(f"\nVerdict: {verdict_result['verdict']}")
            for check_name, check in verdict_result["checks"].items():
                status = "PASS" if check["pass"] else "FAIL"
                print(f"  {check_name}: {status} ({check['value']})")

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
    if "verdict" in results:
        print(f"Verdict: {results['verdict']}")
    elif "overall_verdict" in results:
        print(f"Overall verdict: {results['overall_verdict']}")


if __name__ == "__main__":
    main()
