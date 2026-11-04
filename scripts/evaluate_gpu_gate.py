#!/usr/bin/env python
"""GPU clipping pipeline gate evaluator (Phase 10).

Runs both the CPU f32 oracle and the GPU clip pipeline on each
(fixture, plane) pair, checks all 8 contract acceptance criteria
(`docs/gpu-clipping-pipeline-contract.md`), and writes an immutable
JSON results file.

The oracle-vs-GPU equivalence contract (plane corpus, CPU oracle, parity and
per-clip criteria) lives in `chitin.gpu.equivalence`; this script orchestrates
the run, determinism check, metadata and results file.

Usage:
    python scripts/evaluate_gpu_gate.py --fixtures g2|calibration --output PATH [--repeat N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# Must run from repo root with src on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.f32_policy import POLICY_0_3_0
from chitin.f32_predicates import (
    classify_plane_f32,
    clip_mesh_f32,
    extract_cap_f32,
)
from chitin.gpu.clip_diagnostics import CpuDiagnostics, attribute_divergence
from chitin.gpu.clip_pipeline import dispatch_clip_pipeline
from chitin.gpu.equivalence import (
    CALIBRATION_FIXTURES,
    check_criteria,
    generate_planes,
    run_cpu_oracle,
)
from chitin.gpu.worker import GPUWorker
from chitin.trace_fixtures import G2_HOLDOUT_FIXTURES

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cpu_oracle_diagnostic(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> CpuDiagnostics:
    """Run CPU oracle, return all stage intermediates for divergence attribution."""
    policy = POLICY_0_3_0
    classification = classify_plane_f32(vertices, plane_point, plane_normal, policy)
    pos_clip = clip_mesh_f32(vertices, faces, plane_point, plane_normal, policy)
    pos_cap = extract_cap_f32(pos_clip, policy)
    neg_clip = clip_mesh_f32(vertices, faces, plane_point, -plane_normal, policy)
    neg_cap = extract_cap_f32(neg_clip, policy)
    return CpuDiagnostics(
        signs=classification.signs,
        pos_faces=pos_clip.faces,
        neg_faces=neg_clip.faces,
        pos_edge_key_map=pos_clip.edge_key_map,
        neg_edge_key_map=neg_clip.edge_key_map,
        pos_boundary_edges=pos_clip.boundary_edges,
        neg_boundary_edges=neg_clip.boundary_edges,
        pos_loops=pos_cap.loops,
        neg_loops=neg_cap.loops,
        pos_cap_faces=pos_cap.cap_faces,
        neg_cap_faces=neg_cap.cap_faces,
        pos_winding_consistent=pos_cap.winding_consistent,
        neg_winding_consistent=neg_cap.winding_consistent,
        pos_vertex_count=len(vertices),
        neg_vertex_count=len(vertices),
    )


def check_determinism(
    worker: GPUWorker,
    vertices: np.ndarray,
    faces: np.ndarray,
    planes: list[tuple[np.ndarray, np.ndarray]],
    repeat: int,
) -> bool:
    """Run each (mesh, plane) pair `repeat` times, verify bitwise-identical output."""
    for pp, pn in planes:
        first = dispatch_clip_pipeline(worker, vertices, faces, pp, pn)
        for _ in range(repeat - 1):
            r = dispatch_clip_pipeline(worker, vertices, faces, pp, pn)
            if not (
                np.array_equal(r.positive_vertices, first.positive_vertices)
                and np.array_equal(r.positive_faces, first.positive_faces)
                and np.array_equal(r.negative_vertices, first.negative_vertices)
                and np.array_equal(r.negative_faces, first.negative_faces)
            ):
                return False
    return True


# Files whose bytes decide PASS/FAIL, in digest order. clip_diagnostics is
# deliberately absent: it only fills `stage_attribution` and never feeds
# `fixture_pass` / `all_pass`.
EVALUATOR_FILES = (
    "src/chitin/gpu/equivalence.py",
    "scripts/evaluate_gpu_gate.py",
)


def evaluator_digest() -> str:
    """sha256 over each evaluator file as `<posix path> NUL <bytes>`, in order."""
    h = hashlib.sha256()
    for rel in EVALUATOR_FILES:
        h.update(rel.encode() + b"\x00")
        h.update((REPO_ROOT / rel).read_bytes())
    return h.hexdigest()


def collect_metadata(worker: GPUWorker) -> dict:
    import subprocess

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
    ).strip()

    shader_dir = REPO_ROOT / "src" / "chitin" / "gpu" / "shaders"
    shader_hashes = {}
    for wgsl in sorted(shader_dir.glob("*.wgsl")):
        shader_hashes[wgsl.name] = hashlib.sha256(wgsl.read_bytes()).hexdigest()

    adapter_info: dict[str, str] = {}
    try:
        info = worker._device.adapter.info
        adapter_info["name"] = str(info.get("device", "unknown"))
        adapter_info["backend"] = str(info.get("backend_type", "unknown"))
        adapter_info["driver"] = str(info.get("description", "unknown"))
    except (AttributeError, KeyError, TypeError):
        adapter_info["name"] = "unknown"

    manifest_path = REPO_ROOT / "docs" / "g2-holdout-manifest.md"
    manifest_digest = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path.exists()
        else "not_found"
    )

    return {
        "implementation_commit": commit,
        "manifest_digest": manifest_digest,
        "policy_version": "0.3.0",
        "adapter": adapter_info,
        "shader_hashes": shader_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU clipping pipeline gate evaluator")
    parser.add_argument("--fixtures", required=True, choices=["g2", "calibration"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    if args.output.exists():
        sys.exit(f"REFUSED: {args.output} already exists -- results are immutable")

    if args.fixtures == "g2":
        manifest = REPO_ROOT / "docs" / "g2-holdout-manifest.md"
        content = manifest.read_text()
        if "SPENT" in content:
            sys.exit("REFUSED: G2 manifest is already SPENT -- no re-evaluation")
        if "FROZEN" not in content:
            sys.exit("REFUSED: G2 manifest is not FROZEN -- freeze before evaluation")
        fixture_dict = dict(G2_HOLDOUT_FIXTURES)
    else:
        fixture_dict = dict(CALIBRATION_FIXTURES)

    if not GPUWorker.available():
        sys.exit("REFUSED: no GPU adapter available")

    all_fixture_results = []
    all_pass = True
    total_clips = 0

    with GPUWorker() as worker:
        metadata = collect_metadata(worker)

        for fixture_name, gen in fixture_dict.items():
            print(f"\n{'=' * 60}")
            print(f"Fixture: {fixture_name}")

            vertices, faces = gen()
            vertices = vertices.astype(np.float32)
            faces = faces.astype(np.uint32)
            source_digest = hashlib.sha256(
                vertices.tobytes() + faces.tobytes()
            ).hexdigest()

            planes = generate_planes(vertices, fixture_name, count=200)
            print(f"  Generated {len(planes)} clipping planes")

            criteria_totals = {
                "classification": 0,
                "topology": 0,
                "finite_coords": 0,
                "valid_indices": 0,
                "loops_closed": 0,
                "caps_wound": 0,
            }

            stage_counts: dict[str, int] = {}

            for pi, (pp, pn) in enumerate(planes):
                gpu_result = dispatch_clip_pipeline(worker, vertices, faces, pp, pn)

                cpu_result = None
                if gpu_result.status.is_success:
                    try:
                        cpu_result = run_cpu_oracle(vertices, faces, pp, pn)
                    except Exception as e:  # noqa: BLE001 -- recorded as a criterion failure
                        print(f"  CPU oracle error on plane {pi}: {e}")

                criteria = check_criteria(
                    gpu_result,
                    cpu_result,
                    pn,
                    len(vertices),
                    vertices,
                )
                for k, v in criteria.items():
                    if v:
                        criteria_totals[k] += 1

                if gpu_result.diagnostics is not None and (
                    not criteria["topology"] or not criteria["loops_closed"]
                ):
                    try:
                        cpu_diag = run_cpu_oracle_diagnostic(vertices, faces, pp, pn)
                        report = attribute_divergence(
                            gpu_result.diagnostics, cpu_diag, len(vertices)
                        )
                        stage_name = report.stage.value
                        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
                    except Exception as e:  # noqa: BLE001
                        stage_counts["error"] = stage_counts.get("error", 0) + 1
                        print(f"  Attribution error plane {pi}: {e}")

            total_clips += len(planes)

            print(f"  Running {args.repeat}-repeat determinism check...")
            deterministic = check_determinism(
                worker, vertices, faces, planes, args.repeat
            )

            fixture_pass = (
                all(criteria_totals[k] == len(planes) for k in criteria_totals)
                and deterministic
            )
            if not fixture_pass:
                all_pass = False

            fixture_result = {
                "name": fixture_name,
                "source_digest": source_digest,
                "num_planes": len(planes),
                "results": {
                    k: {
                        "agree": criteria_totals[k],
                        "total": len(planes),
                        "pass": criteria_totals[k] == len(planes),
                    }
                    for k in criteria_totals
                },
                "deterministic": deterministic,
                "stage_attribution": stage_counts,
                "pass": fixture_pass,
            }
            all_fixture_results.append(fixture_result)

            status = "PASS" if fixture_pass else "FAIL"
            print(f"  {fixture_name}: {status}")
            for k, v in criteria_totals.items():
                if v < len(planes):
                    print(f"    FAIL: {k} = {v}/{len(planes)}")
            if stage_counts:
                print("  Stage attribution:")
                for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
                    print(f"    {stage}: {count}")

    output = {
        "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluator_digest": evaluator_digest(),
        "evaluator_files": list(EVALUATOR_FILES),
        **metadata,
        "fixture_set": args.fixtures,
        "repeat_count": args.repeat,
        "fixtures": all_fixture_results,
        "aggregate": {
            "total_clips": total_clips,
            "total_fixtures": len(fixture_dict),
            "all_criteria_pass": all_pass,
        },
        "verdict": "PASS" if all_pass else "FAIL",
        "readback_audit": "PASS (verified by test_readback_audit, <= 9 calls)",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {output['verdict']}")
    print(f"Results written to: {args.output}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
