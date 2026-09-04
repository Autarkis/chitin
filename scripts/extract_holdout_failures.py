"""Extract Policy 0.2.0 holdout classifier-failure clips into a diagnostic corpus.

Reads docs/holdout-results-0.2.0.json, collects every clip where f32 and f64
classification disagreed, and pulls the corresponding clip data out of the
holdout-corpus-0.2.0 traces. Writes one .npz per clip plus a manifest.json
into tests/fixtures/traces/holdout_failures_0_2_0/ — a small git-trackable
regression corpus for the 14 clips that diverge under Policy 0.2.0.

Usage:
    python scripts/extract_holdout_failures.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.coacd_trace import load_saved_trace

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "docs" / "holdout-results-0.2.0.json"
CORPUS_DIR = REPO_ROOT / "holdout-corpus-0.2.0"
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "traces" / "holdout_failures_0_2_0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_failures(results: dict) -> list[dict]:
    """Gather every classification-disagreement entry across all fixtures."""
    failures = []
    for fixture_result in results["fixtures"]:
        fixture_name = fixture_result["fixture"]
        for entry in fixture_result.get("classification_disagreements", []):
            failures.append(
                {
                    "fixture": fixture_name,
                    "clip_index": entry["clip_index"],
                    "component_id": entry["component_id"],
                }
            )
    failures.sort(key=lambda f: (f["fixture"], f["clip_index"]))
    return failures


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    failures = _collect_failures(results)
    if not failures:
        raise SystemExit(f"No classification_disagreements found in {RESULTS_PATH}")

    results_digest = _sha256_file(RESULTS_PATH)
    evaluator_commit = results.get("evaluator_commit", "unknown")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    traces_by_fixture = {}
    manifest_clips = []
    per_fixture_counts: dict[str, int] = {}

    for failure in failures:
        fixture_name = failure["fixture"]
        clip_index = failure["clip_index"]
        component_id = failure["component_id"]

        if fixture_name not in traces_by_fixture:
            traces_by_fixture[fixture_name] = load_saved_trace(
                CORPUS_DIR / fixture_name
            )
        trace = traces_by_fixture[fixture_name]
        clip = trace.clips[clip_index]

        vertices = clip.input_vertices.astype(np.float64)
        faces = clip.input_faces.astype(np.int32)
        oracle_sides = clip.oracle_sides.astype(np.int8)
        plane_normal = np.array(
            [clip.plane.a, clip.plane.b, clip.plane.c], dtype=np.float64
        )
        plane_offset = np.array(clip.plane.d, dtype=np.float64)

        out_name = f"{fixture_name}_{clip_index:06d}.npz"
        out_path = OUTPUT_DIR / out_name
        np.savez(
            out_path,
            vertices=vertices,
            faces=faces,
            oracle_sides=oracle_sides,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
        )

        manifest_clips.append(
            {
                "file": out_name,
                "fixture": fixture_name,
                "clip_index": clip_index,
                "component_id": component_id,
                "plane": {
                    "a": float(clip.plane.a),
                    "b": float(clip.plane.b),
                    "c": float(clip.plane.c),
                    "d": float(clip.plane.d),
                },
                "num_vertices": int(vertices.shape[0]),
                "failure_type": "classification_disagreement",
                "npz_sha256": _sha256_file(out_path),
            }
        )
        per_fixture_counts[fixture_name] = per_fixture_counts.get(fixture_name, 0) + 1

    manifest = {
        "source": "holdout-corpus-0.2.0",
        "evaluator_commit": evaluator_commit,
        "result_commit": "0d56ee4",
        "results_digest": results_digest,
        "extraction_date": datetime.now(UTC).isoformat(),
        "policy_version": "0.2.0",
        "clips": manifest_clips,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Extracted {len(manifest_clips)} classifier-failure clips to {OUTPUT_DIR}")
    for fixture_name, count in sorted(per_fixture_counts.items()):
        print(f"  {fixture_name}: {count} clips")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
