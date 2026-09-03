"""Extract 114 known-failing clips into git-tracked regression corpus.

Policy 0.1.0's holdout gate verdict (docs/holdout-results.json) is FAIL:
all 114 classification disagreements changed clip connectivity. This
script extracts those clips as individual compressed .npz files with
oracle truth, small enough for git (~12 MB).

Usage:
    python scripts/extract_regression_corpus.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.coacd_trace import load_saved_trace
from chitin.coacd_trace_replay import compare_oracle
from chitin.f32_policy import DEFAULT_POLICY

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_ROOT / "tests" / "fixtures" / "traces"
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "regression"
HOLDOUT_RESULTS = REPO_ROOT / "docs" / "holdout-results.json"

SOURCE_VERDICT = "FAIL"
SOURCE_POLICY = "0.1.0"

FAILING_CLIPS: dict[str, list[int]] = {
    "t_shape": [340, 341],
    "h_shape": [
        24,
        41,
        428,
        492,
        2143,
        5439,
        5765,
        5904,
        7503,
        7591,
        7929,
        7940,
        8306,
        8323,
        8574,
        8608,
        8825,
        8846,
        8878,
        9104,
        9247,
        9277,
        9287,
        9380,
        9413,
        9433,
        9467,
        9540,
        9679,
        9786,
        9818,
        9884,
        9924,
        10047,
        10079,
        10081,
        10346,
        10518,
        10531,
        10714,
        10834,
        10942,
        11000,
        11032,
        11230,
        11240,
        11446,
        11559,
        11594,
        11656,
        11688,
        11705,
        11768,
        11911,
        11975,
        12161,
        12207,
        12283,
        12467,
        12530,
        12534,
        12610,
        12643,
        12888,
        12958,
        13118,
        13144,
        13171,
        13185,
        13215,
        13251,
        13314,
        13387,
        13419,
        13441,
        13630,
        13689,
        13775,
        13826,
        13828,
        13840,
        13847,
        13922,
        13933,
        13942,
        14078,
        14314,
        14414,
        14575,
        14576,
        14843,
        14983,
        14998,
        15005,
        15019,
        15023,
        15028,
        15037,
        15040,
        15056,
        15066,
        15076,
        15084,
        15098,
        15112,
        15156,
        15167,
        15193,
        15241,
        15294,
        15296,
        15405,
    ],
}

TOTAL_CLIPS = sum(len(v) for v in FAILING_CLIPS.values())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not HOLDOUT_RESULTS.exists():
        print(f"Error: holdout results not found at {HOLDOUT_RESULTS}")
        return 1

    holdout_digest = _sha256_file(HOLDOUT_RESULTS)
    print(f"Holdout results digest: {holdout_digest[:16]}...")

    # Phase 1: load traces and validate all failing clips have oracle data.
    traces: dict[str, object] = {}
    errors: list[str] = []

    for fixture, indices in FAILING_CLIPS.items():
        trace_dir = TRACES_DIR / fixture
        if not trace_dir.exists():
            print(f"Error: trace directory not found: {trace_dir}")
            return 1

        print(f"Loading {fixture} ({len(indices)} clips)...")
        trace = load_saved_trace(trace_dir)
        traces[fixture] = trace

        for clip_index in indices:
            if clip_index >= len(trace.clips):
                errors.append(
                    f"{fixture}: clip {clip_index} out of range ({len(trace.clips)} clips)"
                )
                continue
            clip = trace.clips[clip_index]
            for field in [
                "input_vertices",
                "oracle_sides",
                "pos_vertices",
                "pos_triangles",
                "neg_vertices",
                "neg_triangles",
            ]:
                if getattr(clip, field) is None:
                    errors.append(f"{fixture}: clip {clip_index} missing {field}")

    if errors:
        print(f"\n{len(errors)} validation errors:")
        for e in errors:
            print(f"  {e}")
        return 1

    print("Validation passed. Extracting...")

    # Phase 2: extract clips and write per-clip npz files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_clips: list[dict] = []
    manifest_fixtures: dict[str, dict] = {}
    written = 0

    for fixture, indices in FAILING_CLIPS.items():
        trace = traces[fixture]
        fixture_dir = OUTPUT_DIR / fixture
        fixture_dir.mkdir(exist_ok=True)

        source_npz = TRACES_DIR / fixture / "arrays.npz"
        source_digest = _sha256_file(source_npz) if source_npz.exists() else "missing"
        manifest_fixtures[fixture] = {
            "clip_indices": indices,
            "source_corpus_digest": source_digest,
        }

        for clip_index in indices:
            clip = trace.clips[clip_index]

            plane = np.array(
                [clip.plane.a, clip.plane.b, clip.plane.c, clip.plane.d],
                dtype=np.float64,
            )
            input_faces = (
                clip.input_faces
                if clip.input_faces is not None
                else np.zeros((0, 3), dtype=np.int32)
            )
            cut_points = (
                clip.cut_points
                if clip.cut_points is not None
                else np.zeros((0, 3), dtype=np.float64)
            )

            npz_path = fixture_dir / f"clip_{clip_index}.npz"
            np.savez_compressed(
                str(npz_path),
                input_vertices=clip.input_vertices,
                input_faces=input_faces,
                plane=plane,
                oracle_sides=clip.oracle_sides,
                pos_vertices=clip.pos_vertices,
                pos_triangles=clip.pos_triangles,
                neg_vertices=clip.neg_vertices,
                neg_triangles=clip.neg_triangles,
                cut_points=cut_points,
                component_id=np.int64(clip.component_id),
            )

            npz_digest = _sha256_file(npz_path)

            comparison = compare_oracle(clip, clip_index, DEFAULT_POLICY)
            num_disagree = comparison.num_disagree if comparison else -1
            near_plane = comparison.near_plane_disagree if comparison else -1
            max_dot = comparison.max_dot_at_disagree if comparison else -1.0

            manifest_clips.append(
                {
                    "fixture": fixture,
                    "clip_index": clip_index,
                    "component_id": clip.component_id,
                    "num_vertices": len(clip.input_vertices),
                    "num_disagree": num_disagree,
                    "near_plane_disagree": near_plane,
                    "max_dot_at_disagree": max_dot,
                    "npz_sha256": npz_digest,
                }
            )
            written += 1

        print(f"  {fixture}: {len(indices)} clips extracted")

    manifest = {
        "version": "1.0",
        "source_verdict": SOURCE_VERDICT,
        "source_policy": SOURCE_POLICY,
        "source_holdout_result_digest": holdout_digest,
        "extraction_date": datetime.now(UTC).isoformat(),
        "total_clips": TOTAL_CLIPS,
        "fixtures": manifest_fixtures,
        "clips": manifest_clips,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExtracted {written}/{TOTAL_CLIPS} clips to {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
