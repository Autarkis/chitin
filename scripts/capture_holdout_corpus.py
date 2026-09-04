"""Capture holdout corpus from a selection manifest.

Usage:
    python scripts/capture_holdout_corpus.py --manifest MANIFEST --output-dir DIR

Unlike scripts/capture_trace_corpus.py (which captures the CI-tier fixture
set into the established tests/fixtures/traces tree and must not be
modified), this script captures an arbitrary holdout corpus selected by a
manifest into a caller-chosen --output-dir. It never writes into the CI
trace tree.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.coacd_trace import capture_trace, save_trace
from chitin.trace_fixtures import HOLDOUT_FIXTURES

SCRIPT_PATH = Path(__file__).resolve()


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_arrays(vertices: np.ndarray, faces: np.ndarray) -> str:
    """SHA-256 of geometry arrays serialized via np.savez."""
    buf = io.BytesIO()
    np.savez(buf, vertices=vertices, faces=faces)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _find_coacd_library() -> Path | None:
    """Locate the CoACD shared library in the installed package."""
    try:
        import coacd

        pkg_dir = Path(coacd.__file__).parent
        for name in [
            "lib_coacd.dll",
            "libcoacd.so",
            "libcoacd.dylib",
            "_coacd.pyd",
            "_coacd.so",
        ]:
            candidate = pkg_dir / name
            if candidate.exists():
                return candidate
        # Search one level deep for platform-specific subdirs
        for child in pkg_dir.iterdir():
            if child.is_dir():
                for name in [
                    "lib_coacd.dll",
                    "libcoacd.so",
                    "libcoacd.dylib",
                ]:
                    candidate = child / name
                    if candidate.exists():
                        return candidate
    except ImportError:
        pass
    return None


def _abort(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def capture_single(name: str, output_dir: Path) -> None:
    """Capture one fixture (invoked as a subprocess to get fresh DLL state)."""
    gen = HOLDOUT_FIXTURES[name]
    verts, faces = gen()
    out = output_dir / name
    trace = capture_trace(verts, faces, threshold=0.05)
    save_trace(trace, out)
    print(
        f"{name}: clips={len(trace.clips)}, planes={len(trace.planes)}, "
        f"mcts={len(trace.mcts_transitions)}, parts={len(trace.output_parts)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture holdout corpus trace data from a selection manifest. "
            "Writes into --output-dir only; never touches the CI trace tree "
            "used by scripts/capture_trace_corpus.py."
        ),
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the holdout corpus manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write trace data into (not the CI trace tree)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing fixture output directories (development only)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest_path: Path = args.manifest
    output_dir: Path = args.output_dir

    if not manifest_path.is_file():
        _abort(f"manifest not found: {manifest_path}")

    ci_trace_tree = (
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces"
    ).resolve()
    if output_dir.resolve().is_relative_to(ci_trace_tree):
        _abort(
            f"--output-dir must not point inside the CI trace tree "
            f"({ci_trace_tree}) — use a separate directory"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_digest = _sha256_file(manifest_path)

    expected_dll_digest = manifest.get("traced_coacd", {}).get("dll_digest")
    dll_path = _find_coacd_library()
    if dll_path:
        actual_dll_digest = _sha256_file(dll_path)
        if expected_dll_digest and actual_dll_digest != expected_dll_digest:
            _abort(
                f"Traced CoACD DLL digest mismatch:\n"
                f"  manifest: {expected_dll_digest}\n"
                f"  actual:   {actual_dll_digest}\n"
                f"  path:     {dll_path}"
            )
        print(f"Traced CoACD DLL verified: {dll_path}")
        print(f"  digest: {actual_dll_digest}")
    else:
        _abort(
            "Could not locate CoACD shared library — "
            "holdout capture requires verified compiler identity"
        )

    fixtures = manifest.get("fixtures")
    if not fixtures:
        _abort("manifest has no 'fixtures' array")

    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in fixtures:
        name = entry["name"]
        expected_digest = entry["source_digest"]
        stratum = entry.get("stratum", "<unknown>")

        if name not in HOLDOUT_FIXTURES:
            _abort(
                f"unknown fixture '{name}' (not in chitin.trace_fixtures.HOLDOUT_FIXTURES)"
            )

        fixture_out = output_dir / name
        if fixture_out.exists() and not args.force:
            _abort(
                f"output directory already exists for fixture '{name}': {fixture_out} "
                "(pass --force to overwrite; development only)"
            )

        print(
            f"Verifying source geometry for {name} (stratum={stratum})...", flush=True
        )
        verts, faces = HOLDOUT_FIXTURES[name]()
        actual_digest = _sha256_arrays(verts, faces)
        if actual_digest != expected_digest:
            _abort(
                f"source digest mismatch for '{name}': "
                f"manifest={expected_digest} actual={actual_digest}"
            )

        print(f"Capturing {name}...", flush=True)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), name, str(output_dir)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            _abort(f"capture failed for '{name}': {result.stderr.strip()}")
        print(f"  {result.stdout.strip()}")

        arrays_path = fixture_out / "arrays.npz"
        if not arrays_path.exists():
            _abort(f"capture for '{name}' did not produce {arrays_path}")
        trace_digest = _sha256_file(arrays_path)

        meta_path = fixture_out / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        records.append(
            {
                "name": name,
                "source_digest": expected_digest,
                "source_digest_verified": True,
                "trace_digest": trace_digest,
                "clips": meta["num_clips"],
                "parts": meta["num_parts"],
            }
        )

    fixture_strata = {
        entry["name"]: entry.get("stratum", "ordinary") for entry in fixtures
    }
    strata_counts: dict[str, int] = {}
    for rec in records:
        s = fixture_strata[rec["name"]]
        strata_counts[s] = strata_counts.get(s, 0) + rec["clips"]

    expected_strata = {"ordinary", "large-offset"}
    actual_strata = set(strata_counts.keys())
    if actual_strata != expected_strata:
        _abort(f"Unexpected strata: expected {expected_strata}, got {actual_strata}")

    corpus_floor = manifest.get("corpus_size_floor", {}).get(
        "min_clips_per_stratum", 30000
    )
    corpus_floor_met = all(count >= corpus_floor for count in strata_counts.values())
    if not corpus_floor_met:
        for stratum, count in strata_counts.items():
            if count < corpus_floor:
                print(
                    f"WARNING: stratum '{stratum}' has {count} clips, "
                    f"below floor {corpus_floor} — corpus is inadequate for evaluation",
                    file=sys.stderr,
                )

    capture_record = {
        "capture_date": datetime.now(UTC).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_digest": manifest_digest,
        "fixtures": records,
        "traced_coacd": {
            "dll_path": str(dll_path) if dll_path else None,
            "dll_digest": actual_dll_digest if dll_path else None,
            "dll_verified": dll_path is not None,
        },
        "strata_clip_counts": strata_counts,
        "corpus_floor": corpus_floor,
        "corpus_floor_met": corpus_floor_met,
    }
    record_path = output_dir / "capture-record.json"
    with open(record_path, "w") as f:
        json.dump(capture_record, f, indent=2)

    print(f"\nCapture record written to {record_path}")
    print(f"Corpus saved to {output_dir}")


if __name__ == "__main__":
    # Subprocess re-invocation (fresh DLL state) passes the fixture name as a
    # bare positional argv[1], followed by the output dir as argv[2]. Normal
    # top-level invocation always uses "--manifest"/"--output-dir"/"--force"
    # flags, so a leading "--" reliably distinguishes the two call shapes.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        capture_single(sys.argv[1], Path(sys.argv[2]))
    else:
        main()
