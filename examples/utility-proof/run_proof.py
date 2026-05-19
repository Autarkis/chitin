# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
"""Run chitin against utility-proof datasets and produce structured reports.

Usage:
    python run_proof.py                     # run all downloaded datasets
    python run_proof.py scanned-object      # run one dataset by key
    python run_proof.py --list              # list available runs

Each run produces:
    reports/<key>/
        report.json     — metrics (hull count, build time, file sizes, warnings)
        colliders.phys  — generated .phys sidecar
        colliders.json  — JSON companion
        validate.txt    — chitin validate output
        inspect.txt     — chitin inspect output
"""

import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

MANIFEST = Path(__file__).parent / "datasets.toml"
ASSETS_DIR = Path(__file__).parent / "assets"
REPORTS_DIR = Path(__file__).parent / "reports"

FORMAT_EXTENSIONS = {
    "obj": [".obj"],
    "ply": [".ply"],
    "fbx": [".fbx"],
    "glb": [".glb", ".gltf"],
}


def load_manifest():
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)


def find_input_file(asset_dir, entry):
    for path_key in ("avatar_path", "asset_path"):
        if path_key in entry:
            candidate = asset_dir / entry[path_key]
            if candidate.exists():
                return candidate

    extensions = FORMAT_EXTENSIONS.get(entry["format"], [f".{entry['format']}"])
    for ext in extensions:
        matches = list(asset_dir.rglob(f"*{ext}"))
        if matches:
            return max(matches, key=lambda p: p.stat().st_size)

    return None


def run_chitin_extract(input_path, output_dir, concavity=0.05):
    phys_path = output_dir / "colliders.phys"
    json_path = output_dir / "colliders.json"

    t0 = time.monotonic()
    result = subprocess.run(
        [
            "chitin",
            "extract",
            str(input_path),
            "-o",
            str(phys_path),
            "--concavity",
            str(concavity),
        ],
        capture_output=True,
        text=True,
    )
    build_time = time.monotonic() - t0

    subprocess.run(
        [
            "chitin",
            "extract",
            str(input_path),
            "-o",
            str(json_path),
            "--format",
            "json",
            "--concavity",
            str(concavity),
        ],
        capture_output=True,
        text=True,
    )

    return {
        "build_time_s": round(build_time, 2),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "phys_exists": phys_path.exists(),
        "json_exists": json_path.exists(),
        "phys_size_bytes": phys_path.stat().st_size if phys_path.exists() else 0,
        "json_size_bytes": json_path.stat().st_size if json_path.exists() else 0,
    }


def run_chitin_inspect(phys_path):
    result = subprocess.run(
        ["chitin", "inspect", str(phys_path)],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def run_chitin_validate(phys_path):
    result = subprocess.run(
        ["chitin", "validate", str(phys_path)],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr, result.returncode


def run_one(key, entry):
    asset_dir = ASSETS_DIR / key
    if not asset_dir.exists():
        print(f"  [{key}] asset not downloaded, run download.py first")
        return None

    input_file = find_input_file(asset_dir, entry)
    if not input_file:
        print(f"  [{key}] no {entry['format']} file found in {asset_dir}")
        return None

    report_dir = REPORTS_DIR / key
    report_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"  [{key}] input: {input_file.name} ({input_file.stat().st_size / 1e6:.1f} MB)"
    )

    print(f"  [{key}] extracting...")
    extract_result = run_chitin_extract(input_file, report_dir)

    if not extract_result["phys_exists"]:
        print(f"  [{key}] FAILED: no .phys output")
        print(f"    stderr: {extract_result['stderr'][:500]}")
        report = {
            "key": key,
            "name": entry["name"],
            "category": entry["category"],
            "status": "failed",
            "error": extract_result["stderr"][:1000],
            "build_time_s": extract_result["build_time_s"],
        }
        (report_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    print(f"  [{key}] inspecting...")
    inspect_output = run_chitin_inspect(report_dir / "colliders.phys")
    (report_dir / "inspect.txt").write_text(inspect_output)

    print(f"  [{key}] validating...")
    validate_output, validate_rc = run_chitin_validate(report_dir / "colliders.phys")
    (report_dir / "validate.txt").write_text(validate_output)

    report = {
        "key": key,
        "name": entry["name"],
        "category": entry["category"],
        "input_file": input_file.name,
        "input_size_mb": round(input_file.stat().st_size / 1e6, 2),
        "status": "ok" if validate_rc == 0 else "warnings",
        "build_time_s": extract_result["build_time_s"],
        "phys_size_bytes": extract_result["phys_size_bytes"],
        "json_size_bytes": extract_result["json_size_bytes"],
        "validation_clean": validate_rc == 0,
        "inspect": inspect_output.strip(),
        "validate": validate_output.strip(),
    }

    (report_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(
        f"  [{key}] done: {extract_result['build_time_s']}s, "
        f".phys={extract_result['phys_size_bytes']} bytes, "
        f"valid={'yes' if validate_rc == 0 else 'NO'}"
    )
    return report


def print_summary(reports):
    print("\n" + "=" * 72)
    print(f"{'Dataset':<24} {'Status':<10} {'Time':>8} {'Phys':>10} {'Hulls'}")
    print("-" * 72)
    for r in reports:
        if r is None:
            continue
        hulls = "?"
        inspect_text = r.get("inspect", "")
        for line in inspect_text.split("\n"):
            if "hull" in line.lower() and "count" not in line.lower():
                pass
            if line.strip().startswith("hulls:"):
                hulls = line.strip().split(":")[1].strip()
        print(
            f"{r['key']:<24} {r['status']:<10} {r['build_time_s']:>6.1f}s "
            f"{r.get('phys_size_bytes', 0):>8} B  {hulls}"
        )
    print("=" * 72)


def main():
    manifest = load_manifest()

    if "--list" in sys.argv:
        for key in manifest:
            report_path = REPORTS_DIR / key / "report.json"
            status = "not run"
            if report_path.exists():
                r = json.loads(report_path.read_text())
                status = r.get("status", "unknown")
            print(f"  {key:<24} {status}")
        return

    keys = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not keys:
        keys = list(manifest.keys())

    reports = []
    for key in keys:
        if key not in manifest:
            print(f"  unknown dataset: {key}")
            continue
        entry = manifest[key]
        print(f"\n=== {entry['name']} ===")
        reports.append(run_one(key, entry))

    print_summary(reports)


if __name__ == "__main__":
    main()
