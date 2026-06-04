"""Compare quality metrics across chitin artifact bundles.

Usage: python scripts/compare_bundles.py BUNDLE_DIR [BUNDLE_DIR ...]

Prints a side-by-side table of the build-plan quality signals (hull count,
stage deltas, coverage block) for each bundle's build-plan.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROWS = [
    "reconciled_hulls",
    "cell_count",
    "cells_skipped_sparse",
    "cells_failed",
    "dedup_removed",
    "containment_culled",
    "consolidated",
    "seam_repair_delta",
]

COVERAGE_ROWS = [
    "covered_fraction",
    "worst_cell_fraction",
    "worst_decile_fraction",
    "uncovered_count",
    "slack_p50",
    "slack_p95",
    "tolerance",
]


def load_detected(bundle_dir: Path) -> dict:
    plan_path = bundle_dir / "build-plan.json"
    if not plan_path.exists():
        sys.exit(f"no build-plan.json in {bundle_dir}")
    return json.loads(plan_path.read_text())["detected"]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bundles = [Path(p) for p in sys.argv[1:]]
    detected = [load_detected(b) for b in bundles]
    names = [b.parent.name if b.name == "scene_bundle" else b.name for b in bundles]

    width = max(len(n) for n in names) + 2
    label_w = 24

    print(f"{'':<{label_w}}" + "".join(f"{n:>{width}}" for n in names))
    for row in ROWS:
        vals = [d.get(row, "-") for d in detected]
        print(f"{row:<{label_w}}" + "".join(f"{str(v):>{width}}" for v in vals))
    print("coverage:")
    for row in COVERAGE_ROWS:
        vals = [d.get("coverage", {}).get(row, "-") for d in detected]
        print(f"  {row:<{label_w - 2}}" + "".join(f"{str(v):>{width}}" for v in vals))
    print("worst cells (cell:fraction:samples):")
    for name, d in zip(names, detected):
        cells = d.get("coverage", {}).get("worst_cells", [])[:5]
        desc = ", ".join(
            f"{c['cell']}:{c['covered_fraction']}:{c['samples']}" for c in cells
        )
        print(f"  {name}: {desc}")


if __name__ == "__main__":
    main()
