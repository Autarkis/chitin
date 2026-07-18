"""Run N identical garden builds and compare their bundles for divergence.

Usage: python scripts/determinism_repro.py [N] [OUT_DIR]

Runs N sequential `chitin extract` builds with the rebaseline config
(concavity 0.1, density-quantile 0.3) against the garden splat, writing
each bundle to OUT_DIR/run-i/, then prints the compare_bundles table.
Defaults: N=3, OUT_DIR=/tmp/chitin-determinism-<timestamp>.
"""
# Existing-check: scripts/ (compare_bundles.py reused for diffing),
# ~/.claude/scripts/, devops_tools/ - no runner exists, creating new.

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUT = (
    REPO / "examples/utility-proof/assets/garden-3dgs/mipnerf360_garden_crop_table.ply"
)
# repo venv, not PATH: the miniconda chitin is py3.13 and lacks open3d
CHITIN = REPO / ".venv/bin/chitin"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(f"/tmp/chitin-determinism-{time.strftime('%Y%m%d-%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = []
    for i in range(1, n + 1):
        run_dir = out_dir / f"run-{i}"
        run_dir.mkdir(exist_ok=True)
        t0 = time.time()
        subprocess.run(
            [
                str(CHITIN),
                "extract",
                str(INPUT),
                "-o",
                str(run_dir / "scene.phys"),
                "--concavity",
                "0.1",
                "--density-quantile",
                "0.3",
                "-b",
                "-q",
            ],
            check=True,
            stdout=(run_dir / "stdout.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        print(f"run-{i} done in {time.time() - t0:.0f}s", flush=True)
        bundles.append(str(run_dir / "scene_bundle"))

    table = subprocess.run(
        [sys.executable, str(REPO / "scripts/compare_bundles.py"), *bundles],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (out_dir / "comparison.txt").write_text(table)
    print(table)
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
