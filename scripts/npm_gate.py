"""Run a web package's CI gate (`npm test` then `npm run build`) locally.

Usage: python scripts/npm_gate.py PACKAGE_DIR

This mirrors what the CI `web` job runs for each package under `integrations/`,
and is wired into pre-commit so a TypeScript change fails in seconds locally
instead of on a pushed branch.

It exists as a Python shim rather than an inline pre-commit `entry:` because
`npm` is `npm.cmd` on Windows, which pre-commit's `language: system` cannot
resolve on its own; `shutil.which` applies PATHEXT and finds it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

STEPS = [
    ("test", ["test"]),
    ("build", ["run", "build"]),
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)

    pkg = Path(sys.argv[1]).resolve()
    if not (pkg / "package.json").exists():
        sys.exit(f"no package.json in {pkg}")

    npm = shutil.which("npm")
    if npm is None:
        sys.exit(
            f"npm not found on PATH, but {pkg.name} changed.\n"
            "Install Node.js 20+, or commit with --no-verify and let CI run the "
            "web gate."
        )

    if not (pkg / "node_modules").exists():
        sys.exit(f"{pkg.name} has no node_modules; run `npm ci` in {pkg}")

    for label, args in STEPS:
        result = subprocess.run([npm, *args], cwd=pkg, check=False)
        if result.returncode != 0:
            sys.exit(f"{pkg.name}: npm {label} failed")


if __name__ == "__main__":
    main()
