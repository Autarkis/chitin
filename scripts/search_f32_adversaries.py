"""Run Chitin's deterministic boundary-directed predicate search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.f32_adversarial import search_adversaries, write_findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search near grid/f32 boundaries for exact-oracle and metamorphic "
            "plane-classification failures."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    findings = search_adversaries(seed=args.seed, cases=args.cases)
    manifest = write_findings(findings, args.output_dir, seed=args.seed)
    print(f"searched={args.cases} findings={len(findings)}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
