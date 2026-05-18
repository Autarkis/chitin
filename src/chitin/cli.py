# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import argparse
import time
from pathlib import Path

from chitin.config import Config
from chitin.core import extract


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chitin",
        description="Convex collision geometry from point clouds and meshes",
    )
    parser.add_argument("input", type=Path, help="Input file (PLY, OBJ, STL)")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output file path"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["usd", "json", "phys"],
        default=None,
        help="Output format (inferred from extension if omitted)",
    )
    parser.add_argument(
        "--concavity",
        type=float,
        default=0.05,
        help="CoACD concavity threshold (default: 0.05)",
    )
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.1,
        help="Minimum opacity to keep a splat (default: 0.1)",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=8,
        help="Poisson reconstruction depth (default: 8)",
    )
    parser.add_argument(
        "--max-hulls",
        type=int,
        default=2048,
        help="Maximum number of convex hulls (default: 2048)",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="scene",
        help="Root prim name for USD output (default: scene)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args(argv)

    fmt = args.format or _infer_format(args.output)
    if fmt is None:
        parser.error(f"Cannot infer format from {args.output.suffix}. Use --format.")

    config = Config(
        concavity=args.concavity,
        opacity_threshold=args.opacity_threshold,
        poisson_depth=args.poisson_depth,
        max_hulls=args.max_hulls,
    )

    if not args.quiet:
        print(f"chitin: {args.input} -> {args.output} ({fmt})")

    t0 = time.monotonic()
    result = extract(args.input, config)
    dt = time.monotonic() - t0

    if fmt == "usd":
        result.to_usd(args.output, scene_name=args.scene_name)
    elif fmt == "json":
        result.to_json(args.output)
    elif fmt == "phys":
        result.to_phys(args.output)

    if not args.quiet:
        print(
            f"chitin: {len(result.hulls)} hulls from "
            f"{result.source_vertex_count} source verts in {dt:.1f}s"
        )


FORMAT_MAP = {
    ".usda": "usd",
    ".usd": "usd",
    ".json": "json",
    ".phys": "phys",
}


def _infer_format(path: Path) -> str | None:
    return FORMAT_MAP.get(path.suffix.lower())


if __name__ == "__main__":
    main()
