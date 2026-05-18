# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from chitin.config import Config
from chitin.core import extract
from chitin.hooks import get_post_process_command, run_post_process
from chitin.preflight import check as preflight_check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chitin",
        description="Convex collision geometry from point clouds and meshes",
    )
    sub = parser.add_subparsers(dest="command")

    _add_extract_parser(sub)
    _add_inspect_parser(sub)
    _add_validate_parser(sub)

    args = parser.parse_args(argv)
    if args.command == "extract":
        _cmd_extract(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "validate":
        _cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


def _add_extract_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("extract", help="extract collision geometry from an asset")
    p.add_argument(
        "input",
        type=Path,
        help="Input file (PLY, OBJ, STL, OFF, GLB, GLTF, FBX, USD, USDA, USDC)",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="Output file path")
    p.add_argument(
        "-f",
        "--format",
        choices=["usd", "json", "phys"],
        default=None,
        help="Output format (inferred from extension if omitted)",
    )
    p.add_argument(
        "--concavity",
        type=float,
        default=0.05,
        help="CoACD concavity threshold (default: 0.05)",
    )
    p.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.1,
        help="Minimum opacity to keep a splat (default: 0.1)",
    )
    p.add_argument(
        "--poisson-depth",
        type=int,
        default=8,
        help="Poisson reconstruction depth (default: 8)",
    )
    p.add_argument(
        "--max-hulls",
        type=int,
        default=2048,
        help="Maximum number of convex hulls (default: 2048)",
    )
    p.add_argument(
        "--scene-name",
        type=str,
        default="scene",
        help="Root prim name for USD output (default: scene)",
    )
    p.add_argument(
        "--post-process",
        type=str,
        default=None,
        help="Post-process command to run with {input} substituted. "
        "Overrides ~/.config/chitin/config.toml",
    )
    p.add_argument("--no-hook", action="store_true", help="Skip post-process hook")
    p.add_argument(
        "--cloud",
        action="store_true",
        help="Submit job to chitin cloud service instead of running locally",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Run locally even when preflight check says the input is too large",
    )
    p.add_argument("-q", "--quiet", action="store_true")


def _add_inspect_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("inspect", help="inspect a .phys file")
    p.add_argument("file", type=Path, help="path to .phys file")


def _add_validate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("validate", help="validate a .phys file")
    p.add_argument("file", type=Path, help="path to .phys file")


FORMAT_MAP = {
    ".usda": "usd",
    ".usd": "usd",
    ".json": "json",
    ".phys": "phys",
}


def _infer_format(path: Path) -> str | None:
    return FORMAT_MAP.get(path.suffix.lower())


def _cmd_extract(args: argparse.Namespace) -> None:
    if args.cloud:
        print("chitin: cloud service is not yet available", file=sys.stderr)
        sys.exit(1)

    fmt = args.format or _infer_format(args.output)
    if fmt is None:
        print(
            f"chitin: cannot infer format from {args.output.suffix}, use --format",
            file=sys.stderr,
        )
        sys.exit(1)

    pf = preflight_check(args.input)
    if pf.level == "red" and not args.force:
        print(f"chitin: {pf.message}", file=sys.stderr)
        print(
            "chitin: use --force to run anyway, or --cloud to offload",
            file=sys.stderr,
        )
        sys.exit(1)
    elif pf.level == "yellow" and not args.quiet:
        print(f"chitin: warning: {pf.message}", file=sys.stderr)

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

    if not args.no_hook:
        hook_cmd = get_post_process_command(args.post_process)
        if hook_cmd:
            run_post_process(hook_cmd, args.input, quiet=args.quiet)


def _cmd_inspect(args: argparse.Namespace) -> None:
    from chitin.phys import read_phys

    try:
        pf = read_phys(args.file)
    except ValueError as e:
        print(f"chitin: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"version:    {pf.version}")
    print(f"hulls:      {len(pf.hulls)}")
    print(f"vertices:   {pf.total_vertices}")
    print(f"triangles:  {pf.total_triangles}")
    print(f"rigged:     {pf.has_bones}")
    if pf.bones:
        print(f"bones:      {len(pf.bones)}")
        for b in pf.bones:
            print(f"  {b.name}")

    print()
    for i, h in enumerate(pf.hulls):
        line = f"  hull {i}: {len(h.vertices)} verts, {len(h.indices) // 3} tris"
        if h.bone_index is not None:
            line += f", bone {h.bone_index}"
        aabb_size = h.aabb_max - h.aabb_min
        line += f", size [{aabb_size[0]:.3f}, {aabb_size[1]:.3f}, {aabb_size[2]:.3f}]"
        print(line)


def _cmd_validate(args: argparse.Namespace) -> None:
    from chitin.phys import validate_phys

    issues = validate_phys(args.file)
    if not issues:
        print(f"{args.file}: ok")
        return

    for issue in issues:
        print(f"{args.file}: {issue}", file=sys.stderr)

    errors = sum(1 for i in issues if i.severity == "error")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
