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
    _add_check_parser(sub)
    _add_inspect_parser(sub)
    _add_validate_parser(sub)
    _add_probe_parser(sub)
    _add_sweep_parser(sub)
    _add_convert_parser(sub)

    args = parser.parse_args(argv)
    if args.command == "extract":
        _cmd_extract(args)
    elif args.command == "check":
        _cmd_check(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "probe":
        _cmd_probe(args)
    elif args.command == "sweep":
        _cmd_sweep(args)
    elif args.command == "convert":
        _cmd_convert(args)
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
        default=None,
        help="Poisson reconstruction depth (default: auto from point count)",
    )
    p.add_argument(
        "--max-hulls",
        type=int,
        default=2048,
        help="Max convex hulls per decomposition unit (per octree cell / "
        "per bone), not a global cap (default: 2048)",
    )
    p.add_argument(
        "--lod-concavities",
        type=str,
        default=None,
        help="Comma-separated concavity thresholds for LOD tiers (e.g. 0.1,0.3,0.5)",
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
        "--force",
        action="store_true",
        help="Run locally even when preflight check says the input is too large",
    )
    p.add_argument(
        "--density-quantile",
        type=float,
        default=0.1,
        help="Poisson density filter quantile (default: 0.1, raise to 0.3+ for environments)",
    )
    p.add_argument(
        "--proximity-filter",
        type=float,
        default=0.0,
        help="Remove mesh vertices farther than N * median_nn_distance from input points (0 = disabled)",
    )
    p.add_argument(
        "--thin-shell",
        action="store_true",
        help="Extrude filtered surface into a thin solid before decomposition (for environment scans)",
    )
    p.add_argument(
        "--thin-shell-thickness",
        type=float,
        default=0.0,
        help="Shell thickness (0 = auto from mesh extent)",
    )
    p.add_argument(
        "--flatness-threshold",
        type=float,
        default=0.9,
        help="PCA eigenvalue ratio to classify octree cell as flat (0 = disabled, 1 = everything is flat)",
    )
    p.add_argument(
        "--auto-verify",
        action="store_true",
        help="Run raycast probe after extraction and print coverage summary",
    )
    p.add_argument(
        "--no-auto-environment",
        action="store_true",
        help="Disable auto-detection of environment scans (thin-shell, proximity filter)",
    )
    p.add_argument(
        "--no-seam-repair",
        action="store_true",
        help="Disable seam repair pass (skip re-merging cells at octree boundaries)",
    )
    p.add_argument(
        "--snug-fit",
        action="store_true",
        help="Tighten hull face planes onto covered input points (experimental)",
    )
    p.add_argument(
        "--target-height",
        type=float,
        default=None,
        help="Uniformly rescale the input so its height (up-axis extent) is N "
        "meters before extraction (for non-metric source assets)",
    )
    p.add_argument(
        "--target-footprint",
        type=float,
        default=None,
        help="Real-world footprint (largest horizontal extent, meters) used "
        "instead of --target-height for flat objects like rugs",
    )
    p.add_argument(
        "--up-axis",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Which axis is up/height for --target-height (default: 1, glTF Y-up)",
    )
    p.add_argument(
        "-b",
        "--bundle",
        action="store_true",
        help="Write artifact bundle (build-plan.json, analysis.json, resolved-config.json) alongside output",
    )
    p.add_argument("-q", "--quiet", action="store_true")


def _add_check_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("check", help="check which processing path an input needs")
    p.add_argument(
        "input",
        type=Path,
        help="Input file (PLY, OBJ, STL, OFF, GLB, GLTF, FBX, USD, USDA, USDC)",
    )


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
            "chitin: use --force to run anyway",
            file=sys.stderr,
        )
        sys.exit(1)
    elif pf.level == "yellow" and not args.quiet:
        print(f"chitin: warning: {pf.message}", file=sys.stderr)

    if pf.hints and not args.quiet:
        for hint in pf.hints:
            print(f"chitin: hint: {hint}", file=sys.stderr)

    lod_concavities = None
    if args.lod_concavities:
        lod_concavities = [float(x.strip()) for x in args.lod_concavities.split(",")]

    config = Config(
        concavity=args.concavity,
        opacity_threshold=args.opacity_threshold,
        poisson_depth=args.poisson_depth,
        max_hulls=args.max_hulls,
        lod_concavities=lod_concavities,
        poisson_density_quantile=args.density_quantile,
        surface_proximity_filter=args.proximity_filter,
        thin_shell=args.thin_shell,
        thin_shell_thickness=args.thin_shell_thickness,
        flatness_threshold=args.flatness_threshold,
        auto_environment=not args.no_auto_environment,
        seam_repair=not args.no_seam_repair,
        snug_fit=args.snug_fit,
        target_height=args.target_height,
        target_footprint=args.target_footprint,
        up_axis=args.up_axis,
    )

    if not args.quiet:
        print(f"chitin: {args.input} -> {args.output} ({fmt})")

    t0 = time.monotonic()
    result = extract(args.input, config)
    dt = time.monotonic() - t0

    if args.bundle:
        from chitin.exporters.bundle import export_bundle

        bundle_dir = args.output.parent / (args.output.stem + "_bundle")
        export_bundle(result, bundle_dir, fmt=fmt, scene_name=args.scene_name)
        if not args.quiet:
            print(f"chitin: bundle written to {bundle_dir}/")
    else:
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

    phys_path = (bundle_dir / "scene.phys") if args.bundle else args.output
    if args.auto_verify and fmt == "phys":
        from chitin.verify.probe import probe

        pr = probe(phys_path, grid_resolution=32)
        pct = pr.coverage * 100
        print(
            f"chitin: verify: {pct:.1f}% coverage "
            f"({pr.hits}/{pr.total_rays} rays, "
            f"{pr.gap_clusters} gap clusters, "
            f"confidence={pr.confidence})"
        )
        if pr.confidence == "low":
            print(
                "chitin: verify: low coverage — consider "
                "--concavity 0.01, --density-quantile 0.05, "
                "or --thin-shell for environment scans",
                file=sys.stderr,
            )
        if args.bundle:
            pr.to_json(bundle_dir / "probe.json")

    if not args.no_hook:
        hook_cmd = get_post_process_command(args.post_process)
        if hook_cmd:
            run_post_process(hook_cmd, args.input, quiet=args.quiet)


def _cmd_check(args: argparse.Namespace) -> None:
    from chitin.analyze import analyze_input

    path = Path(args.input)
    if not path.exists():
        print(f"chitin: {path} not found", file=sys.stderr)
        sys.exit(1)

    file_size = path.stat().st_size
    print(f"file:       {path.name}")
    print(f"format:     {path.suffix.lower().lstrip('.')}")
    print(f"size:       {file_size / 1024:.0f} KB")

    try:
        analysis = analyze_input(path)
    except ValueError as e:
        print(f"chitin: {e}", file=sys.stderr)
        sys.exit(1)

    _print_analysis(analysis)


def _print_analysis(analysis) -> None:
    from chitin.analyze import InputAnalysis

    a: InputAnalysis = analysis

    print(f"vertices:   {a.point_count:,}")
    if a.face_count is not None:
        print(f"faces:      {a.face_count:,}")
    if a.has_opacity or a.has_covariance:
        print(f"opacity:    {'yes' if a.has_opacity else 'no'}")
        print(f"covariance: {'yes (scale + rotation)' if a.has_covariance else 'no'}")
    if a.is_manifold is not None:
        print(f"manifold:   {'yes' if a.is_manifold else 'no'}")

    is_splat = a.has_covariance or a.has_opacity

    if is_splat:
        print("type:       gaussian splat point cloud")
        print("path:       server  (pip install chitin[splat])")
        print("reason:     splat data requires Poisson reconstruction (Open3D)")
    elif a.face_count is None:
        print("type:       plain point cloud")
        print("path:       server  (pip install chitin[splat])")
        print("reason:     point cloud requires Poisson reconstruction (Open3D)")
    elif a.is_skinned:
        print("type:       mesh (possibly rigged)")
        if not a.is_manifold:
            print("path:       server  (pip install chitin)")
            print("reason:     rigged or non-manifold mesh needs Python pipeline")
        else:
            _print_mesh_path(a.is_manifold)
    elif a.format in ("usd", "usda", "usdc"):
        print("type:       mesh (USD)")
        print("path:       server  (pip install chitin)")
        print("reason:     USD input requires Python pipeline")
    else:
        print("type:       mesh")
        _print_mesh_path(a.is_manifold or False)

    if a.is_environment_likely:
        print(
            "hint:       point distribution looks like an environment scan "
            "(hollow shell) — consider --thin-shell --proximity-filter 5.0"
        )


def _print_mesh_path(is_watertight: bool) -> None:
    if is_watertight:
        print("path:       either")
        print("  server:   pip install chitin")
        print("  browser:  npm install @autarkis/chitin-lite")
        print("reason:     manifold mesh, eligible for browser-side decomposition")
    else:
        print("path:       server  (pip install chitin)")
        print("reason:     non-manifold mesh (browser path requires watertight input)")


def _print_hull_table(label: str, hulls: list) -> None:
    print(f"{label}: {len(hulls)} hulls")
    for i, h in enumerate(hulls):
        line = f"  hull {i}: {len(h.vertices)} verts, {len(h.indices) // 3} tris"
        if h.bone_index is not None:
            line += f", bone {h.bone_index}"
        aabb_size = h.aabb_max - h.aabb_min
        line += f", size [{aabb_size[0]:.3f}, {aabb_size[1]:.3f}, {aabb_size[2]:.3f}]"
        print(line)


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
    if pf.lod_tiers:
        print(f"lod_tiers:  {len(pf.lod_tiers)}")

    print()
    _print_hull_table("LOD 0", pf.hulls)

    if pf.lod_tiers:
        for t, tier in enumerate(pf.lod_tiers):
            print()
            _print_hull_table(
                f"LOD {t + 1} (concavity={tier.concavity:.3f})", tier.hulls
            )


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


def _add_probe_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("probe", help="raycast coverage probe for collision quality")
    p.add_argument("file", type=Path, help="path to .phys file")
    p.add_argument(
        "--grid",
        type=int,
        default=64,
        help="Grid resolution per axis (default: 64, total rays = grid^2)",
    )
    p.add_argument(
        "--capsule-radius",
        type=float,
        default=0.3,
        help="Character capsule radius for gap classification (default: 0.3)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write detailed results to JSON",
    )
    p.add_argument("-q", "--quiet", action="store_true")


def _cmd_probe(args: argparse.Namespace) -> None:
    import time

    from chitin.verify.probe import probe

    if not args.quiet:
        print(f"chitin probe: {args.file} ({args.grid}x{args.grid} grid)")

    t0 = time.monotonic()
    result = probe(
        args.file, grid_resolution=args.grid, capsule_radius=args.capsule_radius
    )
    dt = time.monotonic() - t0

    pct = result.coverage * 100
    print(f"coverage:   {pct:.1f}% ({result.hits}/{result.total_rays} rays)")
    print(f"confidence: {result.confidence}")
    print(f"gaps:       {result.misses} rays in {result.gap_clusters} clusters")
    print(f"capsule:    {result.capsule_radius}m radius")
    print(f"time:       {dt:.2f}s")

    if args.output:
        result.to_json(args.output)
        if not args.quiet:
            print(f"wrote:      {args.output}")

    if result.confidence == "low":
        sys.exit(2)


def _add_sweep_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "sweep", help="capsule traversability test for collision quality"
    )
    p.add_argument("file", type=Path, help="path to .phys file")
    p.add_argument(
        "--grid",
        type=int,
        default=32,
        help="Grid resolution per axis (default: 32)",
    )
    p.add_argument(
        "--capsule-radius",
        type=float,
        default=0.3,
        help="Capsule radius in meters (default: 0.3)",
    )
    p.add_argument(
        "--capsule-height",
        type=float,
        default=1.8,
        help="Capsule height in meters (default: 1.8)",
    )
    p.add_argument(
        "--step-height",
        type=float,
        default=0.3,
        help="Max traversable step height in meters (default: 0.3)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write detailed results to JSON",
    )
    p.add_argument("-q", "--quiet", action="store_true")


def _cmd_sweep(args: argparse.Namespace) -> None:
    import time

    from chitin.verify.sweep import sweep

    if not args.quiet:
        print(
            f"chitin sweep: {args.file} "
            f"({args.grid}x{args.grid} grid, "
            f"capsule {args.capsule_radius}m x {args.capsule_height}m)"
        )

    t0 = time.monotonic()
    result = sweep(
        args.file,
        grid_resolution=args.grid,
        capsule_radius=args.capsule_radius,
        capsule_height=args.capsule_height,
        step_height=args.step_height,
    )
    dt = time.monotonic() - t0

    pct = result.traversability * 100
    print(f"ground:     {result.ground_cells}/{result.total_cells} cells")
    print(f"traversable: {pct:.1f}% (largest island: {result.largest_component} cells)")
    print(f"islands:    {result.connected_components}")
    if result.seam_snags:
        print(f"seam snags: {len(result.seam_snags)}")
    print(f"rating:     {result.rating}")
    print(f"time:       {dt:.2f}s")

    if args.output:
        result.to_json(args.output)
        if not args.quiet:
            print(f"wrote:      {args.output}")

    if result.rating == "poor":
        sys.exit(2)


def _add_convert_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "convert",
        help="convert FBX to GLB via Blender headless (requires Blender on PATH)",
    )
    p.add_argument("input", type=Path, help="Input .fbx file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .glb path (default: same name with .glb extension)",
    )
    p.add_argument("-q", "--quiet", action="store_true")


def _cmd_convert(args: argparse.Namespace) -> None:
    from chitin.convert import convert_fbx_to_glb

    input_path = Path(args.input)
    if input_path.suffix.lower() != ".fbx":
        print("chitin: convert currently supports FBX input only", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or input_path.with_suffix(".glb")

    if not args.quiet:
        print(f"chitin: {input_path} -> {output_path} (via Blender)")

    convert_fbx_to_glb(input_path, output_path)

    if not args.quiet:
        print(f"chitin: wrote {output_path}")


if __name__ == "__main__":
    main()
