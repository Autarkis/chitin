"""Diagnose Policy 0.2.0 failures promoted from the spent holdout corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chitin.f32_adversarial import PlaneCase, classify_plane_exact
from chitin.f32_policy import POLICY_0_2_0
from chitin.f32_predicates import (
    _grid_quantization_bound,
    _to_grid_frame,
    classify_plane_f32,
    clip_mesh_f32,
    clip_mesh_f64,
    diff_clips,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "tests" / "fixtures" / "traces" / "holdout_failures_0_2_0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Separate ambiguity-band misses from errors inside Policy 0.2's "
            "unquantized-f32 fallback."
        )
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path)
    return parser


def _point_and_normal(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    normal = data["plane_normal"].astype(np.float64)
    normal /= np.linalg.norm(normal)
    offset = float(data["plane_offset"])
    point = normal * (-offset / np.dot(normal, normal))
    return point, normal


def diagnose_clip(path: Path, fixture: str, clip_index: int) -> dict:
    with np.load(path) as data:
        vertices = data["vertices"].astype(np.float64)
        faces = data["faces"].astype(np.int32)
        point, normal = _point_and_normal(data)

    case = PlaneCase(vertices, point, normal, label=path.name)
    exact_signs = classify_plane_exact(case)
    result = classify_plane_f32(vertices, point, normal, POLICY_0_2_0)
    mismatch_indices = np.flatnonzero(result.signs != exact_signs)

    grid_vertices, grid_point, _centroid, scale = _to_grid_frame(
        vertices, point, POLICY_0_2_0
    )
    grid_normal = (normal * scale).astype(np.float32)
    grid_delta = grid_vertices.astype(np.float32) - grid_point.astype(np.float32)
    grid_dot = np.sum(grid_delta * grid_normal, axis=1)
    grid_bound = _grid_quantization_bound(grid_normal)
    ambiguous = np.abs(grid_dot) <= grid_bound

    vertices_f32 = vertices.astype(np.float32)
    point_f32 = point.astype(np.float32)
    normal_f32 = normal.astype(np.float32)
    world_dot_f32 = np.sum((vertices_f32 - point_f32) * normal_f32, axis=1)
    world_dot_f64 = np.dot(vertices - point, normal)
    f32_input_case = PlaneCase(
        vertices_f32.astype(np.float64),
        point_f32.astype(np.float64),
        normal_f32.astype(np.float64),
    )
    f32_input_exact_signs = classify_plane_exact(f32_input_case)
    canonical_clip_reference = clip_mesh_f64(
        f32_input_case.vertices,
        faces,
        f32_input_case.plane_point,
        f32_input_case.plane_normal,
    )
    canonical_clip_candidate = clip_mesh_f32(
        f32_input_case.vertices,
        faces,
        f32_input_case.plane_point,
        f32_input_case.plane_normal,
        POLICY_0_2_0,
    )
    canonical_clip_diff = diff_clips(
        canonical_clip_reference, canonical_clip_candidate, POLICY_0_2_0
    )

    mismatches = []
    for vertex_index in mismatch_indices:
        if (
            exact_signs[vertex_index] != f32_input_exact_signs[vertex_index]
            and result.signs[vertex_index] == f32_input_exact_signs[vertex_index]
        ):
            cause = "source_delta_below_f32_input_precision"
        elif not ambiguous[vertex_index]:
            cause = "ambiguity_band_miss"
        elif result.signs[vertex_index] != f32_input_exact_signs[vertex_index]:
            cause = "f32_arithmetic_error"
        else:
            cause = "unclassified"
        mismatches.append(
            {
                "vertex_index": int(vertex_index),
                "cause": cause,
                "exact_sign": int(exact_signs[vertex_index]),
                "policy_sign": int(result.signs[vertex_index]),
                "f32_input_exact_sign": int(f32_input_exact_signs[vertex_index]),
                "grid_dot": float(grid_dot[vertex_index]),
                "grid_bound": grid_bound,
                "entered_ambiguity_path": bool(ambiguous[vertex_index]),
                "world_dot_f64": float(world_dot_f64[vertex_index]),
                "world_dot_f32": float(world_dot_f32[vertex_index]),
                "vertex": vertices[vertex_index].tolist(),
                "vertex_f32": vertices_f32[vertex_index].astype(np.float64).tolist(),
                "plane_point": point.tolist(),
                "plane_point_f32": point_f32.astype(np.float64).tolist(),
            }
        )

    return {
        "file": path.name,
        "fixture": fixture,
        "clip_index": clip_index,
        "vertices": len(vertices),
        "ambiguity_path_count": result.ambiguity_path_count,
        "mismatch_count": len(mismatches),
        "canonical_f32_clip_agrees": canonical_clip_diff.agrees,
        "canonical_f32_face_set_agrees": canonical_clip_diff.details["face_set_agrees"],
        "mismatches": mismatches,
    }


def diagnose_corpus(corpus_dir: Path) -> dict:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [
        diagnose_clip(corpus_dir / entry["file"], entry["fixture"], entry["clip_index"])
        for entry in manifest["clips"]
    ]
    causes = Counter(
        mismatch["cause"] for clip in clips for mismatch in clip["mismatches"]
    )
    return {
        "policy_version": POLICY_0_2_0.version,
        "source_manifest": str(manifest_path),
        "clip_count": len(clips),
        "mismatching_vertex_count": sum(clip["mismatch_count"] for clip in clips),
        "canonical_f32_clip_agree_count": sum(
            clip["canonical_f32_clip_agrees"] for clip in clips
        ),
        "causes": dict(sorted(causes.items())),
        "clips": clips,
    }


def main() -> None:
    args = build_parser().parse_args()
    report = diagnose_corpus(args.corpus_dir)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
