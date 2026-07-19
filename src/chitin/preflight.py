from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class SystemInfo:
    cores: int
    ram_gb: float
    platform: str


@dataclass(frozen=True)
class PreflightResult:
    level: str  # "green", "yellow", "red"
    message: str | None
    estimated_minutes: float | None
    hints: list[str] | None = None


def get_system_info() -> SystemInfo:
    return SystemInfo(
        cores=os.cpu_count() or 1,
        ram_gb=psutil.virtual_memory().total / (1024**3),
        platform=sys.platform,
    )


def estimate_input_size(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return _estimate_ply_vertices(path)
    file_size = path.stat().st_size
    return int(file_size / 32)


def _estimate_ply_vertices(path: Path) -> int:
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


# Baseline: plant_330k.ply (184k filtered verts) = ~5 min on 10-core M1 Pro / 16GB
BASELINE_VERTS = 184_000
BASELINE_MINUTES = 5.0
BASELINE_CORES = 10

RAM_PER_100K_VERTS_GB = 1.5

YELLOW_MINUTES = 10.0
RED_RAM_HEADROOM = 0.8


def check(path: Path, sys_info: SystemInfo | None = None) -> PreflightResult:
    info = sys_info or get_system_info()
    vertex_count = estimate_input_size(path)

    if vertex_count == 0:
        return PreflightResult(level="green", message=None, estimated_minutes=None)

    scale = vertex_count / BASELINE_VERTS
    core_factor = BASELINE_CORES / max(info.cores, 1)
    est_minutes = BASELINE_MINUTES * scale * core_factor

    ram_needed = (vertex_count / 100_000) * RAM_PER_100K_VERTS_GB
    ram_available = info.ram_gb * RED_RAM_HEADROOM

    if ram_needed > ram_available:
        return PreflightResult(
            level="red",
            message=(
                f"{vertex_count:,} vertices needs ~{ram_needed:.1f}GB RAM "
                f"but only {info.ram_gb:.1f}GB available — "
                f"use --force to run anyway, or reduce input size"
            ),
            estimated_minutes=est_minutes,
        )

    if est_minutes > YELLOW_MINUTES:
        return PreflightResult(
            level="yellow",
            message=(
                f"{vertex_count:,} vertices on {info.cores} cores / "
                f"{info.ram_gb:.1f}GB RAM — estimated {est_minutes:.0f}+ min, "
                f"this may take a while"
            ),
            estimated_minutes=est_minutes,
        )

    hints = detect_environment_hints(path)
    return PreflightResult(
        level="green", message=None, estimated_minutes=est_minutes, hints=hints
    )


def detect_environment_hints(path: Path) -> list[str] | None:
    if path.suffix.lower() != ".ply":
        return None
    try:
        from chitin.adapters.ply_reader import read_ply_vertex

        vertex = read_ply_vertex(path)
        names = set(vertex.data.dtype.names)
        if not all(c in names for c in ("x", "y", "z")):
            return None

        positions = _sample_positions(vertex, max_samples=20_000)
        if len(positions) < 100:
            return None

        return _classify_point_distribution(positions)
    except Exception:
        return None


def _sample_positions(vertex_data, max_samples: int):
    import numpy as np

    n = len(vertex_data)
    if n <= max_samples:
        idx = slice(None)
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)

    x = np.array(vertex_data["x"][idx], dtype=np.float64)
    y = np.array(vertex_data["y"][idx], dtype=np.float64)
    z = np.array(vertex_data["z"][idx], dtype=np.float64)
    return np.stack([x, y, z], axis=1)


def _classify_point_distribution(positions) -> list[str] | None:
    import numpy as np

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    extent = bbox_max - bbox_min
    bbox_vol = float(np.prod(extent))
    if bbox_vol <= 0:
        return None

    center = (bbox_min + bbox_max) / 2
    half = extent / 2
    half = np.where(half == 0, 1.0, half)
    normalized = np.abs(positions - center) / half

    shell_fraction = float(np.mean(np.max(normalized, axis=1) > 0.7))

    aspect = extent / max(extent.max(), 1e-12)
    is_flat = float(aspect.min()) < 0.15

    hints = []
    if shell_fraction > 0.6:
        hints.append(
            "point distribution looks like an environment scan "
            "(hollow shell) — consider --thin-shell --proximity-filter 5.0"
        )
    if is_flat and not hints:
        hints.append(
            "scene is very flat — the flatness detector (--flatness-threshold) "
            "should handle this, but consider --thin-shell for indoor scans"
        )

    return hints if hints else None
