# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
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
                f"use --cloud or reduce input size"
            ),
            estimated_minutes=est_minutes,
        )

    if est_minutes > YELLOW_MINUTES:
        return PreflightResult(
            level="yellow",
            message=(
                f"{vertex_count:,} vertices on {info.cores} cores / "
                f"{info.ram_gb:.1f}GB RAM — estimated {est_minutes:.0f}+ min, "
                f"consider --cloud for large inputs"
            ),
            estimated_minutes=est_minutes,
        )

    return PreflightResult(level="green", message=None, estimated_minutes=est_minutes)
