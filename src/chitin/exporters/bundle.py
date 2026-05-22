from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitin.result import ExtractionResult


def export_bundle(
    result: ExtractionResult,
    output_dir: str | Path,
    fmt: str = "phys",
    scene_name: str = "scene",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "phys":
        primary = output_dir / "scene.phys"
        result.to_phys(primary)
    elif fmt == "usd":
        primary = output_dir / "scene.usda"
        result.to_usd(primary, scene_name=scene_name)
    elif fmt == "json":
        primary = output_dir / "colliders.json"
        result.to_json(primary)
    else:
        primary = output_dir / f"scene.{fmt}"

    if result.build_plan is not None:
        _write_json(output_dir / "build-plan.json", result.build_plan.to_dict())

    if result.analysis is not None and hasattr(result.analysis, "to_dict"):
        _write_json(output_dir / "analysis.json", result.analysis.to_dict())

    if result.resolved is not None and hasattr(result.resolved, "to_dict"):
        _write_json(output_dir / "resolved-config.json", result.resolved.to_dict())

    return primary


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))
