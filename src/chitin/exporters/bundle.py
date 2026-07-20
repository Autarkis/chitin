from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitin.acceptance import Verdict
    from chitin.config import Config
    from chitin.result import ExtractionResult


def export_bundle(
    result: ExtractionResult,
    output_dir: str | Path,
    fmt: str = "phys",
    scene_name: str = "scene",
    input_path: str | Path | None = None,
    config: Config | None = None,
    verdict: Verdict | None = None,
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

    _write_manifest(output_dir, result, input_path=input_path, config=config,
                    verdict=verdict)

    return primary


def _write_manifest(
    output_dir: Path,
    result: ExtractionResult,
    *,
    input_path: str | Path | None,
    config: Config | None,
    verdict: Verdict | None,
) -> None:
    # Written last so it can hash every other file the bundle emitted.
    from chitin.acceptance import report_metrics
    from chitin.manifest import MANIFEST_FILENAME, quality_warnings, write_manifest

    output_files = [
        p.name
        for p in sorted(output_dir.iterdir())
        if p.is_file() and p.name != MANIFEST_FILENAME
    ]
    resolved_dict = (
        result.resolved.to_dict()
        if result.resolved is not None and hasattr(result.resolved, "to_dict")
        else None
    )
    write_manifest(
        output_dir,
        output_files=output_files,
        input_path=input_path,
        config_dict=dataclasses.asdict(config) if config is not None else None,
        resolved_dict=resolved_dict,
        metrics=report_metrics(result),
        warnings=quality_warnings(result),
        verdict=verdict.to_dict() if verdict is not None else None,
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))
