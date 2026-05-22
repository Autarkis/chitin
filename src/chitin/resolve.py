from __future__ import annotations

import math
from dataclasses import dataclass, field

from chitin.analyze import InputAnalysis
from chitin.config import Config


def _auto_poisson_depth(n_points: int) -> int:
    if n_points < 1:
        return 4
    return max(4, min(7, math.floor(math.log2(n_points) / 3)))


@dataclass(frozen=True)
class ResolvedConfig:
    concavity: float
    opacity_threshold: float
    poisson_depth: int
    min_hull_vertices: int
    max_hulls: int
    opacity_is_logit: bool
    coacd_preprocess_mode: str
    coacd_preprocess_resolution: int
    max_decompose_vertices: int
    lod_concavities: list[float] | None
    splat_scale_is_log: bool
    splat_surface_ratio: float
    spatial_split_threshold: int
    poisson_density_quantile: float
    surface_proximity_filter: float
    thin_shell: bool
    thin_shell_thickness: float
    flatness_threshold: float
    auto_environment: bool
    seam_repair: bool

    use_spatial_split: bool
    use_seam_repair: bool
    pipeline_path: str

    decisions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "concavity": self.concavity,
            "opacity_threshold": self.opacity_threshold,
            "poisson_depth": self.poisson_depth,
            "min_hull_vertices": self.min_hull_vertices,
            "max_hulls": self.max_hulls,
            "opacity_is_logit": self.opacity_is_logit,
            "coacd_preprocess_mode": self.coacd_preprocess_mode,
            "coacd_preprocess_resolution": self.coacd_preprocess_resolution,
            "max_decompose_vertices": self.max_decompose_vertices,
            "lod_concavities": self.lod_concavities,
            "splat_scale_is_log": self.splat_scale_is_log,
            "splat_surface_ratio": self.splat_surface_ratio,
            "spatial_split_threshold": self.spatial_split_threshold,
            "poisson_density_quantile": self.poisson_density_quantile,
            "surface_proximity_filter": self.surface_proximity_filter,
            "thin_shell": self.thin_shell,
            "thin_shell_thickness": self.thin_shell_thickness,
            "flatness_threshold": self.flatness_threshold,
            "auto_environment": self.auto_environment,
            "seam_repair": self.seam_repair,
            "use_spatial_split": self.use_spatial_split,
            "use_seam_repair": self.use_seam_repair,
            "pipeline_path": self.pipeline_path,
            "decisions": dict(self.decisions),
        }


def resolve_config(config: Config, analysis: InputAnalysis) -> ResolvedConfig:
    decisions: dict[str, str] = {}

    if config.poisson_depth is None:
        depth = _auto_poisson_depth(analysis.point_count)
        decisions["poisson_depth"] = (
            f"auto: depth {depth} from {analysis.point_count} points"
        )
    else:
        depth = config.poisson_depth
        decisions["poisson_depth"] = f"user: {config.poisson_depth}"

    thin_shell = config.thin_shell
    proximity_filter = config.surface_proximity_filter
    if config.auto_environment and analysis.is_environment_likely:
        decisions["is_environment"] = (
            f"detected: inner density ratio {analysis.inner_density_ratio:.3f}"
        )
        if not config.thin_shell:
            thin_shell = True
            decisions["thin_shell"] = "auto: environment detected"
        if config.surface_proximity_filter == 0.0:
            proximity_filter = 5.0
            decisions["surface_proximity_filter"] = (
                "auto: environment detected, set to 5.0"
            )

    opacity_is_logit = config.opacity_is_logit
    if analysis.opacity_is_logit and not config.opacity_is_logit:
        opacity_is_logit = True
        decisions["opacity_is_logit"] = "auto: detected logit-space values in input"

    use_spatial_split = (
        analysis.has_covariance
        and analysis.point_count > config.spatial_split_threshold
        and config.splat_surface_ratio > 0
    )
    decisions["use_spatial_split"] = (
        f"yes: {analysis.point_count} points > {config.spatial_split_threshold} threshold"
        if use_spatial_split
        else "no"
    )

    use_seam_repair = config.seam_repair and use_spatial_split
    if use_spatial_split and not config.seam_repair:
        decisions["use_seam_repair"] = "no: disabled by config"
    elif not use_spatial_split:
        decisions["use_seam_repair"] = "no: no spatial split"
    else:
        decisions["use_seam_repair"] = "yes"

    if analysis.is_skinned:
        pipeline_path = "rigged"
    elif analysis.has_covariance:
        pipeline_path = "splat"
    elif analysis.format in ("usd", "usda", "usdc"):
        pipeline_path = "usd"
    else:
        pipeline_path = "mesh"
    decisions["pipeline_path"] = pipeline_path

    return ResolvedConfig(
        concavity=config.concavity,
        opacity_threshold=config.opacity_threshold,
        poisson_depth=depth,
        min_hull_vertices=config.min_hull_vertices,
        max_hulls=config.max_hulls,
        opacity_is_logit=opacity_is_logit,
        coacd_preprocess_mode=config.coacd_preprocess_mode,
        coacd_preprocess_resolution=config.coacd_preprocess_resolution,
        max_decompose_vertices=config.max_decompose_vertices,
        lod_concavities=config.lod_concavities,
        splat_scale_is_log=config.splat_scale_is_log,
        splat_surface_ratio=config.splat_surface_ratio,
        spatial_split_threshold=config.spatial_split_threshold,
        poisson_density_quantile=config.poisson_density_quantile,
        surface_proximity_filter=proximity_filter,
        thin_shell=thin_shell,
        thin_shell_thickness=config.thin_shell_thickness,
        flatness_threshold=config.flatness_threshold,
        auto_environment=config.auto_environment,
        seam_repair=config.seam_repair,
        use_spatial_split=use_spatial_split,
        use_seam_repair=use_seam_repair,
        pipeline_path=pipeline_path,
        decisions=decisions,
    )
