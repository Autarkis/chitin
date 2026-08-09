from __future__ import annotations

from dataclasses import dataclass

from chitin._shared_constants import (
    COACD_CONCAVITY_THRESHOLD,
    COACD_PREPROCESS_RESOLUTION,
    NATIVE_MIN_HULL_VERTICES,
)

# Hard-kill budget for a single CoACD call. It is a backstop against a native
# stall that never finishes (the manifold-remesh explosion on non-watertight
# input), not a quality knob: a budget low enough to cut off legitimate work
# silently substitutes a bounding box, which is worse than waiting. A concave
# asset can take minutes with `coacd_deterministic` on, so the backstop is
# generous.
DEFAULT_COACD_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class Config:
    concavity: float = COACD_CONCAVITY_THRESHOLD
    opacity_threshold: float = 0.5
    poisson_depth: int | None = None
    min_hull_vertices: int = NATIVE_MIN_HULL_VERTICES
    max_hulls: int = 2048  # per decomposition unit (per octree cell / per bone)
    opacity_is_logit: bool = False
    coacd_preprocess_mode: str = "auto"
    coacd_preprocess_resolution: int = COACD_PREPROCESS_RESOLUTION
    coacd_adaptive_preprocess: bool = True
    # CoACD's search is multithreaded, and its thread scheduling changes which
    # decomposition it settles on: the same mesh yields different hull counts
    # and different bytes run to run. Pinning it to one thread is what makes
    # "same input bytes + same config = same output bytes" true, at 2-4x the
    # wall time on concave assets. Off trades reproducible output for speed.
    coacd_deterministic: bool = True
    coacd_timeout: float = DEFAULT_COACD_TIMEOUT_SECONDS
    max_decompose_vertices: int = 200_000
    lod_concavities: list[float] | None = None
    splat_scale_is_log: bool = True
    splat_surface_ratio: float = 0.2
    spatial_split_threshold: int = 50_000
    poisson_density_quantile: float = 0.1
    surface_proximity_filter: float | None = None
    thin_shell: bool = False
    thin_shell_thickness: float = 0.0
    flatness_threshold: float = 0.9
    auto_environment: bool = True
    force_environment: bool = False
    seam_repair: bool = True
    snug_fit: bool = False
    target_height: float | None = None
    target_footprint: float | None = None
    up_axis: int = 1
    flat_aspect_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.concavity <= 0:
            raise ValueError(f"concavity must be positive, got {self.concavity}")
        if not 0 <= self.opacity_threshold <= 1:
            raise ValueError(
                f"opacity_threshold must be in [0, 1], got {self.opacity_threshold}"
            )
        if self.poisson_depth is not None and not 1 <= self.poisson_depth <= 12:
            raise ValueError(
                f"poisson_depth must be in [1, 12], got {self.poisson_depth}"
            )
        if self.min_hull_vertices < 4:
            raise ValueError(
                f"min_hull_vertices must be >= 4, got {self.min_hull_vertices}"
            )
        if self.max_hulls < 1:
            raise ValueError(f"max_hulls must be >= 1, got {self.max_hulls}")
        if self.coacd_preprocess_mode not in ("auto", "on", "off"):
            raise ValueError(
                "coacd_preprocess_mode must be one of auto, on, or off, "
                f"got {self.coacd_preprocess_mode!r}"
            )
        if self.coacd_preprocess_resolution < 1:
            raise ValueError(
                f"coacd_preprocess_resolution must be >= 1, "
                f"got {self.coacd_preprocess_resolution}"
            )
        if self.coacd_timeout <= 0:
            raise ValueError(
                f"coacd_timeout must be positive, got {self.coacd_timeout}"
            )
        if not 0 <= self.poisson_density_quantile <= 1:
            raise ValueError(
                f"poisson_density_quantile must be in [0, 1], "
                f"got {self.poisson_density_quantile}"
            )
        if self.splat_surface_ratio < 0:
            raise ValueError(
                f"splat_surface_ratio must be >= 0, got {self.splat_surface_ratio}"
            )
        if self.spatial_split_threshold < 1:
            raise ValueError(
                f"spatial_split_threshold must be >= 1, "
                f"got {self.spatial_split_threshold}"
            )
        if (
            self.surface_proximity_filter is not None
            and self.surface_proximity_filter < 0
        ):
            raise ValueError(
                f"surface_proximity_filter must be >= 0, "
                f"got {self.surface_proximity_filter}"
            )
        if self.thin_shell_thickness < 0:
            raise ValueError(
                f"thin_shell_thickness must be >= 0, got {self.thin_shell_thickness}"
            )
        if not 0 <= self.flatness_threshold <= 1:
            raise ValueError(
                f"flatness_threshold must be in [0, 1], got {self.flatness_threshold}"
            )
        if self.max_decompose_vertices < 100:
            raise ValueError(
                f"max_decompose_vertices must be >= 100, "
                f"got {self.max_decompose_vertices}"
            )
        if self.lod_concavities is not None:
            for c in self.lod_concavities:
                # LOD0 is the base concavity (highest detail); every additional
                # tier must be strictly coarser, i.e. a larger concavity.
                if c <= self.concavity:
                    raise ValueError(
                        "lod_concavities must be greater than the base concavity "
                        f"({self.concavity}) so each tier is coarser than LOD0, got {c}"
                    )
        if self.target_height is not None and self.target_height <= 0:
            raise ValueError(
                f"target_height must be positive, got {self.target_height}"
            )
        if self.target_footprint is not None and self.target_footprint <= 0:
            raise ValueError(
                f"target_footprint must be positive, got {self.target_footprint}"
            )
        if self.up_axis not in (0, 1, 2):
            raise ValueError(f"up_axis must be one of 0, 1, 2, got {self.up_axis}")
        if self.flat_aspect_ratio <= 0:
            raise ValueError(
                f"flat_aspect_ratio must be positive, got {self.flat_aspect_ratio}"
            )
