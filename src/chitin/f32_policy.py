from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_UNSET = -1


@dataclass
class QuantizationPolicy:
    """Versioned quantization parameters for f32-safe geometric predicates."""

    version: str = "0.1.0"
    grid_bits: int = 20
    classification_ulp_margin: int = 0
    intersection_snap_bits: int = _UNSET
    winding_check: bool = True
    ambiguity_fallback: bool = False
    canonical_f32_inputs: bool = False

    def __post_init__(self) -> None:
        if self.intersection_snap_bits == _UNSET:
            self.intersection_snap_bits = self.grid_bits

    @property
    def grid_scale(self) -> int:
        """Size of the integer quantization grid along one axis."""
        return 2**self.grid_bits

    def normalize_to_grid(
        self, vertices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Center, unit-cube-scale, and quantize vertices onto the integer grid."""
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        # epsilon guard only prevents divide-by-zero for a degenerate (single-point)
        # input; it is not a geometric tolerance, per the scale-relative policy.
        extent = max(float(np.abs(centered).max()), 1e-30)
        scale_factor = 1.0 / (2.0 * extent)
        unit_coords = centered * scale_factor
        grid_coords = unit_coords * self.grid_scale
        return grid_coords.astype(np.int64), centroid, scale_factor

    def classify_sign(self, dot_products: np.ndarray) -> np.ndarray:
        """Classify sign of dot products, snapping a grid-unit margin around zero to 0."""
        margin = self.classification_ulp_margin
        result = np.sign(dot_products).astype(np.int8)
        result[np.abs(dot_products) <= margin] = 0
        return result


DEFAULT_POLICY = QuantizationPolicy()

POLICY_0_2_0 = QuantizationPolicy(
    version="0.2.0",
    grid_bits=20,
    classification_ulp_margin=0,
    intersection_snap_bits=20,
    ambiguity_fallback=True,
)

POLICY_0_3_0 = QuantizationPolicy(
    version="0.3.0",
    grid_bits=20,
    classification_ulp_margin=0,
    intersection_snap_bits=20,
    winding_check=True,
    ambiguity_fallback=True,
    canonical_f32_inputs=True,
)


def sweep_policies(grid_bits_range: range) -> list[QuantizationPolicy]:
    """Build one default policy per grid_bits value in the given range."""
    return [QuantizationPolicy(grid_bits=bits) for bits in grid_bits_range]
