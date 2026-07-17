from __future__ import annotations

import numpy as np


def normalize_to_target(
    positions: np.ndarray,
    *,
    target_height: float | None = None,
    target_footprint: float | None = None,
    up_axis: int = 1,
    flat_aspect_ratio: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """Uniformly scale a mesh so a real-world reference extent is matched.

    Source assets exported from 3D-FRONT / 3D-FUTURE are model-normalized, not
    metric: a king bed and a nightstand can both occupy roughly a unit cube.
    This rescales the geometry (uniform, proportion-preserving) so that the
    model's height matches ``target_height`` in meters — the most semantically
    stable dimension (seats ~0.45 m, tables ~0.75 m, beds ~0.55 m regardless of
    style).

    Genuinely flat objects (rugs, mats, ceiling panels) have negligible height,
    so matching height would explode their footprint. When the *source* geometry
    is flat (height < ``flat_aspect_ratio`` x its largest horizontal extent) and
    a ``target_footprint`` is supplied, the footprint is matched instead, keeping
    the choice of source/target dimension consistent.

    Scaling is about the origin, so a model whose origin sits on its base keeps
    its base on the ground plane; absolute placement is re-anchored downstream by
    the consuming scene layer. The scale stays uniform, so colliders
    extracted afterward inherit the correct metric size without any per-axis
    distortion.

    Returns the scaled positions and a stats dict for the build plan. A no-op
    (no target given, empty input, or degenerate extent) returns the input
    unchanged with an empty-or-explanatory stats dict.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0 or (target_height is None and target_footprint is None):
        return pos, {}

    extent = pos.max(axis=0) - pos.min(axis=0)
    height = float(extent[up_axis])
    horizontal = float(max(e for i, e in enumerate(extent) if i != up_axis))

    is_flat = horizontal > 0 and height < flat_aspect_ratio * horizontal

    if is_flat and target_footprint is not None:
        source, target, matched = horizontal, target_footprint, "footprint"
    elif target_height is not None:
        source, target, matched = height, target_height, "height"
    elif target_footprint is not None:
        # No height target, but flat-guard did not fire (or no footprint case):
        # fall back to matching the footprint.
        source, target, matched = horizontal, target_footprint, "footprint"
    else:
        return pos, {}

    if source <= 0 or target <= 0:
        return pos, {"normalized": False, "normalize_reason": "degenerate_extent"}

    scale = target / source
    stats = {
        "normalized": True,
        "normalize_matched": matched,
        "normalize_scale": scale,
        "normalize_source_extent": source,
        "normalize_target_extent": target,
        "normalize_is_flat": is_flat,
    }
    return pos * scale, stats
