import numpy as np

from chitin.verify.raycast import ray_hull_spans
from chitin.verify.sweep import sweep_hulls


def _phys_hull(hull):
    from chitin.phys import PhysHull

    verts = np.asarray(hull.vertices, dtype=np.float32)
    return PhysHull(
        vertices=verts,
        indices=hull.indices,
        aabb_min=verts.min(axis=0),
        aabb_max=verts.max(axis=0),
    )


def _floor(box_hull):
    # 4x4 slab, top face at y=0.
    return box_hull(center=(0.0, -0.05, 0.0), half=(2.0, 0.05, 2.0))


def test_hull_spans_report_entry_and_exit(box_hull):
    hull = _phys_hull(box_hull(center=(0.0, 0.5, 0.0), half=(1.0, 0.5, 1.0)))
    origins = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    near, far = ray_hull_spans(origins, direction, [hull])

    assert np.isclose(near[0, 0], 4.0)  # enters at y=1.0
    assert np.isclose(far[0, 0], 5.0)  # exits at y=0.0


def test_hull_spans_miss_is_infinite(box_hull):
    hull = _phys_hull(box_hull(center=(0.0, 0.5, 0.0), half=(1.0, 0.5, 1.0)))
    origins = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    near, far = ray_hull_spans(origins, direction, [hull])

    assert not np.isfinite(near[0, 0])
    assert not np.isfinite(far[0, 0])


def test_open_slab_is_fully_traversable(box_hull):
    result = sweep_hulls([_floor(box_hull)], grid_resolution=16)

    assert result.ground_cells == result.total_cells
    assert result.standable_cells == result.total_cells
    assert result.clearance_blocked == 0
    assert result.radius_blocked == 0
    assert result.traversability == 1.0
    assert result.rating == "excellent"


def test_empty_hull_list():
    result = sweep_hulls([])

    assert result.total_cells == 0
    assert result.standable_cells == 0
    assert result.traversability == 0.0


def test_low_ceiling_rejects_the_floor(box_hull):
    floor = _floor(box_hull)
    ceiling = box_hull(center=(0.0, 1.55, 0.0), half=(2.0, 0.05, 2.0))

    tall = sweep_hulls([floor, ceiling], grid_resolution=16, capsule_height=1.8)
    short = sweep_hulls([floor, ceiling], grid_resolution=16, capsule_height=1.2)

    # 1.8m capsule does not fit under a 1.5m ceiling: the floor is skipped.
    assert tall.clearance_blocked == tall.standable_cells
    # 1.2m capsule does, so the floor stays the ground layer.
    assert short.clearance_blocked == 0
    assert short.traversability == 1.0


def test_headroom_ignores_the_gap_inside_a_decomposed_solid(box_hull):
    # Two hulls stacked into one 1.8m column, as a convex decomposition emits.
    lower = box_hull(center=(0.0, 0.45, 0.0), half=(2.0, 0.45, 2.0))
    upper = box_hull(center=(0.0, 1.35, 0.0), half=(2.0, 0.45, 2.0))

    result = sweep_hulls([lower, upper], grid_resolution=16, capsule_height=1.8)

    # The seam at y=0.9 is interior, not standable ground under a 0.9m ceiling.
    assert result.clearance_blocked == 0
    assert result.traversability == 1.0


def test_table_top_shadows_the_floor_beneath_it(box_hull):
    floor = _floor(box_hull)
    table = box_hull(center=(1.0, 0.7, 0.0), half=(0.6, 0.02, 0.6))

    result = sweep_hulls([floor, table], grid_resolution=16, capsule_height=1.8)

    # Cells under the table climb to the table top, which is a step off the floor,
    # and the floor ring around it is inside the capsule's lateral envelope.
    assert result.clearance_blocked > 0
    assert result.radius_blocked > 0
    assert result.connected_components > 1
    assert result.traversability < 1.0


def test_capsule_radius_clears_cells_beside_a_wall(box_hull):
    floor = _floor(box_hull)
    wall = box_hull(center=(0.0, 1.0, 0.0), half=(0.05, 1.0, 2.0))

    wide = sweep_hulls([floor, wall], grid_resolution=16, capsule_radius=0.4)
    thin = sweep_hulls([floor, wall], grid_resolution=16, capsule_radius=0.05)

    assert wide.radius_blocked > thin.radius_blocked
    assert wide.standable_cells < wide.ground_cells
    # The wall splits the slab, so the floor is not one island.
    assert wide.connected_components > 1
    assert wide.traversability < 1.0


def test_radius_ignores_obstacles_below_step_height(box_hull):
    floor = _floor(box_hull)
    curb = box_hull(center=(0.0, 0.1, 0.0), half=(0.05, 0.1, 2.0))

    result = sweep_hulls(
        [floor, curb], grid_resolution=16, capsule_radius=0.4, step_height=0.3
    )

    assert result.radius_blocked == 0


def test_json_round_trip_carries_the_new_counters(box_hull, tmp_path):
    import json

    floor = _floor(box_hull)
    wall = box_hull(center=(0.0, 1.0, 0.0), half=(0.05, 1.0, 2.0))
    result = sweep_hulls([floor, wall], grid_resolution=16)

    out = tmp_path / "sweep.json"
    result.to_json(out)
    data = json.loads(out.read_text())

    assert data["standable_cells"] == result.standable_cells
    assert data["radius_blocked"] == result.radius_blocked
    assert data["clearance_blocked"] == result.clearance_blocked
