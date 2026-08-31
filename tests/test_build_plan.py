import numpy as np
import pytest

from chitin import (
    Config,
    extract_from_arrays,
    extract_from_mesh,
)

try:
    import open3d  # noqa: F401

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False
requires_open3d = pytest.mark.skipif(not _HAS_OPEN3D, reason="requires chitin[splat]")


def test_mesh_plan(box_mesh):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    assert r.build_plan is not None
    assert r.build_plan.collider_kind == "static"
    assert "decompose" in r.build_plan.pipeline
    assert r.build_plan.source_vertices == len(verts)


@requires_open3d
def test_point_cloud_plan(sphere_points):
    r = extract_from_arrays(
        sphere_points, normals=sphere_points, config=Config(concavity=0.5)
    )
    assert r.build_plan is not None
    assert r.build_plan.collider_kind == "point_cloud"
    assert "reconstruct" in r.build_plan.pipeline
    assert "decompose" in r.build_plan.pipeline
    assert r.build_plan.source_vertices == len(sphere_points)


@requires_open3d
def test_point_cloud_with_opacity_plan(sphere_points):
    opacity = np.ones(len(sphere_points), dtype=np.float64)
    r = extract_from_arrays(
        sphere_points,
        opacity=opacity,
        normals=sphere_points,
        config=Config(concavity=0.5),
    )
    assert r.build_plan is not None
    assert "opacity_filter" in r.build_plan.pipeline
    assert r.build_plan.detected.get("filtered_vertices") is not None


@requires_open3d
def test_point_cloud_normal_estimation_plan(sphere_points):
    r = extract_from_arrays(sphere_points, config=Config(concavity=0.5))
    assert r.build_plan is not None
    assert "normal_estimation" in r.build_plan.pipeline


def test_rigged_plan(rigged_result):
    r = rigged_result
    assert r.build_plan is not None
    assert r.build_plan.collider_kind == "rigged"
    assert "segment_by_bone" in r.build_plan.pipeline
    assert "per_bone_decompose" in r.build_plan.pipeline
    assert r.build_plan.detected["bone_count"] == 2
    assert r.build_plan.detected["segment_count"] == 2


def test_signals_merge_counters_by_sum_and_flags_by_and():
    # Octree cells and bones report upward across a process boundary. Counters
    # add; the reproducibility flag ANDs, so one piece decomposed unpinned is
    # not averaged away by the pieces around it.
    from chitin.plan import BuildPlan

    parent = BuildPlan(input_kind="splat", collider_kind="static")
    clean = BuildPlan(input_kind="splat", collider_kind="static")
    clean.detected["coacd_deterministic"] = True
    clean.detected["fallback_hulls"] = 1
    dirty = BuildPlan(input_kind="splat", collider_kind="static")
    dirty.detected["coacd_deterministic"] = False
    dirty.detected["fallback_hulls"] = 2

    parent.merge_signals(clean.child_signals())
    assert parent.detected["coacd_deterministic"] is True
    parent.merge_signals(dirty.child_signals())

    assert parent.detected["fallback_hulls"] == 3
    assert parent.detected["coacd_deterministic"] is False


def test_decomposition_records_whether_it_was_reproducible(box_mesh):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    assert r.build_plan.detected["coacd_deterministic"] is True

    fast = extract_from_mesh(
        verts, faces, config=Config(concavity=0.5, coacd_deterministic=False)
    )
    assert fast.build_plan.detected["coacd_deterministic"] is False


def test_plan_to_dict(box_mesh):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    d = r.build_plan.to_dict()
    assert d["input_kind"] == "mesh"
    assert d["collider_kind"] == "static"
    assert isinstance(d["pipeline"], list)
    assert isinstance(d["detected"], dict)
