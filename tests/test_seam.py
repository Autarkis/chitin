from chitin.verify.seam import dedup_snags, find_hull_seam_snags


def test_step_between_adjacent_hulls_detected(box_hull):
    # Two ground slabs meeting at x=1 with a 1.0 vertical step.
    low = box_hull(center=(0.0, 0.25, 0.0), half=(1.0, 0.25, 2.0))
    high = box_hull(center=(2.0, 0.75, 0.0), half=(1.0, 0.75, 2.0))
    snags = find_hull_seam_snags([low, high], step_height=0.3)
    assert len(snags) > 0
    # Snags cluster at the boundary between the slabs.
    assert all(abs(s[0] - 1.0) < 0.5 for s in snags)


def test_aligned_hulls_no_snags(box_hull):
    low = box_hull(center=(0.0, 0.25, 0.0), half=(1.0, 0.25, 2.0))
    level = box_hull(center=(2.0, 0.25, 0.0), half=(1.0, 0.25, 2.0))
    snags = find_hull_seam_snags([low, level], step_height=0.3)
    assert snags == []


def test_step_below_threshold_ignored(box_hull):
    low = box_hull(center=(0.0, 0.25, 0.0), half=(1.0, 0.25, 2.0))
    slightly_high = box_hull(center=(2.0, 0.35, 0.0), half=(1.0, 0.35, 2.0))
    snags = find_hull_seam_snags([low, slightly_high], step_height=0.3)
    assert snags == []


def test_empty_hull_list():
    assert find_hull_seam_snags([]) == []


def test_dedup_snags_merges_nearby():
    snags = [(0.0, 1.0, 0.0), (0.05, 1.0, 0.05), (5.0, 1.0, 5.0)]
    result = dedup_snags(snags, radius=0.3)
    assert len(result) == 2
