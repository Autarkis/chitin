import numpy as np
import pytest

from chitin.surface_boundary import surface_loops


def test_touching_sheets_use_face_fans():
    faces = np.array([[0, 1, 2], [0, 3, 4]])
    loops = surface_loops(faces)
    assert sorted(map(tuple, loops)) == [(0, 1, 2), (0, 3, 4)]


def test_interior_diagonal_is_not_boundary():
    assert [list(loop) for loop in surface_loops(np.array([[0, 1, 2], [0, 2, 3]]))] == [
        [0, 1, 2, 3]
    ]


def test_nonmanifold_edge_fails():
    with pytest.raises(ValueError, match="nonmanifold"):
        surface_loops(np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]))
