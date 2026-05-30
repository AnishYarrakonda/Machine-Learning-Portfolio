import math
import pytest
from inference_pursuit.geometry import torus_dist_1d, torus_dist


def test_same_cell():
    assert torus_dist_1d(3, 3, 10) == 0.0


def test_adjacent():
    assert torus_dist_1d(0, 1, 10) == 1.0
    assert torus_dist_1d(1, 0, 10) == 1.0


def test_wraps():
    # 0 and 9 on a size-10 grid are 1 apart
    assert torus_dist_1d(0, 9, 10) == 1.0
    assert torus_dist_1d(9, 0, 10) == 1.0


def test_symmetric_1d():
    assert torus_dist_1d(2, 7, 10) == torus_dist_1d(7, 2, 10)


def test_diagonal():
    # (0,0) to (3,4): 1D dists 3 and 4 → Euclidean 5
    assert abs(torus_dist((0, 0), (3, 4), 20) - 5.0) < 1e-9


def test_wraps_diagonal():
    # (0,0) to (9,9) on size-10: 1D dists 1 and 1 → sqrt(2)
    assert abs(torus_dist((0, 0), (9, 9), 10) - math.sqrt(2)) < 1e-9


def test_symmetric_2d():
    assert abs(torus_dist((1, 3), (7, 2), 10) - torus_dist((7, 2), (1, 3), 10)) < 1e-9
