import math
from typing import Tuple


def torus_dist_1d(a: int, b: int, N: int) -> float:
    d = abs(a - b)
    return float(min(d, N - d))


def torus_dist(pos_a: Tuple[int, int], pos_b: Tuple[int, int], N: int) -> float:
    dx = torus_dist_1d(pos_a[0], pos_b[0], N)
    dy = torus_dist_1d(pos_a[1], pos_b[1], N)
    return math.sqrt(dx * dx + dy * dy)
