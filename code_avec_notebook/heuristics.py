from __future__ import annotations
from typing import Tuple

Pos = Tuple[int, int]

def h_zero(p: Pos, goal: Pos) -> int:
    return 0

def manhattan(p: Pos, goal: Pos) -> int:
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
