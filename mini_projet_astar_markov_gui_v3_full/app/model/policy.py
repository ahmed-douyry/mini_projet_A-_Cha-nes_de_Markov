from __future__ import annotations
from typing import Dict, List
from .grid import Pos
Action = str  # 'U','D','L','R','S'

def action_from_step(a: Pos, b: Pos) -> Action:
    dr, dc = b.r - a.r, b.c - a.c
    if dr == -1 and dc == 0: return 'U'
    if dr == 1 and dc == 0: return 'D'
    if dr == 0 and dc == -1: return 'L'
    if dr == 0 and dc == 1: return 'R'
    return 'S'

def build_path_policy(path: List[Pos]) -> Dict[Pos, Action]:
    pol: Dict[Pos, Action] = {}
    if not path:
        return pol
    for i in range(len(path) - 1):
        pol[path[i]] = action_from_step(path[i], path[i+1])
    pol[path[-1]] = 'S'
    return pol
