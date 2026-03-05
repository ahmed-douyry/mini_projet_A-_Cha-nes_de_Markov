from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict

Pos = Tuple[int, int]  # (r, c)

@dataclass
class Grid:
    rows: int
    cols: int
    obstacles: Set[Pos]
    start: Pos
    goal: Pos

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, p: Pos) -> bool:
        return self.in_bounds(p) and (p not in self.obstacles)

    def neighbors4(self, p: Pos) -> List[Pos]:
        r, c = p
        cand = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [q for q in cand if self.is_free(q)]

    def all_free_cells(self) -> List[Pos]:
        free: List[Pos] = []
        for r in range(self.rows):
            for c in range(self.cols):
                p = (r, c)
                if self.is_free(p):
                    free.append(p)
        return free

    def to_json(self) -> Dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "start": {"r": self.start[0], "c": self.start[1]},
            "goal": {"r": self.goal[0], "c": self.goal[1]},
            "obstacles": [{"r": r, "c": c} for (r, c) in sorted(self.obstacles)]
        }

    @staticmethod
    def from_json(d: Dict) -> "Grid":
        rows, cols = int(d["rows"]), int(d["cols"])
        start = (int(d["start"]["r"]), int(d["start"]["c"]))
        goal = (int(d["goal"]["r"]), int(d["goal"]["c"]))
        obstacles = {(int(o["r"]), int(o["c"])) for o in d.get("obstacles", [])}
        obstacles.discard(start)
        obstacles.discard(goal)
        return Grid(rows=rows, cols=cols, obstacles=obstacles, start=start, goal=goal)
