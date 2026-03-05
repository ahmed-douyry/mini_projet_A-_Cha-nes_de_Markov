from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Dict, Any

@dataclass(frozen=True, slots=True)
class Pos:
    r: int
    c: int

class GridModel:
    def __init__(self, rows: int = 15, cols: int = 15):
        self.rows = rows
        self.cols = cols
        self.obstacles: Set[Pos] = set()
        self.start = Pos(0, 0)
        self.goal = Pos(rows - 1, cols - 1)

    def resize(self, rows: int, cols: int) -> None:
        self.rows, self.cols = rows, cols
        self.obstacles = {p for p in self.obstacles if 0 <= p.r < rows and 0 <= p.c < cols}
        self.start = Pos(min(self.start.r, rows - 1), min(self.start.c, cols - 1))
        self.goal = Pos(min(self.goal.r, rows - 1), min(self.goal.c, cols - 1))
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

    def in_bounds(self, p: Pos) -> bool:
        return 0 <= p.r < self.rows and 0 <= p.c < self.cols

    def is_free(self, p: Pos) -> bool:
        return self.in_bounds(p) and p not in self.obstacles

    def toggle_obstacle(self, p: Pos) -> None:
        if p == self.start or p == self.goal:
            return
        if p in self.obstacles:
            self.obstacles.remove(p)
        else:
            self.obstacles.add(p)

    def set_start(self, p: Pos) -> None:
        if p != self.goal and p not in self.obstacles and self.in_bounds(p):
            self.start = p

    def set_goal(self, p: Pos) -> None:
        if p != self.start and p not in self.obstacles and self.in_bounds(p):
            self.goal = p

    def neighbors4(self, p: Pos) -> List[Pos]:
        cand = [Pos(p.r-1, p.c), Pos(p.r+1, p.c), Pos(p.r, p.c-1), Pos(p.r, p.c+1)]
        return [q for q in cand if self.is_free(q)]

    def all_free_cells(self) -> List[Pos]:
        res: List[Pos] = []
        for r in range(self.rows):
            for c in range(self.cols):
                p = Pos(r, c)
                if self.is_free(p):
                    res.append(p)
        return res

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "start": {"r": self.start.r, "c": self.start.c},
            "goal": {"r": self.goal.r, "c": self.goal.c},
            "obstacles": [{"r": p.r, "c": p.c} for p in sorted(self.obstacles, key=lambda x: (x.r, x.c))],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GridModel":
        rows = int(d.get("rows", 15))
        cols = int(d.get("cols", 15))
        gm = GridModel(rows, cols)
        s = d.get("start", {"r": 0, "c": 0})
        g = d.get("goal", {"r": rows-1, "c": cols-1})
        gm.start = Pos(int(s["r"]), int(s["c"]))
        gm.goal = Pos(int(g["r"]), int(g["c"]))
        obs = set()
        for o in d.get("obstacles", []):
            obs.add(Pos(int(o["r"]), int(o["c"])))
        gm.obstacles = obs
        gm.obstacles.discard(gm.start)
        gm.obstacles.discard(gm.goal)
        return gm
