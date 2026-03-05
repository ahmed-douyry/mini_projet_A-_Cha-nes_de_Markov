from __future__ import annotations
import heapq
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from grid import Grid, Pos
from heuristics import manhattan

@dataclass
class SearchResult:
    path: List[Pos]
    cost: float
    expansions: int
    open_max: int
    elapsed_ms: float
    success: bool

def _reconstruct(parent: Dict[Pos, Optional[Pos]], goal: Pos) -> List[Pos]:
    cur: Optional[Pos] = goal
    out: List[Pos] = []
    while cur is not None:
        out.append(cur)
        cur = parent.get(cur)
    out.reverse()
    return out

def _best_first(grid: Grid, f: Callable[[Pos, float], float]) -> SearchResult:
    t0 = time.perf_counter()
    start, goal = grid.start, grid.goal

    g: Dict[Pos, float] = {start: 0.0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}
    closed: Set[Pos] = set()

    heap: List[Tuple[float, int, Pos]] = []
    counter = 0
    heapq.heappush(heap, (f(start, 0.0), counter, start))
    counter += 1

    expansions = 0
    open_max = 1

    while heap:
        open_max = max(open_max, len(heap))
        _, _, u = heapq.heappop(heap)

        if u in closed:
            continue
        closed.add(u)

        if u == goal:
            path = _reconstruct(parent, goal)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return SearchResult(path=path, cost=g[goal], expansions=expansions, open_max=open_max, elapsed_ms=elapsed, success=True)

        expansions += 1
        for v in grid.neighbors4(u):
            cand = g[u] + 1.0
            if cand < g.get(v, float("inf")):
                g[v] = cand
                parent[v] = u
                heapq.heappush(heap, (f(v, cand), counter, v))
                counter += 1

    elapsed = (time.perf_counter() - t0) * 1000.0
    return SearchResult(path=[], cost=float("inf"), expansions=expansions, open_max=open_max, elapsed_ms=elapsed, success=False)

def ucs(grid: Grid) -> SearchResult:
    return _best_first(grid, f=lambda node, gcost: gcost)

def greedy(grid: Grid, h: Callable[[Pos, Pos], float] = manhattan) -> SearchResult:
    return _best_first(grid, f=lambda node, gcost: float(h(node, grid.goal)))

def astar(grid: Grid, h: Callable[[Pos, Pos], float] = manhattan) -> SearchResult:
    return _best_first(grid, f=lambda node, gcost: gcost + float(h(node, grid.goal)))

def weighted_astar(grid: Grid, w: float = 1.5, h: Callable[[Pos, Pos], float] = manhattan) -> SearchResult:
    """Option E4 : Weighted A* (w>=1). Plus w est grand, plus c'est rapide mais moins optimal."""
    return _best_first(grid, f=lambda node, gcost: gcost + w * float(h(node, grid.goal)))
