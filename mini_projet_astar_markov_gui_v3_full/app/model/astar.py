from __future__ import annotations
import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Callable
from .grid import GridModel, Pos

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)

@dataclass(order=True, slots=True)
class PQItem:
    f: float
    g: float
    tiebreak: int
    node: Pos

@dataclass(slots=True)
class SearchResult:
    path: List[Pos]
    explored: Set[Pos]
    explored_order: List[Pos]
    g_cost: float
    expansions: int
    open_max: int
    elapsed_ms: float
    success: bool

def reconstruct(parent: Dict[Pos, Optional[Pos]], goal: Pos) -> List[Pos]:
    cur = goal
    path: List[Pos] = []
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path

def best_first_search(grid: GridModel, start: Pos, goal: Pos, f_key: Callable[[Pos, float], float]) -> SearchResult:
    t0 = time.perf_counter()
    open_heap: List[PQItem] = []
    g: Dict[Pos, float] = {start: 0.0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}
    explored: Set[Pos] = set()
    explored_order: List[Pos] = []
    closed: Set[Pos] = set()

    counter = 0
    heapq.heappush(open_heap, PQItem(f_key(start, 0.0), 0.0, counter, start))
    counter += 1
    open_max = 1
    expansions = 0

    while open_heap:
        open_max = max(open_max, len(open_heap))
        item = heapq.heappop(open_heap)
        cur = item.node
        if cur in closed:
            continue

        explored.add(cur)
        explored_order.append(cur)

        if cur == goal:
            path = reconstruct(parent, goal)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return SearchResult(path, explored, explored_order, g[goal], expansions, open_max, elapsed, True)

        closed.add(cur)
        expansions += 1

        for nb in grid.neighbors4(cur):
            g_new = g[cur] + 1.0
            if g_new < g.get(nb, float("inf")):
                g[nb] = g_new
                parent[nb] = cur
                heapq.heappush(open_heap, PQItem(f_key(nb, g_new), g_new, counter, nb))
                counter += 1

    elapsed = (time.perf_counter() - t0) * 1000.0
    return SearchResult([], explored, explored_order, float("inf"), expansions, open_max, elapsed, False)

def run_astar(grid: GridModel) -> SearchResult:
    s, g_ = grid.start, grid.goal
    return best_first_search(grid, s, g_, f_key=lambda node, gcost: gcost + float(manhattan(node, g_)))

def run_ucs(grid: GridModel) -> SearchResult:
    s, g_ = grid.start, grid.goal
    return best_first_search(grid, s, g_, f_key=lambda node, gcost: gcost)

def run_greedy(grid: GridModel) -> SearchResult:
    s, g_ = grid.start, grid.goal
    return best_first_search(grid, s, g_, f_key=lambda node, gcost: float(manhattan(node, g_)))
