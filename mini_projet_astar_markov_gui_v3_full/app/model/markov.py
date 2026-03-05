from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from .grid import GridModel, Pos
from .policy import Action

@dataclass(slots=True)
class MarkovModel:
    states: List[Pos]
    idx: Dict[Pos, int]
    goal_index: int
    fail_index: Optional[int]
    trans: List[List[Tuple[int, float]]]
    P: Optional[np.ndarray] = None  # dense transition matrix if built

def _intended_and_laterals(action: Action) -> Tuple[Tuple[int,int], Tuple[Tuple[int,int], Tuple[int,int]]]:
    if action == 'U': return (-1,0), ((0,-1),(0,1))
    if action == 'D': return (1,0), ((0,-1),(0,1))
    if action == 'L': return (0,-1), ((-1,0),(1,0))
    if action == 'R': return (0,1), ((-1,0),(1,0))
    return (0,0), ((0,0),(0,0))

def dense_transition_matrix(mm: MarkovModel) -> np.ndarray:
    n = len(mm.states)
    P = np.zeros((n, n), dtype=float)
    for i, row in enumerate(mm.trans):
        for j, pr in row:
            P[i, j] += pr
    rs = P.sum(axis=1)
    for i in range(n):
        if rs[i] > 0:
            P[i, :] /= rs[i]
        else:
            P[i, i] = 1.0
    return P

def build_markov_from_policy(
    grid: GridModel,
    policy: Dict[Pos, Action],
    epsilon: float,
    add_fail: bool = False,
    collision_to_fail: bool = False,
    build_dense_if_n_leq: int = 2500,
) -> MarkovModel:
    free = grid.all_free_cells()
    if grid.goal not in free:
        free.append(grid.goal)

    states = list(free)
    idx = {p:i for i,p in enumerate(states)}

    fail_index: Optional[int] = None
    if add_fail:
        fail_pos = Pos(-1, -1)
        fail_index = len(states)
        states.append(fail_pos)
        idx[fail_pos] = fail_index

    goal_index = idx[grid.goal]
    trans: List[List[Tuple[int,float]]] = [[] for _ in range(len(states))]

    for p in states:
        i = idx[p]
        if fail_index is not None and i == fail_index:
            trans[i] = [(i, 1.0)]
            continue
        if i == goal_index:
            trans[i] = [(i, 1.0)]
            continue

        action = policy.get(p, 'S')
        intended, (lat1, lat2) = _intended_and_laterals(action)
        moves = [(intended, 1.0 - epsilon), (lat1, epsilon/2.0), (lat2, epsilon/2.0)]

        probs: Dict[int, float] = {}
        for (dr, dc), pr in moves:
            dest = Pos(p.r + dr, p.c + dc)
            if (dr, dc) == (0,0):
                j = i
            elif grid.is_free(dest):
                j = idx[dest]
            else:
                j = fail_index if (add_fail and collision_to_fail) else i  # type: ignore
            probs[j] = probs.get(j, 0.0) + pr

        s = sum(probs.values())
        trans[i] = [(j, float(pv/s)) for j, pv in probs.items()] if s > 0 else [(i, 1.0)]

    mm = MarkovModel(states=states, idx=idx, goal_index=goal_index, fail_index=fail_index, trans=trans, P=None)
    if len(states) <= build_dense_if_n_leq:
        mm.P = dense_transition_matrix(mm)
    return mm

def evolve_pi_sparse(mm: MarkovModel, start: Pos, steps: int) -> np.ndarray:
    n = len(mm.states)
    pi = np.zeros(n, dtype=float)
    pi[mm.idx[start]] = 1.0
    for _ in range(steps):
        new = np.zeros(n, dtype=float)
        for i, mass in enumerate(pi):
            if mass == 0:
                continue
            for j, pr in mm.trans[i]:
                new[j] += mass * pr
        pi = new
    return pi

def evolve_pi_dense(mm: MarkovModel, start: Pos, steps: int) -> np.ndarray:
    if mm.P is None:
        raise ValueError("Dense matrix P not available for this model size.")
    n = mm.P.shape[0]
    pi0 = np.zeros(n, dtype=float)
    pi0[mm.idx[start]] = 1.0
    Pn = np.linalg.matrix_power(mm.P, steps)
    return pi0 @ Pn

def prob_absorbing_over_time(mm: MarkovModel, start: Pos, horizon: int) -> Dict[str, np.ndarray]:
    goal = np.zeros(horizon+1, dtype=float)
    fail = np.zeros(horizon+1, dtype=float) if mm.fail_index is not None else None

    pi = np.zeros(len(mm.states), dtype=float)
    pi[mm.idx[start]] = 1.0
    goal[0] = pi[mm.goal_index]
    if fail is not None:
        fail[0] = pi[mm.fail_index]  # type: ignore

    for t in range(1, horizon+1):
        new = np.zeros_like(pi)
        for i, mass in enumerate(pi):
            if mass == 0:
                continue
            for j, pr in mm.trans[i]:
                new[j] += mass * pr
        pi = new
        goal[t] = pi[mm.goal_index]
        if fail is not None:
            fail[t] = pi[mm.fail_index]  # type: ignore

    out = {"goal": goal}
    if fail is not None:
        out["fail"] = fail
    return out
