from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from grid import Grid, Pos
from policy import Action

@dataclass
class MarkovModel:
    states: List[Pos]
    idx: Dict[Pos, int]
    P: np.ndarray
    goal_index: int
    fail_index: Optional[int]

def _intended_and_laterals(a: Action) -> Tuple[Tuple[int,int], Tuple[Tuple[int,int], Tuple[int,int]]]:
    if a == 'U': return (-1,0), ((0,-1),(0,1))
    if a == 'D': return ( 1,0), ((0,-1),(0,1))
    if a == 'L': return (0,-1), ((-1,0),(1,0))
    if a == 'R': return (0, 1), ((-1,0),(1,0))
    return (0,0), ((0,0),(0,0))

def build_transition_matrix(
    grid: Grid,
    policy: Dict[Pos, Action],
    epsilon: float,
    add_fail: bool = False,
    collision_to_fail: bool = False,
) -> MarkovModel:
    free = grid.all_free_cells()
    if grid.goal not in free:
        free.append(grid.goal)

    states = list(free)
    idx = {s:i for i,s in enumerate(states)}

    fail_index = None
    if add_fail:
        fail_state = (-1, -1)
        fail_index = len(states)
        states.append(fail_state)
        idx[fail_state] = fail_index

    n = len(states)
    P = np.zeros((n, n), dtype=float)

    goal_index = idx[grid.goal]

    for s in states:
        i = idx[s]

        if i == goal_index:
            P[i, i] = 1.0
            continue
        if fail_index is not None and i == fail_index:
            P[i, i] = 1.0
            continue

        a = policy.get(s, 'S')
        intended, (lat1, lat2) = _intended_and_laterals(a)
        moves = [(intended, 1.0 - epsilon), (lat1, epsilon/2.0), (lat2, epsilon/2.0)]

        for (dr, dc), pr in moves:
            if (dr, dc) == (0,0):
                j = i
            else:
                dest = (s[0] + dr, s[1] + dc)
                if grid.is_free(dest):
                    j = idx[dest]
                else:
                    j = fail_index if (add_fail and collision_to_fail) else i
            P[i, j] += pr

        row_sum = P[i].sum()
        if row_sum > 0:
            P[i] /= row_sum
        else:
            P[i, i] = 1.0

    return MarkovModel(states=states, idx=idx, P=P, goal_index=goal_index, fail_index=fail_index)

def P_power(mm: MarkovModel, n: int) -> np.ndarray:
    return np.linalg.matrix_power(mm.P, n)

def pi_n(mm: MarkovModel, start: Pos, n: int) -> np.ndarray:
    pi0 = np.zeros(mm.P.shape[0], dtype=float)
    pi0[mm.idx[start]] = 1.0
    return pi0 @ P_power(mm, n)

def absorbing_curves(mm: MarkovModel, start: Pos, horizon: int) -> Dict[str, np.ndarray]:
    goal = np.zeros(horizon+1, dtype=float)
    fail = np.zeros(horizon+1, dtype=float) if mm.fail_index is not None else None

    for t in range(horizon+1):
        pi = pi_n(mm, start, t)
        goal[t] = pi[mm.goal_index]
        if fail is not None:
            fail[t] = pi[mm.fail_index]  # type: ignore

    out = {"goal": goal}
    if fail is not None:
        out["fail"] = fail
    return out

def absorption_analysis(mm: MarkovModel) -> Dict[str, np.ndarray]:
    """Analyse optionnelle des chaînes absorbantes.

    Forme canonique :
        P = [[Q, R],
             [0, I]]

    - Q : transitions entre états transients
    - R : transitions transients -> absorbants
    - N = (I - Q)^{-1} : matrice fondamentale

    ⚠️ Attention : (I - Q) peut être **singulière** dans certains cas (modèle pathologique).
    Dans ce cas, on utilise une pseudo-inverse et on renvoie `singular=1`.
    """
    absorb = {mm.goal_index}
    if mm.fail_index is not None:
        absorb.add(mm.fail_index)

    transient = [i for i in range(mm.P.shape[0]) if i not in absorb]
    absorbing = sorted(list(absorb))

    Q = mm.P[np.ix_(transient, transient)]
    R = mm.P[np.ix_(transient, absorbing)]

    if Q.size == 0:
        z = np.zeros((0, 0), dtype=float)
        return {
            "Q": Q,
            "R": R,
            "N": z,
            "N_pinv": z,
            "singular": np.array([0], dtype=int),
            "transient_idx": np.array(transient, dtype=int),
            "absorbing_idx": np.array(absorbing, dtype=int),
        }

    I = np.eye(Q.shape[0], dtype=float)
    A = I - Q

    singular_flag = 0
    try:
        N = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        singular_flag = 1
        N = np.linalg.pinv(A)

    return {
        "Q": Q,
        "R": R,
        "N": N,
        "N_pinv": np.linalg.pinv(A),
        "singular": np.array([singular_flag], dtype=int),
        "transient_idx": np.array(transient, dtype=int),
        "absorbing_idx": np.array(absorbing, dtype=int),
    }
