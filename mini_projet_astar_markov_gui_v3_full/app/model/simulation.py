from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import random
from .grid import Pos
from .markov import MarkovModel

@dataclass(slots=True)
class SimStats:
    success_rate: float
    fail_rate: float
    mean_steps_success: float
    mean_steps_all: float
    steps_samples: np.ndarray

def _sample_next(trans_row, rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for j, pr in trans_row:
        acc += pr
        if r <= acc:
            return j
    return trans_row[-1][0]

def monte_carlo(mm: MarkovModel, start: Pos, episodes: int=2000, max_steps: int=500, seed: int=0) -> SimStats:
    rng = random.Random(seed)
    steps = []
    succ_steps = []
    success = 0
    fail = 0
    start_i = mm.idx[start]

    for _ in range(episodes):
        cur = start_i
        t = 0
        while t < max_steps:
            if cur == mm.goal_index or (mm.fail_index is not None and cur == mm.fail_index):
                break
            cur = _sample_next(mm.trans[cur], rng)
            t += 1
        steps.append(t)
        if cur == mm.goal_index:
            success += 1
            succ_steps.append(t)
        elif mm.fail_index is not None and cur == mm.fail_index:
            fail += 1

    steps_arr = np.array(steps, dtype=float)
    succ_arr = np.array(succ_steps, dtype=float) if succ_steps else np.array([], dtype=float)
    return SimStats(
        success_rate=success/episodes,
        fail_rate=fail/episodes,
        mean_steps_success=float(succ_arr.mean()) if succ_steps else float("nan"),
        mean_steps_all=float(steps_arr.mean()) if steps_arr.size else float("nan"),
        steps_samples=steps_arr
    )
