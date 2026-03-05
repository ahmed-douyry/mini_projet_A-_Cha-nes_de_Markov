from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import random

from grid import Pos
from markov import MarkovModel

@dataclass
class MCStats:
    success_rate: float
    fail_rate: float
    mean_steps_success: float
    mean_steps_all: float

def _sample_next(P_row: np.ndarray, rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for j, pr in enumerate(P_row):
        acc += float(pr)
        if r <= acc:
            return j
    return int(np.argmax(P_row))

def simulate_episode(mm: MarkovModel, start: Pos, max_steps: int, seed: int) -> Tuple[int, str]:
    rng = random.Random(seed)
    cur = mm.idx[start]
    steps = 0

    while steps < max_steps:
        if cur == mm.goal_index:
            return steps, "GOAL"
        if mm.fail_index is not None and cur == mm.fail_index:
            return steps, "FAIL"

        cur = _sample_next(mm.P[cur], rng)
        steps += 1

    return steps, "TIMEOUT"

def monte_carlo(mm: MarkovModel, start: Pos, episodes: int = 2000, max_steps: int = 500, seed: int = 0) -> MCStats:
    rng = random.Random(seed)
    succ = 0
    fail = 0
    steps_all = []
    steps_succ = []

    for _ in range(episodes):
        ep_seed = rng.randint(0, 10_000_000)
        steps, outcome = simulate_episode(mm, start, max_steps=max_steps, seed=ep_seed)
        steps_all.append(steps)
        if outcome == "GOAL":
            succ += 1
            steps_succ.append(steps)
        elif outcome == "FAIL":
            fail += 1

    steps_all = np.array(steps_all, dtype=float)
    steps_succ = np.array(steps_succ, dtype=float) if steps_succ else np.array([], dtype=float)

    return MCStats(
        success_rate=succ / episodes,
        fail_rate=fail / episodes,
        mean_steps_success=float(steps_succ.mean()) if steps_succ.size else float("nan"),
        mean_steps_all=float(steps_all.mean()) if steps_all.size else float("nan"),
    )
