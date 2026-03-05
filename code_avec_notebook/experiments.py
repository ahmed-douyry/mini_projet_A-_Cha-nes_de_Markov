from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from heuristics import h_zero, manhattan
import astar as ast
from policy import policy_from_path
from markov import build_transition_matrix, absorbing_curves
from simulation import monte_carlo

Pos = Tuple[int, int]

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

def save_grid(grid: Grid, name: str):
    with open(RESULTS / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(grid.to_json(), f, indent=2)

def make_grids() -> Dict[str, Grid]:
    easy_obs = {(1,2),(2,2),(3,2),(3,3),(3,4)}
    medium_obs = {(r,5) for r in range(1,10)} | {(7,c) for c in range(2,11)} | {(3,9),(4,9),(5,9)}
    hard_obs = {(r,7) for r in range(0,14)} | {(10,c) for c in range(0,12)} | {(4,c) for c in range(8,15)}
    easy = Grid(rows=10, cols=10, obstacles=easy_obs, start=(0,0), goal=(9,9))
    medium = Grid(rows=12, cols=12, obstacles=medium_obs, start=(0,0), goal=(11,11))
    hard = Grid(rows=15, cols=15, obstacles=hard_obs, start=(0,0), goal=(14,14))
    return {"easy": easy, "medium": medium, "hard": hard}

def experiment_E1(grids: Dict[str, Grid]):
    rows = []
    for name, g in grids.items():
        r_ucs = ast.ucs(g)
        r_greedy = ast.greedy(g, h=manhattan)
        r_astar = ast.astar(g, h=manhattan)
        for algo, r in [("UCS", r_ucs), ("Greedy", r_greedy), ("A*", r_astar)]:
            rows.append([name, algo, r.success, r.cost, r.expansions, r.open_max, r.elapsed_ms])
        save_grid(g, f"grid_{name}")
    out = RESULTS / "E1_search_comparison.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("grid,algo,success,cost,expansions,open_max,elapsed_ms\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print("[E1] saved", out)

def experiment_E2(grid: Grid, eps_list=(0.0, 0.1, 0.2, 0.3)):
    r_astar = ast.astar(grid, h=manhattan)
    if not r_astar.success:
        print("[E2] A* failed on grid; skip")
        return
    pol = policy_from_path(r_astar.path)
    rows = []
    horizon = 60

    for eps in eps_list:
        mm = build_transition_matrix(grid, pol, epsilon=eps, add_fail=True, collision_to_fail=True)
        curves = absorbing_curves(mm, grid.start, horizon=horizon)
        mc = monte_carlo(mm, grid.start, episodes=3000, max_steps=300, seed=0)

        rows.append([eps, r_astar.cost, curves["goal"][-1], curves["fail"][-1], mc.success_rate, mc.fail_rate, mc.mean_steps_all])

        x = np.arange(0, horizon+1)
        plt.figure()
        plt.plot(x, curves["goal"], label="P(GOAL)")
        plt.plot(x, curves["fail"], label="P(FAIL)")
        plt.xlabel("n")
        plt.ylabel("probabilité")
        plt.title(f"E2 - ε={eps}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / f"E2_curve_eps_{eps}.png", dpi=160)
        plt.close()

        np.savetxt(RESULTS / f"E2_P_eps_{eps}.csv", mm.P, delimiter=",")

    out = RESULTS / "E2_eps_impact.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("eps,cost_astar,p_goal_final,p_fail_final,mc_success,mc_fail,mc_mean_steps\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print("[E2] saved", out)

def experiment_E3(grid: Grid):
    r0 = ast.astar(grid, h=h_zero)
    rM = ast.astar(grid, h=manhattan)
    out = RESULTS / "E3_heuristics.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("heuristic,success,cost,expansions,elapsed_ms\n")
        f.write(f"h0,{r0.success},{r0.cost},{r0.expansions},{r0.elapsed_ms}\n")
        f.write(f"manhattan,{rM.success},{rM.cost},{rM.expansions},{rM.elapsed_ms}\n")
    print("[E3] saved", out)

def experiment_E4_optional(grid: Grid, weights=(1.0, 1.5, 2.0)):
    rows = []
    for w in weights:
        r = ast.weighted_astar(grid, w=w, h=manhattan)
        rows.append([w, r.success, r.cost, r.expansions, r.elapsed_ms])
    out = RESULTS / "E4_weighted_astar.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("w,success,cost,expansions,elapsed_ms\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print("[E4] saved", out)

def main():
    grids = make_grids()
    experiment_E1(grids)
    experiment_E2(grids["medium"], eps_list=(0.0, 0.1, 0.2, 0.3))
    experiment_E3(grids["hard"])
    experiment_E4_optional(grids["medium"], weights=(1.0, 1.5, 2.0))
    print("OK. Consulte le dossier results/")

if __name__ == "__main__":
    main()
