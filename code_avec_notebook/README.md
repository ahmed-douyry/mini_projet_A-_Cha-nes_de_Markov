# Mini-projet Bases IA — Structure conforme (A* + Markov + Expériences)

Cette archive suit le squelette attendu :
- `astar.py` : UCS / Greedy / A*
- `policy.py` : politique induite par le chemin
- `markov.py` : construction de la matrice P (stochastique), calculs P^n et π(n)
- `simulation.py` : simulation Monte-Carlo (proba GOAL/FAIL, temps moyen)
- `experiments.py` : expériences recommandées E1–E4 + figures/tableaux (reproductibles)

## Installation
Python 3.10+
```bash
pip install -r requirements.txt
```

## Lancer les expériences (recommandé)
```bash
python experiments.py
```

Les résultats sont enregistrés dans `results/` :
- CSV (mesures)
- PNG (figures)
- JSON (grilles)

## Utilisation rapide (API)
Voir `experiments.py` pour des exemples d’appels :
- `astar.astar(...)`, `astar.ucs(...)`, `astar.greedy(...)`
- `markov.build_transition_matrix(...)`
- `markov.pi_n(...)`
- `simulation.monte_carlo(...)`
