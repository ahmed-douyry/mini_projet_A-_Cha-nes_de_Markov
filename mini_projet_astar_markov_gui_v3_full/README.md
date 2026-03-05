# Mini-projet : Planification robuste sur grille (A* + Markov) — GUI moderne (v3)

## Installation
Python 3.10+
```bash
pip install -r requirements.txt
```

## Lancer
```bash
python -m app.main
```

## Ajouts v3 (pour respecter le mini-projet)
- Construction explicite de la **matrice de transition P** (dense) quand |S| ≤ 2500
- Export CSV de :
  - courbes P(X_n = GOAL) et P(X_n = FAIL) (si FAIL activé)
  - matrice P (si dispo)
  - vecteur π(n)
- Option GUI : calcul π(n)=π(0)P^n (dense) si P est disponible (sinon itération sparse)
