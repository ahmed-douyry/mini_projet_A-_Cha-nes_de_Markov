from __future__ import annotations
import sys
import os
import json
from typing import Optional, Dict
import numpy as np

# Add parent directory to path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPalette, QColor, QKeySequence, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSpinBox, QTabWidget, QGroupBox,
    QFormLayout, QMessageBox, QCheckBox, QComboBox, QFileDialog, QStyle, QToolBar
)

import pyqtgraph as pg

from app.model.grid import GridModel, Pos
from app.model.astar import run_astar, run_greedy, run_ucs, SearchResult
from app.model.policy import build_path_policy
from app.model.markov import (
    build_markov_from_policy, evolve_pi_sparse, evolve_pi_dense,
    prob_absorbing_over_time, MarkovModel
)
from app.model.simulation import monte_carlo
from app.ui.grid_widget import GridWidget
from app.ui.plots import plot_curve, plot_hist

def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#0b1220"))
    pal.setColor(QPalette.WindowText, QColor("#e5e7eb"))
    pal.setColor(QPalette.Base, QColor("#0f172a"))
    pal.setColor(QPalette.AlternateBase, QColor("#111827"))
    pal.setColor(QPalette.ToolTipBase, QColor("#111827"))
    pal.setColor(QPalette.ToolTipText, QColor("#e5e7eb"))
    pal.setColor(QPalette.Text, QColor("#e5e7eb"))
    pal.setColor(QPalette.Button, QColor("#111827"))
    pal.setColor(QPalette.ButtonText, QColor("#e5e7eb"))
    pal.setColor(QPalette.BrightText, QColor("#ef4444"))
    pal.setColor(QPalette.Highlight, QColor("#2563eb"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    app.setStyleSheet("""
    QToolTip { color: #e5e7eb; background-color: #111827; border: 1px solid #334155; }
    QGroupBox { border: 1px solid #1f2937; border-radius: 10px; margin-top: 12px; padding: 8px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #e5e7eb; }
    QPushButton { background-color: #111827; border: 1px solid #334155; padding: 8px 10px; border-radius: 10px; }
    QPushButton:hover { border: 1px solid #60a5fa; }
    QPushButton:pressed { background-color: #0f172a; }
    QTabWidget::pane { border: 1px solid #1f2937; border-radius: 10px; }
    QTabBar::tab { background: #0f172a; padding: 8px 12px; border: 1px solid #1f2937; border-bottom: none; border-top-left-radius: 10px; border-top-right-radius: 10px; }
    QTabBar::tab:selected { background: #111827; border-color: #334155; }
    QLabel { color: #e5e7eb; }
    QComboBox, QSpinBox { background: #0f172a; border: 1px solid #334155; padding: 6px; border-radius: 8px; }
    QSlider::groove:horizontal { height: 6px; background: #1f2937; border-radius: 3px; }
    QSlider::handle:horizontal { width: 14px; background: #60a5fa; margin: -6px 0; border-radius: 7px; }
    """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planification robuste sur grille — A* + Markov (GUI) v3")
        self.model = GridModel(rows=15, cols=15)

        self.last_search: Optional[SearchResult] = None
        self.last_policy: Optional[Dict[Pos, str]] = None
        self.last_markov: Optional[MarkovModel] = None
        self.last_curve_goal: Optional[np.ndarray] = None
        self.last_curve_fail: Optional[np.ndarray] = None
        self.last_pi_n: Optional[np.ndarray] = None
        self.last_pi_n_step: Optional[int] = None

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_tick)
        self.anim_index = 0
        self.anim_speed_ms = 25

        self._build_toolbar()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        layout.addLayout(left, 0)
        self.tabs = QTabWidget()
        left.addWidget(self.tabs)

        # Grille
        tab_grid = QWidget()
        self.tabs.addTab(tab_grid, "Grille")
        g_layout = QVBoxLayout(tab_grid)

        gb_mode = QGroupBox("Mode d'édition")
        fm = QFormLayout(gb_mode)
        g_layout.addWidget(gb_mode)

        btn_obs = QPushButton("Obstacles")
        btn_s = QPushButton("Placer Start")
        btn_g = QPushButton("Placer Goal")
        fm.addRow(btn_obs, btn_s)
        fm.addRow(btn_g)
        btn_obs.clicked.connect(lambda: self.grid.set_mode("obstacle"))
        btn_s.clicked.connect(lambda: self.grid.set_mode("start"))
        btn_g.clicked.connect(lambda: self.grid.set_mode("goal"))

        gb_size = QGroupBox("Taille")
        fs = QFormLayout(gb_size)
        g_layout.addWidget(gb_size)
        self.sb_rows = QSpinBox(); self.sb_rows.setRange(5, 60); self.sb_rows.setValue(15)
        self.sb_cols = QSpinBox(); self.sb_cols.setRange(5, 60); self.sb_cols.setValue(15)
        fs.addRow("Lignes", self.sb_rows)
        fs.addRow("Colonnes", self.sb_cols)
        btn_resize = QPushButton("Appliquer")
        fs.addRow(btn_resize)
        btn_resize.clicked.connect(self.on_resize)

        btn_clear = QPushButton("Clear overlays")
        g_layout.addWidget(btn_clear)
        btn_clear.clicked.connect(self.on_clear_overlays)

        # Planification
        tab_plan = QWidget()
        self.tabs.addTab(tab_plan, "Planification")
        a_layout = QVBoxLayout(tab_plan)

        gb_algo = QGroupBox("Algorithme")
        fa = QFormLayout(gb_algo)
        a_layout.addWidget(gb_algo)
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["A*", "UCS", "Greedy"])
        fa.addRow("Choix", self.combo_algo)

        btn_run = QPushButton("Run (instantané)")
        btn_run.clicked.connect(self.on_run_search)
        a_layout.addWidget(btn_run)

        gb_anim = QGroupBox("Animation pas à pas")
        fan = QFormLayout(gb_anim)
        a_layout.addWidget(gb_anim)

        self.btn_anim_start = QPushButton("▶ Start")
        self.btn_anim_pause = QPushButton("⏸ Pause")
        self.btn_anim_reset = QPushButton("⟲ Reset")
        self.btn_anim_start.clicked.connect(self.on_anim_start)
        self.btn_anim_pause.clicked.connect(self.on_anim_pause)
        self.btn_anim_reset.clicked.connect(self.on_anim_reset)

        row = QWidget()
        row_l = QHBoxLayout(row); row_l.setContentsMargins(0,0,0,0)
        row_l.addWidget(self.btn_anim_start)
        row_l.addWidget(self.btn_anim_pause)
        row_l.addWidget(self.btn_anim_reset)
        fan.addRow(row)

        self.sl_speed = QSlider(Qt.Horizontal)
        self.sl_speed.setRange(5, 200)
        self.sl_speed.setValue(self.anim_speed_ms)
        self.sl_speed.valueChanged.connect(self.on_speed)
        fan.addRow("Vitesse (ms)", self.sl_speed)

        self.lbl_astar_stats = QLabel("Stats : —")
        self.lbl_astar_stats.setWordWrap(True)
        a_layout.addWidget(self.lbl_astar_stats)

        btn_make_policy = QPushButton("Construire politique (depuis chemin)")
        btn_make_policy.clicked.connect(self.on_build_policy)
        a_layout.addWidget(btn_make_policy)

        # Markov
        tab_m = QWidget()
        self.tabs.addTab(tab_m, "Markov")
        m_layout = QVBoxLayout(tab_m)

        self.lbl_eps = QLabel("ε = 0.20")
        self.sl_eps = QSlider(Qt.Horizontal)
        self.sl_eps.setRange(0, 60)
        self.sl_eps.setValue(20)
        self.sl_eps.valueChanged.connect(self.on_eps)
        m_layout.addWidget(self.lbl_eps)
        m_layout.addWidget(self.sl_eps)

        self.cb_add_fail = QCheckBox("Ajouter état FAIL (absorbé)")
        self.cb_collision_to_fail = QCheckBox("Collision -> FAIL (sinon rester)")
        m_layout.addWidget(self.cb_add_fail)
        m_layout.addWidget(self.cb_collision_to_fail)

        gb_h = QGroupBox("Horizon / étape")
        fh = QFormLayout(gb_h)
        m_layout.addWidget(gb_h)
        self.sb_horizon = QSpinBox(); self.sb_horizon.setRange(5, 500); self.sb_horizon.setValue(60)
        self.sb_step = QSpinBox(); self.sb_step.setRange(0, 500); self.sb_step.setValue(20)
        fh.addRow("Horizon n", self.sb_horizon)
        fh.addRow("Afficher π(n) à n =", self.sb_step)

        self.cb_dense_pow = QCheckBox("Utiliser π(n)=π(0)P^n (dense) si possible")
        m_layout.addWidget(self.cb_dense_pow)

        btn_build_markov = QPushButton("Construire Markov + π(n) + courbes")
        btn_build_markov.clicked.connect(self.on_build_markov)
        m_layout.addWidget(btn_build_markov)

        # Simulation
        tab_s = QWidget()
        self.tabs.addTab(tab_s, "Simulation")
        s_layout = QVBoxLayout(tab_s)
        gb_sim = QGroupBox("Monte-Carlo")
        fsim = QFormLayout(gb_sim)
        s_layout.addWidget(gb_sim)
        self.sb_episodes = QSpinBox(); self.sb_episodes.setRange(100, 50000); self.sb_episodes.setValue(2000)
        self.sb_maxsteps = QSpinBox(); self.sb_maxsteps.setRange(10, 2000); self.sb_maxsteps.setValue(300)
        self.sb_seed = QSpinBox(); self.sb_seed.setRange(0, 10000); self.sb_seed.setValue(0)
        fsim.addRow("Episodes", self.sb_episodes)
        fsim.addRow("Max steps", self.sb_maxsteps)
        fsim.addRow("Seed", self.sb_seed)
        btn_sim = QPushButton("Run simulation")
        btn_sim.clicked.connect(self.on_run_sim)
        s_layout.addWidget(btn_sim)
        self.lbl_sim_stats = QLabel("Stats : —")
        self.lbl_sim_stats.setWordWrap(True)
        s_layout.addWidget(self.lbl_sim_stats)

        # Right side
        right = QVBoxLayout()
        layout.addLayout(right, 1)
        self.grid = GridWidget(self.model)
        right.addWidget(self.grid, 2)

        plots = QHBoxLayout()
        right.addLayout(plots, 1)
        self.plot_curve = pg.PlotWidget(title="P(X_n = GOAL/FAIL) vs n")
        self.plot_curve.showGrid(x=True, y=True, alpha=0.2)
        plots.addWidget(self.plot_curve, 1)
        self.plot_hist = pg.PlotWidget(title="Histogramme longueurs (MC)")
        self.plot_hist.showGrid(x=True, y=True, alpha=0.2)
        plots.addWidget(self.plot_hist, 1)

        self.resize(1250, 820)

    def _build_toolbar(self):
        tb = QToolBar("Outils", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        style = self.style()
        icon_open = style.standardIcon(QStyle.SP_DialogOpenButton)
        icon_save = style.standardIcon(QStyle.SP_DialogSaveButton)
        icon_export = style.standardIcon(QStyle.SP_DriveFDIcon)
        icon_refresh = style.standardIcon(QStyle.SP_BrowserReload)
        icon_clear = style.standardIcon(QStyle.SP_TrashIcon)

        act_open = QAction(icon_open, "Charger grille (JSON)", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self.on_load_grid)

        act_save = QAction(icon_save, "Sauver grille (JSON)", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self.on_save_grid)

        act_png = QAction(icon_export, "Exporter PNG (grille)", self)
        act_png.setShortcut(QKeySequence("Ctrl+E"))
        act_png.triggered.connect(self.on_export_png)

        act_csv_curve = QAction(icon_export, "Exporter CSV (courbes GOAL/FAIL)", self)
        act_csv_curve.triggered.connect(self.on_export_csv_curve)

        act_csv_P = QAction(icon_export, "Exporter CSV (matrice P si dispo)", self)
        act_csv_P.triggered.connect(self.on_export_csv_P)

        act_csv_pi = QAction(icon_export, "Exporter CSV (vecteur π(n))", self)
        act_csv_pi.triggered.connect(self.on_export_csv_pi)

        act_reset = QAction(icon_refresh, "Reset overlays", self)
        act_reset.triggered.connect(self.on_clear_overlays)

        act_clear_obs = QAction(icon_clear, "Tout effacer obstacles", self)
        act_clear_obs.triggered.connect(self.on_clear_obstacles)

        tb.addAction(act_open); tb.addAction(act_save)
        tb.addSeparator()
        tb.addAction(act_png); tb.addAction(act_csv_curve)
        tb.addAction(act_csv_P); tb.addAction(act_csv_pi)
        tb.addSeparator()
        tb.addAction(act_reset); tb.addAction(act_clear_obs)

    def on_resize(self):
        self.on_anim_pause()
        self.model.resize(self.sb_rows.value(), self.sb_cols.value())
        self.grid.clear_overlays()
        self.grid.update()
        self._reset_session()

    def _reset_session(self):
        self.last_search = None
        self.last_policy = None
        self.last_markov = None
        self.last_curve_goal = None
        self.last_curve_fail = None
        self.last_pi_n = None
        self.last_pi_n_step = None
        self.lbl_astar_stats.setText("Stats : —")
        self.lbl_sim_stats.setText("Stats : —")
        self.plot_curve.clear()
        self.plot_hist.clear()

    def on_clear_overlays(self):
        self.on_anim_pause()
        self.grid.clear_overlays()
        self.plot_hist.clear()

    def on_clear_obstacles(self):
        self.on_anim_pause()
        self.model.obstacles.clear()
        self.grid.update()

    def on_eps(self):
        eps = self.sl_eps.value()/100.0
        self.lbl_eps.setText(f"ε = {eps:.2f}")

    def _run_search_internal(self) -> Optional[SearchResult]:
        if not self.model.is_free(self.model.start) or not self.model.is_free(self.model.goal):
            QMessageBox.warning(self, "Erreur", "Start/Goal doivent être sur des cellules libres.")
            return None
        algo = self.combo_algo.currentText()
        if algo == "A*": return run_astar(self.model)
        if algo == "UCS": return run_ucs(self.model)
        return run_greedy(self.model)

    def on_run_search(self):
        self.on_anim_pause()
        res = self._run_search_internal()
        if res is None: return
        self.last_search = res
        self.grid.path = res.path
        self.grid.explored = res.explored
        self.grid.current = None
        self.grid.set_heatmap(None)
        self.grid.update()
        self.lbl_astar_stats.setText(
            ("✅ Succès" if res.success else "❌ Échec") +
            f" | coût={res.g_cost:.0f} | expansions={res.expansions} | OPENmax={res.open_max} | {res.elapsed_ms:.1f}ms"
        )
        self.last_policy = None
        self.last_markov = None
        self.last_curve_goal = None
        self.last_curve_fail = None
        self.last_pi_n = None
        self.last_pi_n_step = None
        self.plot_curve.clear()
        self.plot_hist.clear()
        self.lbl_sim_stats.setText("Stats : —")

    def on_speed(self):
        self.anim_speed_ms = int(self.sl_speed.value())
        if self.anim_timer.isActive():
            self.anim_timer.setInterval(self.anim_speed_ms)

    def on_anim_start(self):
        self.on_anim_pause()
        res = self._run_search_internal()
        if res is None: return
        self.last_search = res
        self.anim_index = 0
        self.grid.clear_overlays()
        self.grid.set_heatmap(None)
        self.grid.update()
        self.anim_timer.setInterval(self.anim_speed_ms)
        self.anim_timer.start()

    def on_anim_pause(self):
        if self.anim_timer.isActive():
            self.anim_timer.stop()

    def on_anim_reset(self):
        self.on_anim_pause()
        self.anim_index = 0
        if self.last_search:
            self.grid.explored = set()
            self.grid.current = None
            self.grid.path = []
            self.grid.set_heatmap(None)
            self.grid.update()

    def _anim_tick(self):
        if not self.last_search:
            self.anim_timer.stop()
            return
        order = self.last_search.explored_order
        if self.anim_index >= len(order):
            self.grid.current = None
            self.grid.path = self.last_search.path
            self.grid.update()
            self.anim_timer.stop()
            return
        cur = order[self.anim_index]
        self.grid.explored.add(cur)
        self.grid.current = cur
        self.anim_index += 1
        self.grid.update()

    def on_build_policy(self):
        if not self.last_search or not self.last_search.success or not self.last_search.path:
            QMessageBox.warning(self, "Politique", "Exécute une recherche qui trouve un chemin.")
            return
        self.last_policy = build_path_policy(self.last_search.path)
        QMessageBox.information(self, "Politique", "Politique construite depuis le chemin.")

    def on_build_markov(self):
        if self.last_policy is None:
            QMessageBox.warning(self, "Markov", "Construis d'abord la politique.")
            return
        eps = self.sl_eps.value()/100.0
        mm = build_markov_from_policy(
            self.model, self.last_policy, eps,
            add_fail=self.cb_add_fail.isChecked(),
            collision_to_fail=self.cb_collision_to_fail.isChecked(),
            build_dense_if_n_leq=2500
        )
        self.last_markov = mm

        horizon = self.sb_horizon.value()
        curves = prob_absorbing_over_time(mm, self.model.start, horizon)
        self.last_curve_goal = curves["goal"]
        self.last_curve_fail = curves.get("fail", None)
        x = np.arange(0, horizon+1)

        ys = {"GOAL": self.last_curve_goal}
        if self.last_curve_fail is not None:
            ys["FAIL"] = self.last_curve_fail
        plot_curve(self.plot_curve, x, ys)

        n = min(self.sb_step.value(), horizon)
        self.sb_step.setValue(n)
        if self.cb_dense_pow.isChecked() and mm.P is not None:
            pi_n = evolve_pi_dense(mm, self.model.start, n)
        else:
            pi_n = evolve_pi_sparse(mm, self.model.start, n)

        self.last_pi_n = pi_n
        self.last_pi_n_step = n

        heat = [[0.0 for _ in range(self.model.cols)] for _ in range(self.model.rows)]
        for i, st in enumerate(mm.states):
            if st.r < 0 or st.c < 0:
                continue
            heat[st.r][st.c] = float(pi_n[i])
        self.grid.set_heatmap(heat)

        if self.last_search:
            self.grid.path = self.last_search.path
            self.grid.explored = self.last_search.explored
        self.grid.update()

        info = f"|S|={len(mm.states)} | ε={eps:.2f} | P(GOAL,n={n})={self.last_curve_goal[n]:.3f}"
        info += " | P(dense)=ON" if mm.P is not None else " | P(dense)=OFF"
        QMessageBox.information(self, "Markov", info)

    def on_run_sim(self):
        if self.last_markov is None:
            QMessageBox.warning(self, "Simulation", "Construis d'abord Markov.")
            return
        stats = monte_carlo(
            self.last_markov, self.model.start,
            episodes=self.sb_episodes.value(),
            max_steps=self.sb_maxsteps.value(),
            seed=self.sb_seed.value()
        )
        self.lbl_sim_stats.setText(
            f"🎲 success={stats.success_rate:.3f} | fail={stats.fail_rate:.3f} | E[steps|succ]={stats.mean_steps_success:.1f} | E[steps]={stats.mean_steps_all:.1f}"
        )
        plot_hist(self.plot_hist, stats.steps_samples, bins=25)

    def on_save_grid(self):
        path, _ = QFileDialog.getSaveFileName(self, "Sauver la grille", "grid.json", "JSON (*.json)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model.to_dict(), f, indent=2)

    def on_load_grid(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger une grille", "", "JSON (*.json)")
        if not path: return
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.on_anim_pause()
        self.model = GridModel.from_dict(d)
        self.grid.model = self.model
        self.sb_rows.setValue(self.model.rows)
        self.sb_cols.setValue(self.model.cols)
        self.grid.clear_overlays()
        self.grid.update()
        self._reset_session()

    def on_export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exporter PNG (grille)", "grid.png", "PNG (*.png)")
        if not path: return
        self.grid.grab().save(path, "PNG")

    def on_export_csv_curve(self):
        if self.last_curve_goal is None:
            QMessageBox.warning(self, "CSV", "Construis Markov pour obtenir les courbes.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter CSV (courbes)", "absorbing_curves.csv", "CSV (*.csv)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            if self.last_curve_fail is None:
                f.write("n,p_goal\n")
                for n, v in enumerate(self.last_curve_goal.tolist()):
                    f.write(f"{n},{v}\n")
            else:
                f.write("n,p_goal,p_fail\n")
                for n in range(len(self.last_curve_goal)):
                    f.write(f"{n},{self.last_curve_goal[n]},{self.last_curve_fail[n]}\n")

    def on_export_csv_P(self):
        if self.last_markov is None or self.last_markov.P is None:
            QMessageBox.warning(self, "CSV P", "Matrice P non disponible (trop grande) ou Markov non construit.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter CSV (P)", "P_matrix.csv", "CSV (*.csv)")
        if not path: return
        np.savetxt(path, self.last_markov.P, delimiter=",")

    def on_export_csv_pi(self):
        if self.last_pi_n is None or self.last_pi_n_step is None or self.last_markov is None:
            QMessageBox.warning(self, "CSV π", "Calcule d'abord π(n) (Construire Markov).")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter CSV (π(n))", f"pi_n_{self.last_pi_n_step}.csv", "CSV (*.csv)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write("i,r,c,pi\n")
            for i, st in enumerate(self.last_markov.states):
                f.write(f"{i},{st.r},{st.c},{self.last_pi_n[i]}\n")

def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
