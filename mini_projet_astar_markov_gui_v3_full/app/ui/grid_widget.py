from __future__ import annotations
from typing import List, Set, Optional
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget
from ..model.grid import GridModel, Pos

class GridWidget(QWidget):
    def __init__(self, model: GridModel):
        super().__init__()
        self.model = model
        self.cell = 28
        self.mode = "obstacle"
        self.path: List[Pos] = []
        self.explored: Set[Pos] = set()
        self.current: Optional[Pos] = None
        self.heat: Optional[List[List[float]]] = None
        self.setMinimumSize(QSize(520, 520))

    def set_mode(self, mode: str):
        self.mode = mode

    def clear_overlays(self):
        self.path = []
        self.explored = set()
        self.current = None
        self.heat = None
        self.update()

    def set_heatmap(self, heat: Optional[List[List[float]]]):
        self.heat = heat
        self.update()

    def mousePressEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        p = self._pos_from_mouse(event.position().x(), event.position().y())
        if p is None:
            return
        if self.mode == "obstacle":
            self.model.toggle_obstacle(p)
        elif self.mode == "start":
            self.model.set_start(p)
        elif self.mode == "goal":
            self.model.set_goal(p)
        self.update()

    def _pos_from_mouse(self, x, y):
        c = int(x // self.cell)
        r = int(y // self.cell)
        p = Pos(r, c)
        return p if self.model.in_bounds(p) else None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rows, cols = self.model.rows, self.model.cols
        painter.fillRect(self.rect(), QColor("#0b1220"))

        heat_max = 0.0
        if self.heat is not None and self.heat:
            heat_max = max(max(row) for row in self.heat)

        for r in range(rows):
            for c in range(cols):
                p = Pos(r,c)
                x, y = c*self.cell, r*self.cell
                base = QColor("#0f172a")
                if p in self.model.obstacles:
                    base = QColor("#334155")
                elif p in self.explored:
                    base = QColor("#111827")
                painter.fillRect(x,y,self.cell,self.cell,base)

                if self.heat is not None and p not in self.model.obstacles:
                    val = self.heat[r][c]
                    if heat_max>0 and val>0:
                        alpha = int(20 + 220*(val/heat_max))
                        alpha = max(0, min(255, alpha))
                        painter.fillRect(x,y,self.cell,self.cell,QColor(59,130,246,alpha))

                painter.setPen(QPen(QColor("#1e293b"),1))
                painter.drawRect(x,y,self.cell,self.cell)

        if self.path:
            painter.setBrush(QBrush(QColor("#22c55e")))
            painter.setPen(Qt.NoPen)
            for p in self.path:
                x,y = p.c*self.cell, p.r*self.cell
                painter.drawRect(x+4,y+4,self.cell-8,self.cell-8)

        if self.current is not None:
            painter.setBrush(QBrush(QColor("#eab308")))
            painter.setPen(Qt.NoPen)
            p = self.current
            x,y = p.c*self.cell, p.r*self.cell
            painter.drawRect(x+7,y+7,self.cell-14,self.cell-14)

        self._marker(painter, self.model.start, QColor("#38bdf8"))
        self._marker(painter, self.model.goal, QColor("#fb923c"))

    def _marker(self, painter, p: Pos, color: QColor):
        x,y = p.c*self.cell, p.r*self.cell
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(x+6,y+6,self.cell-12,self.cell-12)
