from __future__ import annotations
import numpy as np
import pyqtgraph as pg

def plot_curve(widget: pg.PlotWidget, x: np.ndarray, ys: dict[str, np.ndarray], clear: bool=True):
    if clear:
        widget.clear()
    widget.showGrid(x=True, y=True, alpha=0.2)
    for name, y in ys.items():
        widget.plot(x, y, pen=pg.mkPen(width=2), name=name)

def plot_hist(widget: pg.PlotWidget, samples: np.ndarray, bins: int=25, clear: bool=True):
    if clear:
        widget.clear()
    widget.showGrid(x=True, y=True, alpha=0.2)
    if samples.size == 0:
        return
    y, edges = np.histogram(samples, bins=bins)
    widget.addItem(pg.BarGraphItem(x=edges[:-1], height=y, width=(edges[1]-edges[0])*0.9))
