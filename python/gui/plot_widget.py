"""Matplotlib-виджеты для встраивания в Qt-окно."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout


class MplCanvas(QWidget):
    """Базовый виджет: фигура matplotlib + панель навигации."""

    def __init__(self, parent=None, figsize=(5.5, 4.5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        self.fig.clear()
        self.canvas.draw_idle()


class HeatmapCanvas(MplCanvas):
    """2D-карта Z(x,y)."""

    def show(self, grid, Z: np.ndarray, title: str, cmap="viridis"):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        x = np.linspace(grid.a, grid.b, grid.nx)
        y = np.linspace(grid.c, grid.d, grid.ny)
        im = ax.pcolormesh(x, y, Z, cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        self.fig.colorbar(im, ax=ax, shrink=0.85)
        self.canvas.draw_idle()


class SurfaceCanvas(MplCanvas):
    """3D-поверхность Z(x,y)."""

    def show(self, grid, Z: np.ndarray, title: str, cmap="viridis"):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        x = np.linspace(grid.a, grid.b, grid.nx)
        y = np.linspace(grid.c, grid.d, grid.ny)
        X, Y = np.meshgrid(x, y)
        # Если сетка большая — продёргиваем для скорости отрисовки.
        stride = max(1, max(grid.nx, grid.ny) // 60)
        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                               rstride=stride, cstride=stride,
                               linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
        self.fig.colorbar(surf, ax=ax, shrink=0.6)
        self.canvas.draw_idle()
