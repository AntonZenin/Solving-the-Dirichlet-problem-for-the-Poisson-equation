"""Таблица значений сеточной функции с прореживанием узлов."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


def fill_grid_table(table: QTableWidget, grid, Z: np.ndarray,
                    max_show: int = 25, fmt: str = "{:+.6e}") -> None:
    """Заполняет таблицу значениями Z[ny, nx].

    Если узлов больше max_show по любой оси — берём равномерное подмножество.
    Узлы 0 и n (граничные крайние) всегда включены.
    """
    ny, nx = Z.shape

    def pick(n_total: int) -> list[int]:
        if n_total <= max_show:
            return list(range(n_total))
        # Равномерные узлы, гарантированно включая 0 и n_total-1.
        idx = np.linspace(0, n_total - 1, max_show).round().astype(int)
        return list(dict.fromkeys(idx.tolist()))

    cols = pick(nx)
    rows = pick(ny)

    table.clear()
    table.setRowCount(len(rows))
    table.setColumnCount(len(cols))
    table.setHorizontalHeaderLabels([f"x[{i}]={grid.x(i):.3f}" for i in cols])
    table.setVerticalHeaderLabels(  [f"y[{j}]={grid.y(j):.3f}" for j in rows])

    for ri, j in enumerate(rows):
        for ci, i in enumerate(cols):
            item = QTableWidgetItem(fmt.format(Z[j, i]))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(ri, ci, item)

    table.resizeColumnsToContents()
