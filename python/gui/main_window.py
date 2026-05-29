"""Главное окно лабораторной работы."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTabWidget, QVBoxLayout,
    QWidget,
)

import _poisson as core
import reports as R

from .plot_widget import HeatmapCanvas, SurfaceCanvas
from .tables import fill_grid_table


# --------------- утилиты для построения SolverParams ---------------

_METHOD_ITEMS = [
    ("Метод верхней релаксации (МВР)", core.Method.SOR),
    ("Метод Зейделя",                  core.Method.Seidel),
]
_NORM_ITEMS = [
    ("max-норма", core.Norm.Max),
    ("Евклидова (L2)", core.Norm.L2),
]
_INIT_ITEMS = [
    ("Интерполяция по x", core.InitialGuess.InterpX),
    ("Интерполяция по y", core.InitialGuess.InterpY),
    ("Нули внутри",       core.InitialGuess.Zero),
]


def _combo(items):
    cb = QComboBox()
    for label, value in items:
        cb.addItem(label, value)
    return cb


class GridParamsBox(QGroupBox):
    """Группа параметров одной сетки + солвера."""

    def __init__(self, title: str, *, default_n=40, default_omega=1.07):
        super().__init__(title)
        self.n_spin = QSpinBox();       self.n_spin.setRange(2, 4000); self.n_spin.setValue(default_n)
        self.m_spin = QSpinBox();       self.m_spin.setRange(2, 4000); self.m_spin.setValue(default_n)

        self.method_combo = _combo(_METHOD_ITEMS)
        self.omega_spin = QDoubleSpinBox(); self.omega_spin.setRange(0.01, 1.99); self.omega_spin.setDecimals(6)
        self.omega_spin.setSingleStep(0.01); self.omega_spin.setValue(default_omega)
        self.opt_omega_btn = QPushButton("ω → оптимальное")

        self.eps_spin = QDoubleSpinBox(); self.eps_spin.setDecimals(15)
        self.eps_spin.setRange(1e-15, 1.0); self.eps_spin.setValue(1e-10)
        self.eps_spin.setSingleStep(1e-11)

        self.nmax_spin = QSpinBox(); self.nmax_spin.setRange(1, 10_000_000); self.nmax_spin.setValue(200_000)

        self.norm_combo = _combo(_NORM_ITEMS)
        self.init_combo = _combo(_INIT_ITEMS)

        form = QFormLayout()
        nm_row = QHBoxLayout()
        nm_row.addWidget(QLabel("n =")); nm_row.addWidget(self.n_spin)
        nm_row.addSpacing(8)
        nm_row.addWidget(QLabel("m =")); nm_row.addWidget(self.m_spin)
        nm_row.addStretch(1)
        form.addRow("Разбиения:", _wrap(nm_row))

        omega_row = QHBoxLayout()
        omega_row.addWidget(self.omega_spin); omega_row.addWidget(self.opt_omega_btn)
        form.addRow("Метод:", self.method_combo)
        form.addRow("ω:", _wrap(omega_row))
        form.addRow("ε_мет:", self.eps_spin)
        form.addRow("N_max:", self.nmax_spin)
        form.addRow("Норма невязки:", self.norm_combo)
        form.addRow("Начальное прибл.:", self.init_combo)
        self.setLayout(form)

        self.opt_omega_btn.clicked.connect(self._set_optimal)
        self.method_combo.currentIndexChanged.connect(self._sync_method)
        self._sync_method()

    def _set_optimal(self):
        g = self.make_grid(0.0, 1.0, 0.0, 1.0)
        self.omega_spin.setValue(core.optimal_omega(g))

    def _sync_method(self):
        is_sor = self.method_combo.currentData() == core.Method.SOR
        self.omega_spin.setEnabled(is_sor)
        self.opt_omega_btn.setEnabled(is_sor)
        if not is_sor:
            self.omega_spin.setValue(1.0)

    def make_grid(self, a: float, b: float, c: float, d: float) -> core.Grid:
        return core.Grid(a, b, c, d, self.n_spin.value(), self.m_spin.value())

    def make_params(self) -> core.SolverParams:
        sp = core.SolverParams()
        sp.method = self.method_combo.currentData()
        sp.omega = self.omega_spin.value() if sp.method == core.Method.SOR else 1.0
        sp.eps_method = self.eps_spin.value()
        sp.max_iters = self.nmax_spin.value()
        sp.residual_norm = self.norm_combo.currentData()
        sp.init = self.init_combo.currentData()
        return sp


def _wrap(layout) -> QWidget:
    w = QWidget(); w.setLayout(layout); return w


# --------------- главное окно ---------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лаб. №1: Задача Дирихле для уравнения Пуассона (вариант 1)")
        self.resize(1400, 900)

        # Состояние: последние результаты.
        self.last_test: R.TestRun | None = None
        self.last_main: R.MainRun | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._build_params_tab(),       "Параметры и запуск")
        tabs.addTab(self._build_report_tab(),       "Справка")
        tabs.addTab(self._build_tables_tab(),       "Таблицы")
        tabs.addTab(self._build_plots_tab(),        "Графики")
        tabs.addTab(self._build_compare_tab(),      "Сравнение сеток")
        tabs.addTab(self._build_convergence_tab(),  "Порядок сходимости")
        self.setCentralWidget(tabs)

    # ---------- Вкладка 1: параметры ----------
    def _build_params_tab(self) -> QWidget:
        self.test_box  = GridParamsBox("Тестовая задача — основная сетка",
                                       default_n=40)
        self.main_box  = GridParamsBox("Основная задача — основная сетка",
                                       default_n=40)
        self.main2_box = GridParamsBox("Основная задача — контрольная сетка (2n, 2m)",
                                       default_n=80)

        # Кнопки запуска
        run_test_btn = QPushButton("▶ Решить тестовую задачу")
        run_main_btn = QPushButton("▶ Решить основную задачу (две сетки)")
        run_test_btn.clicked.connect(self._run_test)
        run_main_btn.clicked.connect(self._run_main)
        for b in (run_test_btn, run_main_btn):
            b.setMinimumHeight(40)
            f = QFont(); f.setBold(True); b.setFont(f)

        self.status_label = QLabel("Готов к запуску.")
        self.status_label.setWordWrap(True)

        col_left = QVBoxLayout()
        col_left.addWidget(self.test_box)
        col_left.addWidget(run_test_btn)
        col_left.addStretch(1)

        col_right = QVBoxLayout()
        col_right.addWidget(self.main_box)
        col_right.addWidget(self.main2_box)
        col_right.addWidget(run_main_btn)
        col_right.addStretch(1)

        top = QHBoxLayout()
        top.addLayout(col_left, 1)
        top.addLayout(col_right, 1)

        outer = QVBoxLayout()
        outer.addLayout(top)
        outer.addWidget(self.status_label)

        return _wrap(outer)

    # ---------- Вкладка 2: справка ----------
    def _build_report_tab(self) -> QWidget:
        self.report_test = QPlainTextEdit(); self.report_test.setReadOnly(True)
        self.report_main = QPlainTextEdit(); self.report_main.setReadOnly(True)
        mono = QFont("Monospace"); mono.setStyleHint(QFont.TypeWriter)
        self.report_test.setFont(mono); self.report_main.setFont(mono)

        splitter = QSplitter(Qt.Horizontal)
        left = _labeled_box("Тестовая задача", self.report_test)
        right = _labeled_box("Основная задача", self.report_main)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([700, 700])
        return splitter

    # ---------- Вкладка 3: таблицы ----------
    def _build_tables_tab(self) -> QWidget:
        # Тестовая: три таблицы.
        self.tab_test_u = QTableWidget()
        self.tab_test_v = QTableWidget()
        self.tab_test_d = QTableWidget()
        # Основная: три таблицы.
        self.tab_main_v  = QTableWidget()
        self.tab_main_v2 = QTableWidget()
        self.tab_main_d  = QTableWidget()

        inner = QTabWidget()
        inner.addTab(_labeled_box("Точное решение u*(x,y)",      self.tab_test_u), "u* (тест)")
        inner.addTab(_labeled_box("Численное решение v(x,y)",    self.tab_test_v), "v (тест)")
        inner.addTab(_labeled_box("Разность u* - v",             self.tab_test_d), "u*-v (тест)")
        inner.addTab(_labeled_box("Численное решение v(x,y)",    self.tab_main_v),  "v (осн.)")
        inner.addTab(_labeled_box("Решение v₂ на (2n,2m)",       self.tab_main_v2), "v₂ (осн.)")
        inner.addTab(_labeled_box("Разность v - v₂ в общих узлах", self.tab_main_d), "v-v₂ (осн.)")
        return inner

    # ---------- Вкладка 4: графики ----------
    def _build_plots_tab(self) -> QWidget:
        self.plot_test_u   = SurfaceCanvas(); self.plot_test_v   = SurfaceCanvas()
        self.plot_test_d   = HeatmapCanvas(); self.plot_test_v0  = SurfaceCanvas()
        self.plot_main_v   = SurfaceCanvas(); self.plot_main_v2  = SurfaceCanvas()
        self.plot_main_d   = HeatmapCanvas()

        inner = QTabWidget()
        inner.addTab(self.plot_test_u,   "u*(x,y)")
        inner.addTab(self.plot_test_v0,  "v⁽⁰⁾ нач. приближение")
        inner.addTab(self.plot_test_v,   "v(x,y) тест")
        inner.addTab(self.plot_test_d,   "Разность u*-v")
        inner.addTab(self.plot_main_v,   "v(x,y) осн.")
        inner.addTab(self.plot_main_v2,  "v₂(x,y) на (2n,2m)")
        inner.addTab(self.plot_main_d,   "Разность v-v₂")
        return inner

    # ---------- Вкладка 5: сравнение сеток (рядом) ----------
    def _build_compare_tab(self) -> QWidget:
        self.cmp_left  = HeatmapCanvas()
        self.cmp_right = HeatmapCanvas()
        self.cmp_diff  = HeatmapCanvas()
        h = QHBoxLayout()
        h.addWidget(_labeled_box("Основная сетка (n,m): v",         self.cmp_left), 1)
        h.addWidget(_labeled_box("Контрольная сетка (2n,2m): v₂",    self.cmp_right), 1)
        h.addWidget(_labeled_box("|v - v₂| в общих узлах",           self.cmp_diff), 1)
        return _wrap(h)

    # ---------- Вкладка 6: порядок сходимости ----------
    def _build_convergence_tab(self) -> QWidget:
        # Простая кнопка: запускает серию решений и выводит таблицу как plain text.
        self.conv_text = QPlainTextEdit(); self.conv_text.setReadOnly(True)
        mono = QFont("Monospace"); mono.setStyleHint(QFont.TypeWriter)
        self.conv_text.setFont(mono)

        btn_test = QPushButton("Прогнать порядок сходимости (тестовая задача)")
        btn_main = QPushButton("Прогнать порядок сходимости (основная задача)")
        btn_test.clicked.connect(self._convergence_test)
        btn_main.clicked.connect(self._convergence_main)

        v = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(btn_test); h.addWidget(btn_main); h.addStretch(1)
        v.addLayout(h)
        v.addWidget(self.conv_text)
        return _wrap(v)

    # ------------------------------------------------------------------
    #                       обработчики
    # ------------------------------------------------------------------
    def _run_test(self):
        try:
            g  = self.test_box.make_grid(0.0, 1.0, 0.0, 1.0)
            sp = self.test_box.make_params()
            self.status_label.setText(f"Считаю тестовую задачу на сетке {g.n}×{g.m}…")
            QApplication.processEvents()
            run = R.run_test(g, sp)
            self.last_test = run
            self.report_test.setPlainText(R.format_test_report(run))
            self._update_test_tables(run)
            self._update_test_plots(run, sp)
            stop = "по точности" if run.result.stop_reason == core.StopReason.ToleranceReached else "по N_max"
            self.status_label.setText(
                f"Тестовая задача: N={run.result.iters} ({stop}), "
                f"ε₁={run.eps1:.3e}, |R|={run.result.residual_norm_val:.3e}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"{type(e).__name__}: {e}")
            raise

    def _run_main(self):
        try:
            g  = self.main_box.make_grid (0.0, 1.0, 0.0, 1.0)
            g2 = self.main2_box.make_grid(0.0, 1.0, 0.0, 1.0)
            if g2.n != 2*g.n or g2.m != 2*g.m:
                QMessageBox.warning(self, "Предупреждение",
                    f"Контрольная сетка должна быть (2n, 2m).\n"
                    f"Сейчас: ({g.n},{g.m}) и ({g2.n},{g2.m}). "
                    f"Общие узлы будут проверены, но сравнение может не выполниться.")
            sp  = self.main_box.make_params()
            sp2 = self.main2_box.make_params()
            self.status_label.setText(
                f"Считаю основную задачу: ({g.n}×{g.m}) и ({g2.n}×{g2.m})…")
            QApplication.processEvents()
            run = R.run_main(g, g2, sp, sp2)
            self.last_main = run
            self.report_main.setPlainText(R.format_main_report(run))
            self._update_main_tables(run)
            self._update_main_plots(run, sp, sp2)
            self._update_compare(run)
            self.status_label.setText(
                f"Основная задача: ε₂={run.eps2:.3e}, "
                f"N={run.result.iters}, N₂={run.result2.iters}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"{type(e).__name__}: {e}")
            raise

    # ------------------------------------------------------------------
    def _update_test_tables(self, run: R.TestRun):
        fill_grid_table(self.tab_test_u, run.grid, run.U)
        fill_grid_table(self.tab_test_v, run.grid, run.V)
        fill_grid_table(self.tab_test_d, run.grid, run.U - run.V)

    def _update_main_tables(self, run: R.MainRun):
        fill_grid_table(self.tab_main_v,  run.grid,  run.V)
        fill_grid_table(self.tab_main_v2, run.grid2, run.V2)
        # Разность в общих узлах: на основной сетке.
        diff = run.V - run.V2[::2, ::2]
        fill_grid_table(self.tab_main_d, run.grid, diff)

    def _update_test_plots(self, run: R.TestRun, sp: core.SolverParams):
        # Начальное приближение
        pr = core.make_test_problem_v1()
        # Чтобы показать v0, нужно его пересобрать — у солвера нет API для возврата;
        # дёшево пересчитаем тем же кодом инициализации через mini-solver:
        # запустим solve с max_iters=0 — нет, у нас цикл >=1. Поэтому продублируем логику.
        V0 = _initial_guess_2d(run.grid, pr, sp.init)
        self.plot_test_v0.show(run.grid, V0,                     "Начальное приближение v⁽⁰⁾")
        self.plot_test_u .show(run.grid, run.U,                  "Точное решение u*(x,y)")
        self.plot_test_v .show(run.grid, run.V,                  "Численное решение v⁽ᴺ⁾(x,y)")
        self.plot_test_d .show(run.grid, np.abs(run.U - run.V),  "|u*(x,y) − v(x,y)|", cmap="magma")

    def _update_main_plots(self, run: R.MainRun, sp: core.SolverParams, sp2: core.SolverParams):
        self.plot_main_v .show(run.grid,  run.V,  "v⁽ᴺ⁾(x,y) на (n,m)")
        self.plot_main_v2.show(run.grid2, run.V2, "v₂⁽ᴺ²⁾(x,y) на (2n,2m)")
        diff = np.abs(run.V - run.V2[::2, ::2])
        self.plot_main_d.show(run.grid, diff, "|v − v₂| в общих узлах", cmap="magma")

    def _update_compare(self, run: R.MainRun):
        self.cmp_left .show(run.grid,  run.V,  "v на (n,m)")
        self.cmp_right.show(run.grid2, run.V2, "v₂ на (2n,2m)")
        diff = np.abs(run.V - run.V2[::2, ::2])
        self.cmp_diff .show(run.grid, diff, "|v − v₂|", cmap="magma")

    # ------------------------------------------------------------------
    def _convergence_test(self):
        ns = [10, 20, 40, 80, 160]
        lines = [f"{'n':>4} {'m':>4} {'omega':>10} {'eps_met':>10} "
                 f"{'iters':>7} {'|R|_inf':>12} {'eps1':>12} {'ratio':>7}"]
        prev = None
        for n in ns:
            g = core.Grid(0, 1, 0, 1, n, n)
            sp = core.SolverParams()
            sp.method = core.Method.SOR
            sp.omega = core.optimal_omega(g)
            sp.eps_method = 1e-12
            sp.max_iters = 500_000
            sp.residual_norm = core.Norm.Max
            sp.init = core.InitialGuess.InterpX
            run = R.run_test(g, sp)
            ratio = (prev / run.eps1) if prev is not None else float("nan")
            lines.append(
                f"{n:>4} {n:>4} {sp.omega:>10.6f} {sp.eps_method:>10.0e} "
                f"{run.result.iters:>7} {run.result.residual_norm_val:>12.3e} "
                f"{run.eps1:>12.3e} {ratio:>7.2f}"
            )
            prev = run.eps1
        lines.append("\nПорядок сходимости ≈ log₂(ratio). Для O(h²) ожидаем ratio ≈ 4.")
        self.conv_text.setPlainText("ТЕСТОВАЯ ЗАДАЧА — порядок сходимости\n\n" + "\n".join(lines))

    def _convergence_main(self):
        ns = [10, 20, 40, 80]
        lines = [f"{'n':>4} {'eps_m':>10} {'eps_m2':>10} "
                 f"{'iters':>7} {'iters2':>7} {'eps2':>12} {'ratio':>7}"]
        prev = None
        for n in ns:
            g  = core.Grid(0, 1, 0, 1, n,   n)
            g2 = core.Grid(0, 1, 0, 1, 2*n, 2*n)
            sp = core.SolverParams()
            sp.method = core.Method.SOR; sp.omega = core.optimal_omega(g)
            sp.eps_method = 1e-11; sp.max_iters = 500_000
            sp.residual_norm = core.Norm.Max; sp.init = core.InitialGuess.InterpX
            sp2 = core.SolverParams()
            sp2.method = core.Method.SOR; sp2.omega = core.optimal_omega(g2)
            sp2.eps_method = 1e-11; sp2.max_iters = 1_000_000
            sp2.residual_norm = core.Norm.Max; sp2.init = core.InitialGuess.InterpX
            run = R.run_main(g, g2, sp, sp2)
            ratio = (prev / run.eps2) if prev is not None else float("nan")
            lines.append(
                f"{n:>4} {sp.eps_method:>10.0e} {sp2.eps_method:>10.0e} "
                f"{run.result.iters:>7} {run.result2.iters:>7} "
                f"{run.eps2:>12.3e} {ratio:>7.2f}"
            )
            prev = run.eps2
        lines.append("\nДля O(h²) ожидаем ratio ≈ 4 (точность ε₂ растёт как h²).")
        self.conv_text.setPlainText("ОСНОВНАЯ ЗАДАЧА — порядок сходимости\n\n" + "\n".join(lines))


# Утилиты ----------------------------------------------------------------

def _labeled_box(label: str, widget: QWidget) -> QWidget:
    box = QGroupBox(label)
    lay = QVBoxLayout(box); lay.addWidget(widget)
    return box


def _initial_guess_2d(grid: core.Grid, pr: core.Problem,
                      kind: core.InitialGuess) -> np.ndarray:
    """Возвращает начальное приближение для отображения (граница + интерполяция)."""
    return core.initial_guess_2d(grid, pr, kind)


# Точка входа ------------------------------------------------------------

def run_app():
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
