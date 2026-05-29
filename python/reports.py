"""Генерация текстовых справок и оценок для отчёта."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import _poisson as core


@dataclass
class TestRun:
    """Результаты решения тестовой задачи."""
    grid: core.Grid
    params: core.SolverParams
    result: core.SolverResult
    V: np.ndarray        # численное решение, (ny, nx)
    U: np.ndarray        # точное решение, (ny, nx)
    eps1: float          # max|u* - v|
    err_i: int
    err_j: int


@dataclass
class MainRun:
    """Результаты решения основной задачи + контрольной сетки."""
    grid:   core.Grid
    grid2:  core.Grid
    params:  core.SolverParams
    params2: core.SolverParams
    result:  core.SolverResult
    result2: core.SolverResult
    V:  np.ndarray
    V2: np.ndarray
    eps2: float
    err_i: int  # узел основной сетки, где максимальное отклонение
    err_j: int


_NORM_NAME = {core.Norm.L2: "Евклидова (L2)", core.Norm.Max: "max-норма"}
_METHOD_NAME = {core.Method.Seidel: "метод Зейделя",
                core.Method.SOR:    "метод верхней релаксации"}
_INIT_NAME = {core.InitialGuess.InterpX: "интерполяция по x",
              core.InitialGuess.InterpY: "интерполяция по y",
              core.InitialGuess.Zero:    "нули внутри + граница"}
_STOP_NAME = {core.StopReason.ToleranceReached: "по точности",
              core.StopReason.MaxItersReached:  "по числу итераций"}


def _norm_residual_to_error_bound(norm_R: float, grid: core.Grid,
                                  norm_kind: core.Norm) -> float:
    """Грубая оценка ||Z||_2 ≤ ||R||_2 / lambda_min(A).

    Для пятиточечной схемы на прямоугольнике [a,b]x[c,d] с шагами h, k
    минимальное собственное число матрицы (-A) равно
        lambda_min = (4/h^2) sin^2(pi h / 2(b-a)) + (4/k^2) sin^2(pi k / 2(d-c)).
    Возвращаем оценку в L2-норме; для max-нормы она же служит верхней.
    """
    h, k = grid.h, grid.k
    Lx = grid.b - grid.a
    Ly = grid.d - grid.c
    sx = np.sin(np.pi * h / (2.0 * Lx))
    sy = np.sin(np.pi * k / (2.0 * Ly))
    lam_min = (4.0 / (h * h)) * sx * sx + (4.0 / (k * k)) * sy * sy
    # Если невязка дана в max-норме, для оценки в L2 нужно домножить на sqrt(N).
    if norm_kind == core.Norm.Max:
        N = (grid.n - 1) * (grid.m - 1)
        return norm_R * np.sqrt(N) / lam_min
    return norm_R / lam_min


def format_test_report(run: TestRun) -> str:
    g, sp, r = run.grid, run.params, run.result
    x_err = g.x(run.err_i)
    y_err = g.y(run.err_j)
    Z_bound = _norm_residual_to_error_bound(r.residual_norm_val, g, sp.residual_norm)

    # Оценка погрешности схемы O(h^2) — берём const = 1 как опорную оценку.
    # В отчёте уточнить константу по результатам эксперимента порядка сходимости.
    h2 = max(g.h, g.k) ** 2

    return (
        f"СПРАВКА ДЛЯ ТЕСТОВОЙ ЗАДАЧИ (вариант 1)\n"
        f"{'=' * 60}\n"
        f"Сетка: n = {g.n}, m = {g.m}    (h = {g.h:.4g}, k = {g.k:.4g})\n"
        f"Метод: {_METHOD_NAME[sp.method]}, ω = {sp.omega:.6f}\n"
        f"Критерии остановки: ε_мет = {sp.eps_method:.2e}, "
        f"N_max = {sp.max_iters}\n"
        f"Начальное приближение: {_INIT_NAME[sp.init]}\n"
        f"\n"
        f"Затрачено итераций: N = {r.iters}    "
        f"(выход {_STOP_NAME[r.stop_reason]})\n"
        f"Достигнута точность метода ε^(N) = {r.achieved_eps:.3e}\n"
        f"Невязка ||R^(0)|| = {r.initial_residual_norm_val:.3e}    "
        f"(на начальном приближении, {_NORM_NAME[sp.residual_norm]})\n"
        f"Невязка ||R^(N)|| = {r.residual_norm_val:.3e}    "
        f"(на конечном приближении, {_NORM_NAME[sp.residual_norm]})\n"
        f"\n"
        f"Требование: тестовая задача решена с погрешностью ≤ 0.5·10⁻⁵\n"
        f"Фактически: ε₁ = max|u* - v^(N)| = {run.eps1:.3e}\n"
        f"Максимальное отклонение в узле x = {x_err:.4f}, y = {y_err:.4f}\n"
        f"\n"
        f"Оценки погрешности (для отчёта):\n"
        f"  Погрешность решения СЛАУ: ||Z^(N)||∞ ≤ ||Z^(N)||₂ ≤ {Z_bound:.3e}\n"
        f"  Погрешность схемы (порядок): ||z||∞ = O(h²),  "
        f"h² ≈ {h2:.3e}\n"
        f"  Общая погрешность: ||z_общ||∞ ≤ ||Z^(N)||∞ + ||z||∞\n"
    )


def format_main_report(run: MainRun) -> str:
    g, g2 = run.grid, run.grid2
    sp, sp2 = run.params, run.params2
    r, r2 = run.result, run.result2
    x_err = g.x(run.err_i)
    y_err = g.y(run.err_j)

    Z_bound  = _norm_residual_to_error_bound(r.residual_norm_val,  g,  sp.residual_norm)
    Z2_bound = _norm_residual_to_error_bound(r2.residual_norm_val, g2, sp2.residual_norm)

    return (
        f"СПРАВКА ДЛЯ ОСНОВНОЙ ЗАДАЧИ (вариант 1)\n"
        f"{'=' * 60}\n"
        f"ОСНОВНАЯ СЕТКА:\n"
        f"  n = {g.n}, m = {g.m}    (h = {g.h:.4g}, k = {g.k:.4g})\n"
        f"  Метод: {_METHOD_NAME[sp.method]}, ω = {sp.omega:.6f}\n"
        f"  ε_мет = {sp.eps_method:.2e}, N_max = {sp.max_iters}\n"
        f"  Начальное приближение: {_INIT_NAME[sp.init]}\n"
        f"  Итераций: N = {r.iters}    (выход {_STOP_NAME[r.stop_reason]})\n"
        f"  ε^(N) = {r.achieved_eps:.3e}\n"
        f"  ||R^(0)|| = {r.initial_residual_norm_val:.3e}    "
        f"(нач. приближение, {_NORM_NAME[sp.residual_norm]})\n"
        f"  ||R^(N)|| = {r.residual_norm_val:.3e}    "
        f"(конечное, {_NORM_NAME[sp.residual_norm]})\n"
        f"  ||Z^(N)||₂ ≤ {Z_bound:.3e}\n"
        f"\n"
        f"КОНТРОЛЬНАЯ СЕТКА (половинный шаг):\n"
        f"  2n = {g2.n}, 2m = {g2.m}    (h = {g2.h:.4g}, k = {g2.k:.4g})\n"
        f"  Метод: {_METHOD_NAME[sp2.method]}, ω₂ = {sp2.omega:.6f}\n"
        f"  ε_мет-2 = {sp2.eps_method:.2e}, N_max-2 = {sp2.max_iters}\n"
        f"  Начальное приближение: {_INIT_NAME[sp2.init]}\n"
        f"  Итераций: N₂ = {r2.iters}    (выход {_STOP_NAME[r2.stop_reason]})\n"
        f"  ε^(N₂) = {r2.achieved_eps:.3e}\n"
        f"  ||R^(0)|| = {r2.initial_residual_norm_val:.3e}    "
        f"(нач. приближение, {_NORM_NAME[sp2.residual_norm]})\n"
        f"  ||R^(N₂)|| = {r2.residual_norm_val:.3e}    "
        f"(конечное, {_NORM_NAME[sp2.residual_norm]})\n"
        f"  ||Z^(N₂)||₂ ≤ {Z2_bound:.3e}\n"
        f"\n"
        f"Требование: основная задача решена с точностью ≤ 0.5·10⁻⁵\n"
        f"Фактически: ε₂ = max|v^(N) - v₂^(N₂)| = {run.eps2:.3e}\n"
        f"Максимальное отклонение в узле x = {x_err:.4f}, y = {y_err:.4f}\n"
    )


def run_test(grid: core.Grid, params: core.SolverParams) -> TestRun:
    pr = core.make_test_problem_v1()
    r = core.solve(grid, pr, params)
    V = r.v_as_2d(grid)
    U = core.exact_on_grid(grid, pr)
    eps1, i, j = core.max_abs_diff(U, V)
    return TestRun(grid=grid, params=params, result=r, V=V, U=U,
                   eps1=eps1, err_i=i, err_j=j)


def run_main(grid: core.Grid, grid2: core.Grid,
             params: core.SolverParams, params2: core.SolverParams) -> MainRun:
    pr = core.make_main_problem_v1()
    r  = core.solve(grid,  pr, params)
    r2 = core.solve(grid2, pr, params2)
    V  = r.v_as_2d(grid)
    V2 = r2.v_as_2d(grid2)
    eps2, i, j = core.max_abs_diff_common_nodes(V, V2)
    return MainRun(grid=grid, grid2=grid2,
                   params=params, params2=params2,
                   result=r, result2=r2,
                   V=V, V2=V2, eps2=eps2, err_i=i, err_j=j)
