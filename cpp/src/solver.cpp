#include "solver.hpp"
#include <algorithm>
#include <cmath>

namespace poisson {

namespace {
constexpr double PI = 3.14159265358979323846;
}

// ---------- Оптимальное omega для прямоугольника ----------
// Для пятиточечной схемы на сетке (n, m) на прямоугольнике [a,b]x[c,d]
// спектральный радиус матрицы Якоби равен:
//   rho(B_J) = (h^2 * cos(pi/n) + k^2 * cos(pi/m)) / (h^2 + k^2)
// (это вес-усреднённое среднее двух косинусов; для квадрата h=k даёт
//  rho = (cos(pi/n) + cos(pi/m)) / 2 ).
// Тогда omega_opt = 2 / (1 + sqrt(1 - rho^2)).
double optimal_omega(const Grid& g) {
    const double h2 = g.h * g.h;
    const double k2 = g.k * g.k;

    const double rho = (h2 * std::cos(PI / g.n) + k2 * std::cos(PI / g.m))
                       / (h2 + k2);
    // На случай численных артефактов:
    const double r = std::min(1.0 - 1e-15, std::max(-1.0 + 1e-15, rho));

    return 2.0 / (1.0 + std::sqrt(1.0 - r * r));
}

// ---------- Граница: задаём точно из mu ----------
static void apply_boundary(const Grid& g, const Problem& pr, std::vector<double>& v) {
    for (int j = 0; j < g.ny; ++j) {
        v[g.idx(0,    j)] = pr.mu1(g.y(j));
        v[g.idx(g.n,  j)] = pr.mu2(g.y(j));
    }
    for (int i = 0; i < g.nx; ++i) {
        v[g.idx(i, 0)]    = pr.mu3(g.x(i));
        v[g.idx(i, g.m)]  = pr.mu4(g.x(i));
    }
}

// ---------- Начальное приближение ----------
std::vector<double> build_initial(const Grid& g, const Problem& pr, InitialGuess kind) {
    std::vector<double> v(g.size(), 0.0);
    apply_boundary(g, pr, v);

    if (kind == InitialGuess::Zero) {
        // во внутренних узлах оставляем 0
        return v;
    }

    if (kind == InitialGuess::InterpX) {
        // На каждой горизонтали y_j линейная интерполяция между mu1(y_j) и mu2(y_j).
        for (int j = 1; j < g.m; ++j) {
            const double left  = v[g.idx(0,   j)];
            const double right = v[g.idx(g.n, j)];
            for (int i = 1; i < g.n; ++i) {
                const double t = static_cast<double>(i) / g.n;
                v[g.idx(i, j)] = (1.0 - t) * left + t * right;
            }
        }
    } else { // InterpY
        for (int i = 1; i < g.n; ++i) {
            const double bot = v[g.idx(i, 0)];
            const double top = v[g.idx(i, g.m)];
            for (int j = 1; j < g.m; ++j) {
                const double t = static_cast<double>(j) / g.m;
                v[g.idx(i, j)] = (1.0 - t) * bot + t * top;
            }
        }
    }
    return v;
}

// ---------- Вектор невязки R = -A v + F (на всех узлах; граница = 0) ----------
std::vector<double> residual(const Grid& g, const Problem& pr,
                             const std::vector<double>& v) {
    std::vector<double> R(g.size(), 0.0);
    const double h2 = g.h * g.h;
    const double k2 = g.k * g.k;
    const double diag = 2.0 / h2 + 2.0 / k2;

    for (int j = 1; j < g.m; ++j) {
        for (int i = 1; i < g.n; ++i) {
            const double rhs =
                  (v[g.idx(i-1, j)] + v[g.idx(i+1, j)]) / h2
                + (v[g.idx(i, j-1)] + v[g.idx(i, j+1)]) / k2
                + pr.f(g.x(i), g.y(j));
            R[g.idx(i, j)] = rhs - diag * v[g.idx(i, j)];
        }
    }
    return R;
}

// ---------- Основной цикл МВР/Зейделя ----------
SolverResult solve(const Grid& g, const Problem& pr, const SolverParams& sp) {
    SolverResult res;
    res.v = build_initial(g, pr, sp.init);

    // Невязка на начальном приближении (для отчёта).
    {
        auto R0 = residual(g, pr, res.v);
        res.initial_residual_norm_val = norm_of(R0, sp.residual_norm);
    }

    const double h2   = g.h * g.h;
    const double k2   = g.k * g.k;
    const double diag = 2.0 / h2 + 2.0 / k2;
    const double w    = (sp.method == Method::Seidel) ? 1.0 : sp.omega;
    const double one_minus_w = 1.0 - w;
    const double w_over_diag = w / diag;

    res.stop_reason = StopReason::MaxItersReached;

    for (int N = 1; N <= sp.max_iters; ++N) {
        double delta_max = 0.0; // ||v^{N} - v^{N-1}||_inf

        // Лексикографический обход: при чтении v[i-1,j] и v[i,j-1]
        // уже содержат значения текущей итерации (Гаусс-Зейдель).
        for (int j = 1; j < g.m; ++j) {
            for (int i = 1; i < g.n; ++i) {
                const double old_v = res.v[g.idx(i, j)];
                const double gs_update =
                      (res.v[g.idx(i-1, j)] + res.v[g.idx(i+1, j)]) / h2
                    + (res.v[g.idx(i, j-1)] + res.v[g.idx(i, j+1)]) / k2
                    + pr.f(g.x(i), g.y(j));
                const double new_v = one_minus_w * old_v + w_over_diag * gs_update;
                res.v[g.idx(i, j)] = new_v;

                const double d = std::fabs(new_v - old_v);
                if (d > delta_max) delta_max = d;
            }
        }

        res.iters        = N;
        res.achieved_eps = delta_max;

        if (delta_max < sp.eps_method) {
            res.stop_reason = StopReason::ToleranceReached;
            break;
        }
    }

    // Финальная норма невязки
    auto R = residual(g, pr, res.v);
    res.residual_norm_val = norm_of(R, sp.residual_norm);

    return res;
}

} // namespace poisson