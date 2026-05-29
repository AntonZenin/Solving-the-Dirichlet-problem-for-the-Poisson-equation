// Простой sanity-check: тестовая задача варианта 1 должна решаться,
// и при удвоении сетки погрешность должна падать примерно в 4 раза (O(h^2)).
#include "grid.hpp"
#include "problem.hpp"
#include "solver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace poisson;

static double max_err(const Grid& g, const Problem& pr,
                      const std::vector<double>& v) {
    double mx = 0.0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            double e = std::fabs(pr.u_exact(g.x(i), g.y(j)) - v[g.idx(i, j)]);
            if (e > mx) mx = e;
        }
    return mx;
}

int main() {
    auto pr = make_test_problem_v1();
    int ns[] = {10, 20, 40, 80};
    double prev = -1.0;
    int bad = 0;
    for (int n : ns) {
        Grid g(0.0, 1.0, 0.0, 1.0, n, n);
        SolverParams sp;
        sp.method        = Method::SOR;
        sp.omega         = optimal_omega(g);
        sp.eps_method    = 1e-12;
        sp.max_iters     = 200000;
        sp.residual_norm = Norm::Max;
        sp.init          = InitialGuess::InterpX;

        auto res = solve(g, pr, sp);
        double err = max_err(g, pr, res.v);
        std::printf("n=%3d  omega=%.6f  iters=%6d  |R|_inf=%.3e  eps1=%.3e",
                    n, sp.omega, res.iters, res.residual_norm_val, err);
        if (prev > 0.0) {
            double ratio = prev / err;
            std::printf("   ratio=%.2f", ratio);
            if (ratio < 3.0) ++bad; // ожидаем ~4
        }
        std::printf("\n");
        prev = err;
    }
    if (bad > 0) {
        std::printf("WARN: convergence order looks off in %d steps\n", bad);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
