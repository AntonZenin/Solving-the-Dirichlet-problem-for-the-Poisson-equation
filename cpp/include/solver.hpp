#pragma once
#include "grid.hpp"
#include "norms.hpp"
#include "problem.hpp"
#include <vector>

namespace poisson {

enum class Method { Seidel, SOR };

enum class InitialGuess {
    InterpX,   // линейная интерполяция граничных значений по x
    InterpY,   // линейная интерполяция граничных значений по y
    Zero       // нулевое (только во внутренних узлах; граница задаётся точно)
};

struct SolverParams {
    Method        method        = Method::SOR;
    double        omega         = 1.0;     // для Seidel игнорируется
    double        eps_method    = 1e-7;    // критерий выхода по точности
    int           max_iters     = 100000;  // критерий выхода по числу итераций
    Norm          residual_norm = Norm::Max;
    InitialGuess  init          = InitialGuess::InterpX;
};

// Причина остановки итераций.
enum class StopReason {
    ToleranceReached,   // итерации сошлись по eps_method
    MaxItersReached     // упёрлись в max_iters
};

struct SolverResult {
    std::vector<double> v;          // сеточная функция, размер nx*ny
    int        iters             = 0;
    double     achieved_eps      = 0.0;  // ||v^{N+1} - v^{N}||_inf на последнем шаге
    double     initial_residual_norm_val = 0.0;  // ||R^{(0)}|| на начальном приближении
    double     residual_norm_val = 0.0;  // ||R^{(N)}|| в выбранной норме
    StopReason stop_reason       = StopReason::MaxItersReached;
};

// Решает разностную схему для задачи Дирихле на сетке g.
SolverResult solve(const Grid& g, const Problem& pr, const SolverParams& sp);

// Оптимальное omega для прямоугольника по теории
// (rho — спектральный радиус матрицы Якоби для пятиточечной схемы):
//   rho = [cos(pi/n)/h^2 + cos(pi/m)/k^2] / (2/h^2 + 2/k^2)
//   omega_opt = 2 / (1 + sqrt(1 - rho^2))
double optimal_omega(const Grid& g);

// Вычисляет вектор невязки R = -A v + F во внутренних узлах (граничные = 0).
std::vector<double> residual(const Grid& g, const Problem& pr,
                             const std::vector<double>& v);

// Заполняет начальное приближение (граничные узлы — точно из mu).
std::vector<double> build_initial(const Grid& g, const Problem& pr, InitialGuess kind);

} // namespace poisson
