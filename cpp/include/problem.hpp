#pragma once
#include "grid.hpp"
#include <functional>

namespace poisson {

// Общий интерфейс для постановки задачи Дирихле:
//   -Delta u = f в (a,b)x(c,d), u = mu на границе.
// Тестовая задача дополнительно знает точное решение u_star(x,y).
struct Problem {
    // Правая часть в уравнении вида -Delta u = f.
    std::function<double(double, double)> f;
    // Граничные условия (мю): mu1 = u(a,y), mu2 = u(b,y), mu3 = u(x,c), mu4 = u(x,d).
    std::function<double(double)> mu1, mu2, mu3, mu4;
    // Точное решение (только для тестовой задачи; для основной = nullptr).
    std::function<double(double, double)> u_exact;

    bool has_exact() const { return static_cast<bool>(u_exact); }
};

// Вариант 1: основная задача
//   -Delta u = sin^2(pi x y), [0,1]x[0,1]
//   u(0,y) = u(1,y) = sin(pi y)
//   u(x,0) = u(x,1) = x - x^2
Problem make_main_problem_v1();

// Вариант 1: тестовая задача
//   u*(x,y) = exp(sin^2(pi x y))
//   f* подбираем так, чтобы -Delta u* = f*
//   mu* — следы u* на сторонах квадрата
Problem make_test_problem_v1();

// Параметры области для варианта 1.
struct Domain { double a, b, c, d; };
constexpr Domain DOMAIN_V1 = {0.0, 1.0, 0.0, 1.0};

} // namespace poisson
