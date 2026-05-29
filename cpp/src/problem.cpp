#include "problem.hpp"
#include <cmath>

namespace poisson {

namespace {
constexpr double PI = 3.14159265358979323846;
}

// ---------- Основная задача (вариант 1) ----------
Problem make_main_problem_v1() {
    Problem p;
    p.f = [](double x, double y) {
        double s = std::sin(PI * x * y);
        return s * s;
    };
    p.mu1 = [](double y) { return std::sin(PI * y); };          // u(0,y)
    p.mu2 = [](double y) { return std::sin(PI * y); };          // u(1,y)
    p.mu3 = [](double x) { return x - x * x; };                 // u(x,0)
    p.mu4 = [](double x) { return x - x * x; };                 // u(x,1)
    p.u_exact = nullptr; // основная задача — точное решение неизвестно
    return p;
}

// ---------- Тестовая задача (вариант 1) ----------
// u*(x,y) = exp(sin^2(pi x y))
// f*(x,y) = -Delta u*(x,y)
//        = -pi^2 (x^2+y^2) exp(sin^2(pi x y)) * [sin^2(2 pi x y) + 2 cos(2 pi x y)]
Problem make_test_problem_v1() {
    auto u_star = [](double x, double y) {
        double s = std::sin(PI * x * y);
        return std::exp(s * s);
    };

    Problem p;
    p.u_exact = u_star;

    p.f = [u_star](double x, double y) {
        double xy   = PI * x * y;
        double s2   = std::sin(2.0 * xy);
        double c2   = std::cos(2.0 * xy);
        double r2   = x * x + y * y;
        return -PI * PI * r2 * u_star(x, y) * (s2 * s2 + 2.0 * c2);
    };

    // Граничные условия — следы u* на сторонах квадрата.
    p.mu1 = [u_star](double y) { return u_star(0.0, y); }; // = 1
    p.mu2 = [u_star](double y) { return u_star(1.0, y); };
    p.mu3 = [u_star](double x) { return u_star(x, 0.0); }; // = 1
    p.mu4 = [u_star](double x) { return u_star(x, 1.0); };
    return p;
}

} // namespace poisson
