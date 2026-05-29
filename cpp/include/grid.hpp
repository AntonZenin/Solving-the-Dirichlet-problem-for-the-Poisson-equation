#pragma once
#include <cstddef>
#include <vector>

namespace poisson {

// Прямоугольная сетка [a,b]x[c,d] с (n+1)x(m+1) узлами.
// Шаги: h = (b-a)/n, k = (d-c)/m.
// Узел (i,j) хранится по линейному индексу j*(n+1)+i.
struct Grid {
    double a, b, c, d;
    int    n, m;          // число разбиений
    double h, k;          // шаги
    int    nx, ny;        // n+1, m+1 — число узлов

    Grid(double a_, double b_, double c_, double d_, int n_, int m_)
        : a(a_), b(b_), c(c_), d(d_), n(n_), m(m_),
          h((b_ - a_) / n_), k((d_ - c_) / m_),
          nx(n_ + 1), ny(m_ + 1) {}

    inline int    idx(int i, int j) const { return j * nx + i; }
    inline double x(int i)          const { return a + i * h; }
    inline double y(int j)          const { return c + j * k; }
    inline std::size_t size()       const { return static_cast<std::size_t>(nx) * ny; }
};

} // namespace poisson
