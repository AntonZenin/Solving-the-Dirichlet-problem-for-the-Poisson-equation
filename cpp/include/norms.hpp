#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

namespace poisson {

enum class Norm { L2, Max };

inline double norm_l2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

inline double norm_max(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) {
        double a = std::fabs(x);
        if (a > m) m = a;
    }
    return m;
}

inline double norm_of(const std::vector<double>& v, Norm kind) {
    return kind == Norm::L2 ? norm_l2(v) : norm_max(v);
}

} // namespace poisson
