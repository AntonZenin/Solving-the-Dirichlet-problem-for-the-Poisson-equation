#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "grid.hpp"
#include "norms.hpp"
#include "problem.hpp"
#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace poisson;

// Преобразует вектор сетки в numpy-массив (ny, nx) — удобно для matplotlib.
static py::array_t<double> to_numpy_2d(const std::vector<double>& v, const Grid& g) {
    py::array_t<double> arr({g.ny, g.nx});
    auto buf = arr.mutable_unchecked<2>();
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            buf(j, i) = v[g.idx(i, j)];
        }
    }
    return arr;
}

// Берёт первый аргумент шаблона; для py::array_t.
static std::vector<double> from_numpy_2d(const py::array_t<double>& arr, const Grid& g) {
    if (arr.ndim() != 2 || arr.shape(0) != g.ny || arr.shape(1) != g.nx) {
        throw std::runtime_error("array shape does not match grid (expect (ny, nx))");
    }
    std::vector<double> v(g.size());
    auto buf = arr.unchecked<2>();
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            v[g.idx(i, j)] = buf(j, i);
    return v;
}

// Возвращает (ny, nx) массив точного решения, если оно задано.
static py::object exact_grid(const Grid& g, const Problem& pr) {
    if (!pr.has_exact()) return py::none();
    py::array_t<double> arr({g.ny, g.nx});
    auto buf = arr.mutable_unchecked<2>();
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            buf(j, i) = pr.u_exact(g.x(i), g.y(j));
    return arr;
}

PYBIND11_MODULE(_poisson, m) {
    m.doc() = "Poisson Dirichlet solver: SOR/Seidel iterative methods (C++ core)";

    py::enum_<Method>(m, "Method")
        .value("Seidel", Method::Seidel)
        .value("SOR",    Method::SOR);

    py::enum_<Norm>(m, "Norm")
        .value("L2",  Norm::L2)
        .value("Max", Norm::Max);

    py::enum_<InitialGuess>(m, "InitialGuess")
        .value("InterpX", InitialGuess::InterpX)
        .value("InterpY", InitialGuess::InterpY)
        .value("Zero",    InitialGuess::Zero);

    py::enum_<StopReason>(m, "StopReason")
        .value("ToleranceReached", StopReason::ToleranceReached)
        .value("MaxItersReached",  StopReason::MaxItersReached);

    py::class_<Grid>(m, "Grid")
        .def(py::init<double, double, double, double, int, int>(),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
             py::arg("n"), py::arg("m"))
        .def_readonly("a",  &Grid::a)
        .def_readonly("b",  &Grid::b)
        .def_readonly("c",  &Grid::c)
        .def_readonly("d",  &Grid::d)
        .def_readonly("n",  &Grid::n)
        .def_readonly("m",  &Grid::m)
        .def_readonly("h",  &Grid::h)
        .def_readonly("k",  &Grid::k)
        .def_readonly("nx", &Grid::nx)
        .def_readonly("ny", &Grid::ny)
        .def("x", &Grid::x)
        .def("y", &Grid::y);

    py::class_<SolverParams>(m, "SolverParams")
        .def(py::init<>())
        .def_readwrite("method",        &SolverParams::method)
        .def_readwrite("omega",         &SolverParams::omega)
        .def_readwrite("eps_method",    &SolverParams::eps_method)
        .def_readwrite("max_iters",     &SolverParams::max_iters)
        .def_readwrite("residual_norm", &SolverParams::residual_norm)
        .def_readwrite("init",          &SolverParams::init);

    py::class_<Problem>(m, "Problem")
        .def("has_exact", &Problem::has_exact);

    // Возвращаемый Python-объект с результатами решения (включая numpy v).
    py::class_<SolverResult>(m, "SolverResult")
        .def_readonly("iters",             &SolverResult::iters)
        .def_readonly("achieved_eps",      &SolverResult::achieved_eps)
        .def_readonly("initial_residual_norm_val", &SolverResult::initial_residual_norm_val)
        .def_readonly("residual_norm_val", &SolverResult::residual_norm_val)
        .def_readonly("stop_reason",       &SolverResult::stop_reason)
        .def("v_as_2d", [](const SolverResult& r, const Grid& g) {
            return to_numpy_2d(r.v, g);
        });

    // Фабрики задач для варианта 1.
    m.def("make_main_problem_v1", &make_main_problem_v1);
    m.def("make_test_problem_v1", &make_test_problem_v1);

    m.def("optimal_omega", &optimal_omega);

    m.def("solve", &solve,
          py::arg("grid"), py::arg("problem"), py::arg("params"),
          "Run SOR/Seidel iterative solver on the discretized Dirichlet problem.");

    m.def("exact_on_grid", &exact_grid,
          py::arg("grid"), py::arg("problem"),
          "Return u_exact(x,y) as a (ny, nx) numpy array, or None if not test problem.");

    m.def("initial_guess_2d", [](const Grid& g, const Problem& pr, InitialGuess kind) {
        auto v = build_initial(g, pr, kind);
        return to_numpy_2d(v, g);
    }, py::arg("grid"), py::arg("problem"), py::arg("init"),
       "Build initial guess (boundary exact, interior interpolated) and return as (ny, nx).");

    m.def("residual_norm_of", [](const Grid& g, const Problem& pr,
                                 const py::array_t<double>& v_arr, Norm nrm) {
        auto v = from_numpy_2d(v_arr, g);
        auto R = residual(g, pr, v);
        return norm_of(R, nrm);
    });

    m.def("max_abs_diff", [](const py::array_t<double>& a,
                             const py::array_t<double>& b) {
        if (a.ndim() != 2 || b.ndim() != 2 ||
            a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
            throw std::runtime_error("max_abs_diff: shape mismatch");
        }
        auto A = a.unchecked<2>();
        auto B = b.unchecked<2>();
        double mx = 0.0; int mi = 0, mj = 0;
        for (py::ssize_t j = 0; j < a.shape(0); ++j)
            for (py::ssize_t i = 0; i < a.shape(1); ++i) {
                double d = std::fabs(A(j, i) - B(j, i));
                if (d > mx) { mx = d; mi = static_cast<int>(i); mj = static_cast<int>(j); }
            }
        return py::make_tuple(mx, mi, mj);
    }, "Returns (max|a-b|, i*, j*) over equally-shaped 2D arrays.");

    // Сравнение решения на (n,m) и (2n,2m) в общих узлах:
    // общий узел (i,j) на основной соответствует (2i, 2j) на сетке с половинным шагом.
    m.def("max_abs_diff_common_nodes",
          [](const py::array_t<double>& coarse,
             const py::array_t<double>& fine) {
        if (coarse.ndim() != 2 || fine.ndim() != 2)
            throw std::runtime_error("expected 2D arrays");
        const py::ssize_t cy = coarse.shape(0), cx = coarse.shape(1);
        if (fine.shape(0) != 2*(cy-1)+1 || fine.shape(1) != 2*(cx-1)+1)
            throw std::runtime_error("fine grid must have shape (2(ny-1)+1, 2(nx-1)+1)");
        auto C = coarse.unchecked<2>();
        auto F = fine.unchecked<2>();
        double mx = 0.0; int mi = 0, mj = 0;
        for (py::ssize_t j = 0; j < cy; ++j)
            for (py::ssize_t i = 0; i < cx; ++i) {
                double d = std::fabs(C(j, i) - F(2*j, 2*i));
                if (d > mx) { mx = d; mi = static_cast<int>(i); mj = static_cast<int>(j); }
            }
        return py::make_tuple(mx, mi, mj);
    });
}
