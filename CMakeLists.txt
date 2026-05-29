cmake_minimum_required(VERSION 3.18)
project(poisson_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Дополнительные флаги (МСС/Clang/MSVC).
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ---------- Ядро как статическая библиотека ----------
add_library(poisson_core STATIC
    cpp/src/problem.cpp
    cpp/src/solver.cpp
)
target_include_directories(poisson_core PUBLIC cpp/include)

# ---------- Тест ----------
add_executable(test_convergence cpp/tests/test_convergence.cpp)
target_link_libraries(test_convergence PRIVATE poisson_core)

# ---------- pybind11 модуль ----------
# Сначала пытаемся найти установленный pybind11 (через pip:  pip install pybind11).
# Если не найден — подтягиваем через FetchContent (требует интернет и git).
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "Using local pybind11: ${pybind11_DIR}")
endif()

pybind11_add_module(_poisson cpp/src/bindings.cpp)
target_link_libraries(_poisson PRIVATE poisson_core)

# Кладём собранный модуль в python/ рядом с GUI, чтобы можно было запускать
# `python main.py` прямо из исходников после сборки.
# Для multi-config генераторов (Visual Studio) нужно задать пути для каждой
# конфигурации отдельно, иначе они добавят подпапку Release/Debug.
set(_OUT "${CMAKE_SOURCE_DIR}/python")
set_target_properties(_poisson PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY         "${_OUT}"
    RUNTIME_OUTPUT_DIRECTORY         "${_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_OUT}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${_OUT}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${_OUT}"
)

message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "pybind11 root:  ${pybind11_SOURCE_DIR}")
message(STATUS "Python module → ${CMAKE_SOURCE_DIR}/python")
