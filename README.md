# Лабораторная работа №4: Задача Дирихле для уравнения Пуассона

Метод верхней релаксации (МВР) + метод Зейделя, вариант 1.

C++ ядро (вычисления) + Python GUI (PySide6 + matplotlib), связаны через pybind11.

## Установка

### Зависимости
- C++17 компилятор (g++ ≥ 9, MSVC 2019+, clang ≥ 10)
- CMake ≥ 3.18
- Python 3.10+
- Установить Python пакеты:

```bash
pip install -r requirements.txt
```

### Сборка C++ модуля

Если `pybind11` не найдётся через `find_package`, CMake автоматически
скачает его через FetchContent — это требует git и интернет.
Чтобы избежать скачивания, передайте путь к установленному (через pip)
pybind11 в `-Dpybind11_DIR`.

**Linux / macOS (bash):**

```bash
# из корня проекта
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build -j
```

**Windows (PowerShell):**

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --config Release -j
```

Замечания для Windows/PowerShell:
- если `python` не находится, попробуйте `py -3 -m pybind11 --cmakedir`;
- генератор Visual Studio многоконфигурационный — флаг `--config Release` обязателен на этапе сборки.

Собранный модуль (`_poisson.*.so` на Linux, `_poisson.*.pyd` на Windows)
автоматически копируется в `python/`.

### Запуск GUI

```bash
cd python
python main.py
```

### Запуск C++ теста сходимости

```bash
./build/test_convergence
```

## Структура проекта

```
cpp/
  include/            заголовки: grid, problem, solver, norms
  src/                реализация + pybind11-биндинги
  tests/              C++ unit-тест
python/
  _poisson.*.so       собранный модуль (после сборки)
  reports.py          формат справок
  main.py             точка входа GUI
  gui/                окно, графики, таблицы
CMakeLists.txt
```

## Что есть в GUI

1. **Параметры и запуск** — выставить n, m, ω, ε, нач. приближение, метод (Зейдель/МВР), норму невязки. Кнопка «ω → оптимальное» подставляет $\omega_{opt}$ для текущей сетки.
2. **Справка** — текст справки для отчёта (готов к копированию в бланк).
3. **Таблицы** — точное / численное / разность по обеим задачам, прорежены до ~25×25.
4. **Графики** — 3D-поверхности и 2D heatmap для u*, v⁽⁰⁾, v⁽ᴺ⁾, разности.
5. **Сравнение сеток** — три heatmap рядом: v на (n,m), v₂ на (2n,2m), |v - v₂|.
6. **Порядок сходимости** — серия запусков на n=10,20,40,80(,160), отношение погрешностей.

## Постановка (вариант 1)

Область: $[0,1]\times[0,1]$.

**Основная задача:**
$$-\Delta u = \sin^2(\pi x y), \quad u\big|_{x=0,1}=\sin(\pi y), \quad u\big|_{y=0,1}=x-x^2.$$

**Тестовая задача:** $u^*(x,y)=\exp(\sin^2(\pi x y))$, $f^*=-\Delta u^*$ выведено аналитически:
$$f^*(x,y)=-\pi^2(x^2+y^2)\,e^{\sin^2(\pi xy)}\big[\sin^2(2\pi xy)+2\cos(2\pi xy)\big].$$

Граничные значения $\mu^*$ — следы $u^*$ на границе квадрата.
