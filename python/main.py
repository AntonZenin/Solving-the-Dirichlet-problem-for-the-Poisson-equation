"""Точка входа: запуск GUI.

Запуск:
    cd <корень проекта>/python
    python main.py
"""
import os, sys

# Чтобы _poisson.so рядом с main.py был виден.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import run_app

if __name__ == "__main__":
    run_app()
