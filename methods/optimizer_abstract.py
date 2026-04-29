import numpy as np
from typing import Callable, Optional
from abc import ABC, abstractmethod

from models.optimization_result import OptimizationResult


class Optimizer(ABC):
    """Базовый класс для методов оптимизации"""

    def __init__(self,
                 epsilon: float = 0.001,
                 max_iterations: int = 1000,
                 verbose: bool = True):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.n_function_evals = 0
        self.n_grad_evals = 0
        self.n_hess_evals = 0

    def reset_counters(self):
        """Сбросить счётчики вызовов"""
        self.n_function_evals = 0
        self.n_grad_evals = 0
        self.n_hess_evals = 0

    @abstractmethod
    def optimize(self,
                 func: Callable,
                 grad: Callable,
                 x0: np.ndarray,
                 hess: Optional[Callable] = None) -> OptimizationResult:
        """Запустить оптимизацию"""
        pass

    def _count_f(self, func: Callable, x: np.ndarray) -> float:
        """Вычислить функцию с подсчётом вызовов и проверкой на NaN"""
        self.n_function_evals += 1
        val = func(x)
        
        #  Проверка на валидность
        if np.isnan(val) or np.isinf(val):
            if self.verbose:
                print(f"⚠️ Функция вернула {val} в точке {x}")
            return np.inf
        return val

    def _count_grad(self, grad: Callable, x: np.ndarray) -> np.ndarray:
        """Вычислить градиент с подсчётом вызовов и проверкой на NaN"""
        self.n_grad_evals += 1
        g = grad(x)
        
        #  Проверка на валидность
        if np.any(np.isnan(g)) or np.any(np.isinf(g)):
            if self.verbose:
                print(f"⚠️ Градиент содержит NaN/Inf в точке {x}")
            return np.zeros_like(g)
        return g

    def _count_hess(self, hess: Callable, x: np.ndarray) -> np.ndarray:
        """Вычислить Гессиан с подсчётом вызовов и проверкой на NaN"""
        self.n_hess_evals += 1
        H = hess(x)
        
        #  Проверка на валидность
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            if self.verbose:
                print(f"⚠️ Гессиан содержит NaN/Inf в точке {x}")
            return np.eye(len(x))
        return H

    def _print_header(self, dim: int = 2):
        """Вывести заголовок таблицы итераций"""
        if self.verbose:
            header = f"{'k':<5}"
            for i in range(min(dim, 2)):  # Показываем максимум 2 координаты
                header += f" {'x'+str(i+1):<12}"
            header += f"{'f(x)':<18} {'||grad||':<15} {'alpha':<12}"
            print(f"\n{'='*85}\nМетод: {self.__class__.__name__}\n{'='*85}")
            print(header)
            print(f"{'-'*85}")

    def _print_iteration(self, k: int, x: np.ndarray, f_val: float, grad_norm: float | np.floating, 
                         alpha: float):
        """Вывести строку итерации"""
        if self.verbose:
            row = f"{k:<5}"
            for i in range(min(len(x), 2)):  # Показываем максимум 2 координаты
                row += f" {x[i]:<12.6f}"
            row += f" {f_val:<18.10f} {grad_norm:<15.10f} {alpha:<12.10f}"
            print(row)

    def _print_footer(self, result: OptimizationResult):
        """Вывести итоги оптимизации"""
        if self.verbose:
            print(f"{'-'*85}")
            print(f"Сходимость: {'Да' if result.converged else 'Нет'}")
            print(f"   Итераций: {result.n_iterations}")
            print(f"   Вызовов f: {result.n_function_evals}")
            print(f"   Вызовов ∇f: {result.n_grad_evals}")
            if result.n_hess_evals > 0:
                print(f"   Вызовов ∇²f: {result.n_hess_evals}")
            print(f"   f(x*) = {result.f_val:.10f}")
            print(f"   ||grad|| = {result.grad_norm:.10f}")
            print(f"{'='*85}\n")