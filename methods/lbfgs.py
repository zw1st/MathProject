import numpy as np
from typing import Callable, Optional
from collections import deque

from methods.optimizer_abstract import Optimizer
from models.optimization_result import OptimizationResult
from utils import line_search_wolfe




class LBFGS(Optimizer):
    """
    L-BFGS: квазиньютоновский метод с ограниченной памятью.
    
    Параметры:
    - m: размер истории (по умолчанию 10)
    """

    def __init__(self,
                 epsilon: float = 0.001,
                 max_iterations: int = 1000,
                 m: int = 10,
                 verbose: bool = True):
        super().__init__(epsilon, max_iterations, verbose)
        self.m = m  # Размер истории
        self._history: deque = deque(maxlen=m)  # Хранит (s, y, rho)

    def _two_loop_recursion(self, g: np.ndarray) -> np.ndarray:
        """
        Двухцикловая рекурсия для вычисления H @ g без явного хранения H.
        
        Возвращает направление поиска d = -H @ g
        """
        q = g.copy()
        alpha_list = []

        # Первый цикл (в обратном порядке)
        for s, y, rho in reversed(self._history):
            alpha = rho * (s @ q)
            alpha_list.append(alpha)
            q = q - alpha * y

        # Начальное приближение H_0 = gamma * I
        if len(self._history) > 0:
            s_last, y_last, _ = self._history[-1]
            gamma = (s_last @ y_last) / (y_last @ y_last + 1e-16)
        else:
            gamma = 1.0

        r = gamma * q

        # Второй цикл (в прямом порядке)
        for (s, y, rho), alpha in zip(self._history, reversed(alpha_list)):
            beta = rho * (y @ r)
            r = r + s * (alpha - beta)

        return -r  # Направление спуска

    def optimize(self,
                 func: Callable,
                 grad: Callable,
                 x0: np.ndarray,
                 hess: Optional[Callable] = None) -> OptimizationResult:

        self.reset_counters()
        n = len(x0)
        self._history.clear()  # Сброс истории

        x = x0.copy()
        history = {
            'k': [], 'x': [], 'f': [], 'grad_norm': [], 'alpha': [], 'd_norm': []
        }

        g = self._count_grad(grad, x)
        converged = False
        final_k = 0

        if self.verbose:
            self._print_header()

        for k in range(self.max_iterations):
            f_val = self._count_f(func, x)
            grad_norm = np.linalg.norm(g)

            # Проверка на расходимость
            if np.isinf(f_val) or grad_norm > 1e15:
                if self.verbose:
                    print(f"⚠️ Расходимость на итерации {k}")
                break

            # 1. Сохраняем историю
            history['k'].append(k)
            history['x'].append(x.copy())
            history['f'].append(f_val)
            history['grad_norm'].append(grad_norm)

            # 2. Проверяем сходимость
            if grad_norm < self.epsilon:
                converged = True
                history['alpha'].append(0.0)
                history['d_norm'].append(0.0)
                final_k = k
                if self.verbose:
                    self._print_iteration(k, x, f_val, grad_norm, 0.0)
                    print(f"✅ Сходимость достигнута на итерации {k}")
                break

            # 3. Направление поиска через двухцикловую рекурсию
            d = self._two_loop_recursion(g)
            d_norm = np.linalg.norm(d)

            # Ограничение длины направления
            if d_norm > 1e10:
                d = d / d_norm * 1e10
                d_norm = 1e10

            # 4. Линейный поиск
            alpha = line_search_wolfe(func, grad, x, d, g)
            if np.isnan(alpha) or np.isinf(alpha) or alpha < 1e-16:
                alpha = 1e-10
            history['alpha'].append(alpha)
            history['d_norm'].append(d_norm)

            if self.verbose:
                self._print_iteration(k, x, f_val, grad_norm, alpha)

            # 5. Обновление точки
            s = alpha * d
            x_new = x + s

            # 6. Новый градиент
            g_new = self._count_grad(grad, x_new)
            y = g_new - g

            # 7. Обновление истории (только если кривизна положительна)
            sy = s @ y
            if sy > 1e-10:
                rho = 1.0 / sy
                self._history.append((s.copy(), y.copy(), rho))
            elif self.verbose:
                print(f"⚠️ Пропущено обновление L-BFGS: s·y = {sy:.2e} <= 0")

            x = x_new
            g = g_new
            final_k = k

        # Финальная точка
        f_val = self._count_f(func, x)
        grad_norm = np.linalg.norm(g)

        if self.verbose:
            self._print_footer(OptimizationResult(
                x=x, f_val=float(f_val), grad_norm=float(grad_norm),
                n_iterations=int(final_k + 1), n_function_evals=self.n_function_evals,
                n_grad_evals=self.n_grad_evals, n_hess_evals=0,
                history=history, converged=converged, method_name="L-BFGS"
            ))

        return OptimizationResult(
            x=x,
            f_val=float(f_val),
            grad_norm=float(grad_norm),
            n_iterations=int(final_k + 1),
            n_function_evals=int(self.n_function_evals),
            n_grad_evals=int(self.n_grad_evals),
            n_hess_evals=0,
            history=history,
            converged=converged,
            method_name="L-BFGS"
        )