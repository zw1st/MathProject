from typing import Optional, Callable

import numpy as np

from models.optimization_result import OptimizationResult
from methods.optimizer_abstract import Optimizer
from utils import line_search_wolfe


class BFGS(Optimizer):
    """Квазиньютоновский метод BFGS"""

    def __init__(self,
                 epsilon: float = 0.001,
                 max_iterations: int = 1000,
                 verbose: bool = True):
        super().__init__(epsilon, max_iterations, verbose)

    def optimize(self,
                 func: Callable,
                 grad: Callable,
                 x0: np.ndarray,
                 hess: Optional[Callable] = None) -> OptimizationResult:

        self.reset_counters()
        n = len(x0)

        x = x0.copy()
        H = np.eye(n)  # Приближение обратного Гессиана

        history = {
            'k': [], 'x': [], 'f': [], 'grad_norm': [], 'alpha': [], 'd_norm': []  # ✅ Добавлен d_norm
        }

        g = self._count_grad(grad, x)
        converged = False
        final_k = 0

        if self.verbose:
            self._print_header()  # ✅ Печать заголовка

        for k in range(self.max_iterations):
            f_val = self._count_f(func, x)
            grad_norm = np.linalg.norm(g)
            d = -H @ g
            d_norm = np.linalg.norm(d)

            # ✅ Проверка на расходимость
            if np.isinf(f_val) or grad_norm > 1e15:
                if self.verbose:
                    print(f"⚠️ Расходимость на итерации {k}")
                break

            # 1. Сохраняем историю
            history['k'].append(k)
            history['x'].append(x.copy())
            history['f'].append(f_val)
            history['grad_norm'].append(grad_norm)
            history['d_norm'].append(d_norm)  # ✅

            # 2. Проверяем сходимость
            if grad_norm < self.epsilon:
                converged = True
                history['alpha'].append(0.0)
                final_k = k
                if self.verbose:
                    self._print_iteration(k, x, f_val, grad_norm, 0.0)
                    print(f"✅ Сходимость достигнута на итерации {k}")
                break

            # ✅ Ограничение длины направления
            if d_norm > 1e10:
                d = d / d_norm * 1e10
                d_norm = 1e10

            # 3. Линейный поиск
            alpha = line_search_wolfe(func, grad, x, d, g)
            if np.isnan(alpha) or np.isinf(alpha) or alpha < 1e-16:
                alpha = 1e-10
            history['alpha'].append(alpha)

            if self.verbose:
                self._print_iteration(k, x, f_val, grad_norm, alpha)  # ✅ Передаём x

            # 4. Обновление точки
            s = alpha * d
            x_new = x + s

            # 5. Новый градиент
            g_new = self._count_grad(grad, x_new)
            y = g_new - g

            # 6. Обновление H по формуле BFGS
            sy = s @ y
            # ✅ Защита: обновляем только если кривизна положительна
            if sy > 1e-10:
                rho = 1.0 / sy
                I = np.eye(n)
                V = I - rho * np.outer(s, y)
                H = V @ H @ V.T + rho * np.outer(s, s)
            elif self.verbose:
                print(f"⚠️ Пропущено обновление BFGS: s·y = {sy:.2e} <= 0")

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
                history=history, converged=converged, method_name="BFGS"
            ))

        return OptimizationResult(
            x=x,
            f_val=float(f_val),           # ✅ Конвертация типов
            grad_norm=float(grad_norm),
            n_iterations=int(final_k + 1),
            n_function_evals=int(self.n_function_evals),
            n_grad_evals=int(self.n_grad_evals),
            n_hess_evals=0,
            history=history,
            converged=converged,
            method_name="BFGS"
        )