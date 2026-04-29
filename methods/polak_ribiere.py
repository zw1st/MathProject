import numpy as np
from typing import Callable, Optional

from methods.optimizer_abstract import Optimizer
from models.optimization_result import OptimizationResult
from utils import line_search_wolfe


class PolakRibiere(Optimizer):
    """Метод сопряжённых градиентов Полака-Рибьера (устойчивая версия)"""

    def __init__(self,
                 epsilon: float = 0.001,
                 max_iterations: int = 1000,
                 restart_frequency: Optional[int] = None,
                 positive_only: bool = True,
                 verbose: bool = True):
        super().__init__(epsilon, max_iterations, verbose)
        self.restart_frequency = restart_frequency
        self.positive_only = positive_only

    def optimize(self,
                 func: Callable,
                 grad: Callable,
                 x0: np.ndarray,
                 hess: Optional[Callable] = None) -> OptimizationResult:

        self.reset_counters()
        n = len(x0)
        if self.restart_frequency is None:
            self.restart_frequency = n

        x = x0.copy().astype(np.float64)
        history = {'k': [], 'x': [], 'f': [], 'grad_norm': [], 'alpha': [], 'd_norm': [], 'beta': []}

        g = self._count_grad(grad, x)
        d = -g.copy()

        converged = False
        final_k = 0
        beta = 0.0

        if self.verbose:
            self._print_header(n)

        for k in range(self.max_iterations):
            f_val = self._count_f(func, x)
            grad_norm = np.linalg.norm(g)
            d_norm_log = np.linalg.norm(d)

            if np.isinf(f_val) or grad_norm > 1e15:
                if self.verbose:
                    print(f"⚠️ Расходимость на итерации {k}")
                break

            history['k'].append(k)
            history['x'].append(x.copy())
            history['f'].append(f_val)
            history['grad_norm'].append(grad_norm)
            history['d_norm'].append(d_norm_log)
            history['beta'].append(beta)

            if grad_norm < self.epsilon:
                converged = True
                history['alpha'].append(0.0)
                final_k = k
                if self.verbose:
                    self._print_iteration(k, x, f_val, grad_norm, 0.0)
                    print(f"✅ Сходимость достигнута на итерации {k}")
                break

            # 🔹 1. Гарантия направления спуска
            if np.dot(g, d) >= 0:
                d = -g.copy()

            # 🔹 2. Нормировка направления
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-16:
                d_unit = d / d_norm
            else:
                d_unit = -g / (grad_norm + 1e-16)

            # 🔹 3. Поиск шага (c2=0.1 критичен для PR в оврагах)
            alpha = None
            n_f_ls, n_g_ls = 0, 0
            try:
                alpha, n_f_ls, n_g_ls = line_search_wolfe(
                    func, grad, x, d_unit, g,
                    c2=0.1, max_iter=40,
                    count_f=self._count_f,
                    count_grad=self._count_grad
                )
            except Exception:
                alpha = None

            if alpha is None or np.isnan(alpha) or np.isinf(alpha) or alpha < 1e-16:
                alpha = 1.0
                slope = np.dot(g, d_unit)
                for _ in range(20):
                    if self._count_f(func, x + alpha * d_unit) <= f_val + 1e-4 * alpha * slope:
                        break
                    alpha *= 0.5
                alpha = max(alpha, 1e-16)

            history['alpha'].append(alpha)
            if self.verbose:
                self._print_iteration(k, x, f_val, grad_norm, alpha)

            # 🔹 4. Обновление
            x = x + alpha * d_unit
            g_new = self._count_grad(grad, x)

            # 🔹 5. Коэффициент Полака-Рибьера
            denom = np.dot(g, g)
            if denom < 1e-16:
                beta = 0.0
            else:
                beta = np.dot(g_new, g_new - g) / denom
                if self.positive_only:
                    beta = max(0.0, beta)

            # 🔹 6. Рестарт: Периодический + Условие Пауэлла
            powell_restart = abs(np.dot(g_new, g)) > 0.2 * np.dot(g_new, g_new)
            loss_of_descent = np.dot(g_new, d) >= 0
            periodic = (k + 1) % self.restart_frequency == 0

            if periodic or powell_restart or loss_of_descent:
                d = -g_new.copy()
                if self.verbose and (periodic or powell_restart):
                    reason = "периодический" if periodic else "Пауэлл/потеря спуска"
                    print(f"🔄 Рестарт ({reason}) на итерации {k}")
            else:
                d = -g_new + beta * d

            g = g_new
            final_k = k

        f_val = self._count_f(func, x)
        grad_norm = np.linalg.norm(g)

        if self.verbose:
            self._print_footer(OptimizationResult(
                x=x, f_val=float(f_val), grad_norm=float(grad_norm),
                n_iterations=int(final_k + 1), n_function_evals=int(self.n_function_evals),
                n_grad_evals=int(self.n_grad_evals), n_hess_evals=0,
                history=history, converged=converged, method_name="Полак-Рибьер"
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
            method_name="Полак-Рибьер"
        )