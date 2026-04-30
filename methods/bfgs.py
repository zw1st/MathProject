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

        x = x0.copy().astype(np.float64)
        H = np.eye(n)  # Приближение обратного Гессиана

        history = {
            'k': [], 'x': [], 'f': [], 'grad_norm': [], 'alpha': [], 'd_norm': []
        }

        g = self._count_grad(grad, x)
        converged = False
        final_k = 0

        if self.verbose:
            self._print_header(n)

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
            history['d_norm'].append(d_norm)

            # 2. Проверяем сходимость
            if grad_norm < self.epsilon:
                converged = True
                history['alpha'].append(0.0)
                final_k = k
                if self.verbose:
                    self._print_iteration(k, x, f_val, grad_norm, 0.0)
                    print(f"✅ Сходимость достигнута на итерации {k}")
                break

            # ✅ Проверка направления спуска: d должно быть направлением убывания
            if np.dot(g, d) >= 0:
                if self.verbose:
                    print(f"⚠️ Направление BFGS не является направлением спуска на итерации {k}, переключаемся на антиградиент")
                d = -g
                # Сброс H к единичной, так как текущее приближение плохое
                H = np.eye(n)

            # ✅ Нормировка направления (стабилизация)
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-16:
                d_unit = d / d_norm
            else:
                d_unit = -g / (grad_norm + 1e-16)
                d_norm = grad_norm

            # ✅ Ограничение длины направления
            if d_norm > 1e10:
                d_unit = d_unit  # уже нормирован
                d_norm = 1e10

            # 3. Линейный поиск (c2=0.9 — стандарт для квазиньютоновских методов)
            alpha = None
            n_f_ls, n_g_ls = 0, 0
            try:
                alpha, n_f_ls, n_g_ls = line_search_wolfe(
                    func, grad, x, d_unit, g,
                    c2=0.9, max_iter=50,
                    count_f=self._count_f,
                    count_grad=self._count_grad
                )
            except Exception:
                alpha = None

            if alpha is None or np.isnan(alpha) or np.isinf(alpha) or alpha < 1e-16:
                # ✅ Fallback: бэктрекинг Армихо
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

            # 4. Обновление точки (используем нормированное направление!)
            s = alpha * d_unit
            x_new = x + s

            # 5. Новый градиент
            g_new = self._count_grad(grad, x_new)
            y = g_new - g

            # 6. Обновление H по формуле BFGS
            sy = np.dot(s, y)
            # ✅ Защита: обновляем только если кривизна положительна и достаточно велика
            if sy > 1e-12:
                rho = 1.0 / sy
                I = np.eye(n)
                V = I - rho * np.outer(s, y)
                H = V @ H @ V.T + rho * np.outer(s, s)
                
                # ✅ Дополнительная проверка: H должен оставаться положительно определённым
                # Быстрая проверка через диагональные элементы
                if np.any(np.diag(H) < 0):
                    if self.verbose:
                        print(f"⚠️ H потерял положительную определённость на итерации {k}, сброс к I")
                    H = np.eye(n)
            elif self.verbose and sy <= 0:
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
                n_iterations=final_k + 1, n_function_evals=self.n_function_evals,
                n_grad_evals=self.n_grad_evals, n_hess_evals=0,
                history=history, converged=converged, method_name="BFGS"
            ))

        return OptimizationResult(
            x=x,
            f_val=float(f_val),
            grad_norm=float(grad_norm),
            n_iterations=final_k + 1,
            n_function_evals=int(self.n_function_evals),
            n_grad_evals=int(self.n_grad_evals),
            n_hess_evals=0,
            history=history,
            converged=converged,
            method_name="BFGS"
        )