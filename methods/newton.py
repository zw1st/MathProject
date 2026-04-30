from typing import Callable, Optional

import numpy as np

from models.optimization_result import OptimizationResult
from methods.optimizer_abstract import Optimizer
from utils import modify_hessian, line_search_wolfe


class NewtonMethod(Optimizer):
    """Модифицированный метод Ньютона с регуляризацией Гессиана"""

    def __init__(self,
                 epsilon: float = 0.001,
                 max_iterations: int = 100,
                 verbose: bool = True):
        super().__init__(epsilon, max_iterations, verbose)

    def optimize(self,
                 func: Callable,
                 grad: Callable,
                 x0: np.ndarray,
                 hess: Optional[Callable] = None) -> OptimizationResult:

        self.reset_counters()

        if hess is None:
            raise ValueError("Метод Ньютона требует Гессиан!")

        x = x0.copy().astype(np.float64)
        n = len(x0)
        history = {
            'k': [], 'x': [], 'f': [], 'grad_norm': [], 'alpha': [], 'd_norm': []
        }

        converged = False
        final_k = 0

        if self.verbose:
            self._print_header(n)

        for k in range(self.max_iterations):
            f_val = self._count_f(func, x)
            g = self._count_grad(grad, x)
            grad_norm = np.linalg.norm(g)

            # ✅ Проверка на расходимость
            if np.isinf(f_val) or grad_norm > 1e15:
                if self.verbose:
                    print(f"⚠️ Расходимость на итерации {k}: f={f_val}, ||grad||={grad_norm}")
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

            # 3. Вычисляем Гессиан
            H = self._count_hess(hess, x)

            # 4. Модификация для положительной определённости
            H = modify_hessian(H, verbose=self.verbose)

            # 5. Направление Ньютона
            try:
                d = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                if self.verbose:
                    print(f"⚠️ LinAlgError на итерации {k}, используем антиградиент")
                d = -g

            # ✅ Проверка направления спуска: d должно быть направлением убывания
            if np.dot(g, d) >= 0:
                if self.verbose:
                    print(f"⚠️ Направление Ньютона не является направлением спуска на итерации {k}, переключаемся на антиградиент")
                d = -g

            # ✅ Нормировка направления (стабилизация в оврагах)
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-16:
                d_unit = d / d_norm
            else:
                d_unit = -g / (grad_norm + 1e-16)
                d_norm = grad_norm  # для истории

            # ✅ Ограничение длины направления
            if d_norm > 1e10:
                d_unit = d_unit  # уже нормирован
                d_norm = 1e10

            # 6. Линейный поиск (c2=0.9 — стандарт для Ньютона)
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
            history['d_norm'].append(d_norm)

            if self.verbose:
                self._print_iteration(k, x, f_val, grad_norm, alpha)

            # 7. Обновление точки
            x = x + alpha * d_unit
            final_k = k

        # Финальная точка
        f_val = self._count_f(func, x)
        g = self._count_grad(grad, x)
        grad_norm = np.linalg.norm(g)

        if self.verbose:
            self._print_footer(OptimizationResult(
                x=x, f_val=f_val, grad_norm=grad_norm,
                n_iterations=final_k + 1, n_function_evals=self.n_function_evals,
                n_grad_evals=self.n_grad_evals, n_hess_evals=self.n_hess_evals,
                history=history, converged=converged, method_name="Ньютон"
            ))

        return OptimizationResult(
            x=x,
            f_val=float(f_val),
            grad_norm=float(grad_norm),
            n_iterations=final_k + 1,
            n_function_evals=int(self.n_function_evals),
            n_grad_evals=int(self.n_grad_evals),
            n_hess_evals=int(self.n_hess_evals),
            history=history,
            converged=converged,
            method_name="Ньютон"
        )