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

        x = x0.copy()
        history = {
            'k': [], 'x': [], 'f': [], 'grad_norm': [], 'alpha': [], 'd_norm': []
        }

        converged = False
        final_k = 0

        if self.verbose:
            self._print_header()

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

            # ✅ Ограничение длины направления
            d_norm = np.linalg.norm(d)
            if d_norm > 1e10:
                d = d / d_norm * 1e10
                d_norm = 1e10

            # 6. Линейный поиск
            alpha = line_search_wolfe(func, grad, x, d, g)
            
            if np.isnan(alpha) or np.isinf(alpha) or alpha < 1e-16:
                alpha = 1e-10

            history['alpha'].append(alpha)
            history['d_norm'].append(d_norm)

            if self.verbose:
                self._print_iteration(k, x, f_val, grad_norm, alpha)

            # 7. Обновление точки
            x = x + alpha * d
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
            f_val=f_val,
            grad_norm=grad_norm,
            n_iterations=final_k + 1,
            n_function_evals=self.n_function_evals,
            n_grad_evals=self.n_grad_evals,
            n_hess_evals=self.n_hess_evals,
            history=history,
            converged=converged,
            method_name="Ньютон"
        )