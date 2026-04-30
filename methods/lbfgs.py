import numpy as np
from typing import Callable, Deque, List, Optional, Tuple
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
        self._history: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=m)

    def _two_loop_recursion(self, g: np.ndarray) -> np.ndarray:
        """
        Двухцикловая рекурсия для вычисления H @ g без явного хранения H.
        
        Возвращает направление поиска d = -H @ g
        """
        q = g.copy().astype(np.float64)
        alpha_list: List[float] = []

        # Первый цикл (в обратном порядке)
        for s, y, rho in reversed(self._history):
            alpha = rho * np.dot(s, q)
            alpha_list.append(alpha)
            q = q - alpha * y

        # Начальное приближение H_0 = gamma * I
        if len(self._history) > 0:
            s_last, y_last, _ = self._history[-1]
            gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-16)
            # ✅ Защита: gamma должна быть положительной
            gamma = max(gamma, 1e-10)
        else:
            gamma = 1.0

        r = gamma * q

        # Второй цикл (в прямом порядке)
        for (s, y, rho), alpha in zip(self._history, reversed(alpha_list)):
            beta = rho * np.dot(y, r)
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

        x = x0.copy().astype(np.float64)
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

            # ✅ Проверка направления спуска
            if np.dot(g, d) >= 0:
                if self.verbose:
                    print(f"⚠️ Направление L-BFGS не является направлением спуска на итерации {k}, переключаемся на антиградиент")
                d = -g
                # Сброс истории, так как текущее приближение плохое
                self._history.clear()

            # ✅ Нормировка направления
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

            # 4. Линейный поиск (c2=0.9 — стандарт для квазиньютоновских методов)
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

            # 5. Обновление точки (используем нормированное направление!)
            s = alpha * d_unit
            x_new = x + s

            # 6. Новый градиент
            g_new = self._count_grad(grad, x_new)
            y = g_new - g

            # 7. Обновление истории (только если кривизна положительна)
            sy = np.dot(s, y)
            if sy > 1e-12:
                rho = 1.0 / sy
                self._history.append((s.copy(), y.copy(), rho))
            elif self.verbose and sy <= 0:
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
                n_iterations=final_k + 1, n_function_evals=self.n_function_evals,
                n_grad_evals=self.n_grad_evals, n_hess_evals=0,
                history=history, converged=converged, method_name="L-BFGS"
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
            method_name="L-BFGS"
        )