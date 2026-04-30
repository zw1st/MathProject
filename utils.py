import numpy as np
from typing import Callable, Optional, Tuple


def line_search_wolfe(func: Callable,
                      grad: Callable,
                      x: np.ndarray,
                      d: np.ndarray,
                      g: np.ndarray,
                      c1: float = 1e-4,
                      c2: float = 0.9,
                      max_iter: int = 50,
                      count_f: Optional[Callable] = None,
                      count_grad: Optional[Callable] = None) -> Tuple[float, int, int]:
    """
    Линейный поиск с условиями Вольфе (исправленная версия).

    Возвращает:
        alpha: float            — найденный шаг
        n_f_evals: int          — количество вызовов функции внутри поиска
        n_grad_evals: int       — количество вызовов градиента внутри поиска
    """
    alpha = 1.0
    alpha_lo = 0.0
    alpha_hi = np.inf

    n_f_evals = 0
    n_grad_evals = 0

    def eval_f(x_eval: np.ndarray) -> float:
        nonlocal n_f_evals
        n_f_evals += 1
        if count_f is not None:
            return count_f(func, x_eval)
        else:
            val = func(x_eval)
            if np.isnan(val) or np.isinf(val):
                return np.inf
            return val

    def eval_grad(x_eval: np.ndarray) -> np.ndarray:
        nonlocal n_grad_evals
        n_grad_evals += 1
        if count_grad is not None:
            return count_grad(grad, x_eval)
        else:
            g_val = grad(x_eval)
            if np.any(np.isnan(g_val)) or np.any(np.isinf(g_val)):
                return np.zeros_like(g_val)
            return g_val

    f0 = eval_f(x)
    if np.isnan(f0) or np.isinf(f0):
        return 1e-10, n_f_evals, n_grad_evals

    g0 = g @ d
    # Если направление не является направлением спуска, возвращаем минимальный шаг
    if g0 >= 0:
        return 1e-10, n_f_evals, n_grad_evals

    # Для стабильности бисекции отслеживаем ширину интервала
    min_interval = 1e-14

    for _ in range(max_iter):
        x_new = x + alpha * d
        f_new = eval_f(x_new)

        # 1. Защита от численной неустойчивости функции
        if np.isnan(f_new) or np.isinf(f_new):
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) * 0.5 if alpha_hi != np.inf else alpha * 0.5
            if alpha < 1e-16:
                return 1e-16, n_f_evals, n_grad_evals
            continue

        # 2. Условие достаточного убывания (Армихо)
        if f_new > f0 + c1 * alpha * g0:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) * 0.5
            if alpha < 1e-16:
                return 1e-16, n_f_evals, n_grad_evals
            continue

        g_new = eval_grad(x_new)
        if np.any(np.isnan(g_new)) or np.any(np.isinf(g_new)):
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) * 0.5 if alpha_hi != np.inf else alpha * 0.5
            if alpha < 1e-16:
                return 1e-16, n_f_evals, n_grad_evals
            continue

        # 3. Условие кривизны (Вольфе) — ДВУСТОРОННЕЕ
        g_new_dot_d = g_new @ d

        # Случай A: слишком крутой спуск (производная слишком отрицательна) → увеличить шаг
        if g_new_dot_d < c2 * g0:
            alpha_lo = alpha
            if alpha_hi == np.inf:
                alpha *= 2.0
                if alpha > 1e6:
                    alpha = 1e6
            else:
                alpha = (alpha_lo + alpha_hi) * 0.5
            if alpha < 1e-16:
                return 1e-16, n_f_evals, n_grad_evals
            continue

        # Случай B: слишком пологий спуск (производная слишком положительна) → уменьшить шаг
        if g_new_dot_d > -c2 * g0:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) * 0.5
            if alpha < 1e-16:
                return 1e-16, n_f_evals, n_grad_evals
            continue

        # Условия Вольфе выполнены: c2*g0 <= g_new_dot_d <= -c2*g0  (эквивалентно |g_new_dot_d| <= -c2*g0)
        # Проверка сходимости интервала
        if alpha_hi != np.inf and (alpha_hi - alpha_lo) < min_interval:
            return alpha, n_f_evals, n_grad_evals

        return alpha, n_f_evals, n_grad_evals

    # Fallback: возвращаем последний допустимый шаг
    return max(alpha, 1e-16), n_f_evals, n_grad_evals


def modify_hessian(H: np.ndarray, delta: float = 1e-6, verbose: bool = False) -> np.ndarray:
    """
    Модифицировать Гессиан для положительной определённости
    
    Параметры:
    - H: исходный Гессиан
    - delta: параметр регуляризации
    - verbose: выводить сообщения
    
    Возвращает:
    - H_mod: модифицированный Гессиан
    """
    n = H.shape[0]

    # ✅ Проверка на NaN/Inf
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        if verbose:
            print("⚠️ Гессиан содержит NaN/Inf, возвращаем единичную матрицу")
        return np.eye(n)

    # Ограничиваем значения
    H = np.clip(H, -1e10, 1e10)

    # ✅ Симметризация (из-за численных погрешностей Гессиан может быть несимметричным)
    H = 0.5 * (H + H.T)

    # Пробуем разложение Холецкого
    try:
        np.linalg.cholesky(H)
        return H
    except np.linalg.LinAlgError:
        pass

    # Вычисляем собственные значения
    try:
        min_eig = np.min(np.linalg.eigvalsh(H))
        if min_eig <= 0:
            H = H + (abs(min_eig) + delta) * np.eye(n)
            if verbose:
                print(f"🔧 Гессиан модифицирован: min_eig={min_eig:.6f}")
    except np.linalg.LinAlgError:
        # Если собственные значения не сошлись
        H = H + (1.0 + delta) * np.eye(n)
        if verbose:
            print("⚠️ Собственные значения не сошлись, добавлена диагональ")

    return H
