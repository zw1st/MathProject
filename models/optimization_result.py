import numpy as np
from typing import Any, List, Dict
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Результат оптимизации"""
    x: np.ndarray
    f_val: float
    grad_norm: float | np.floating
    n_iterations: int
    n_function_evals: int
    n_grad_evals: int
    n_hess_evals: int  # Новый счётчик для Гессиана
    history: Dict[str, List[Any]]
    converged: bool
    method_name: str