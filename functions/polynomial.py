"""
Полином 10D с автоматическим дифференцированием через JAX
"""

import numpy as np
import json
from pathlib import Path
from typing import Callable, Dict, Any


import jax
import jax.numpy as jnp
from jax import grad, hessian

from functions.test_functions import TestFunction




class Polynomial10D:
    """
    Полином 10-й степени с JAX autodiff.
    
    f(x) = Σ dᵢ·xᵢ² + Σ cᵢⱼ·xᵢ·xⱼ + Σ lᵢ·xᵢ + α·xₖ^p
    
    Градиент и Гессиан вычисляются автоматически через JAX.
    """
    
    def __init__(self, config_path: str = "config/polynomial.json"):
        """
        Инициализация полинома из JSON конфигурации.
        
        Параметры:
        - config_path: путь к JSON файлу с коэффициентами
        """
        
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Построение JAX-функции
        self._f_jax = self._build_function()
        
        # Авто-вычисление градиента и Гессиана
        self._grad_jax = grad(self._f_jax)
        self._hess_jax = hessian(self._f_jax)
        
        # Компиляция для ускорения
        self._f_compiled = jax.jit(self._f_jax)
        self._grad_compiled = jax.jit(self._grad_jax)
        self._hess_compiled = jax.jit(self._hess_jax)
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Загрузить конфигурацию из JSON"""
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфигурация не найдена: {path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_function(self) -> Callable:
        """Построить JAX-функцию полинома"""
        # Извлечение коэффициентов из конфига
        d = jnp.array(self.config["diagonal"], dtype=jnp.float64)
        cross = self.config["cross_terms"]
        l = jnp.array(self.config["linear"], dtype=jnp.float64)
        alpha = float(self.config["high_degree"]["alpha"])
        k = int(self.config["high_degree"]["k"])
        p = int(self.config["high_degree"]["p"])
        
        def f(x: jnp.ndarray) -> jnp.ndarray:
            # Квадратичная часть: Σ dᵢ·xᵢ²
            result = jnp.sum(d * x ** 2)
            
            # Перекрёстные члены: Σ cᵢⱼ·xᵢ·xⱼ
            for term in cross:
                i, j, c = term["i"], term["j"], float(term["c"])
                result += c * x[i] * x[j]
            
            # Линейная часть: Σ lᵢ·xᵢ
            result += jnp.sum(l * x)
            
            # Высокостепенной член: α·xₖ^p
            result += alpha * x[k] ** p
            
            return result
        
        return f
    
    def f(self, x: np.ndarray) -> float:
        """Вычислить значение функции"""
        x_jax = jnp.array(x, dtype=jnp.float64)
        return float(self._f_compiled(x_jax))
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Вычислить градиент (авто-дифф JAX)"""
        x_jax = jnp.array(x, dtype=jnp.float64)
        return np.array(self._grad_compiled(x_jax))
    
    def hess(self, x: np.ndarray) -> np.ndarray:
        """Вычислить Гессиан (авто-дифф JAX)"""
        x_jax = jnp.array(x, dtype=jnp.float64)
        return np.array(self._hess_compiled(x_jax))
    
    @property
    def n(self) -> int:
        """Размерность"""
        return int(self.config["n"])
    
    @property
    def x0_default(self) -> np.ndarray:
        """Начальная точка по умолчанию"""
        return np.array(self.config["x0_default"], dtype=np.float64)
    
    @property
    def bounds(self) -> tuple:
        """Границы поиска"""
        min_val = float(self.config["bounds_min"])
        max_val = float(self.config["bounds_max"])
        return (np.full(self.n, min_val), np.full(self.n, max_val))
    
    @property
    def name(self) -> str:
        """Имя функции"""
        return self.config.get("name", "Полином 10D")
    
    def to_test_function(self) -> TestFunction:
        """Преобразовать в TestFunction для оптимизаторов"""
        return TestFunction(
            name=self.name,
            func=self.f,
            grad=self.grad,
            hess=self.hess,
            x0_default=self.x0_default,
            x_min=np.zeros(self.n),
            f_min=0.0,
            bounds=self.bounds,
            is_2d=False
        )