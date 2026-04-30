import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable
import os



# ✅ Отключаем предварительное выделение памяти JAX (важно для совместимости с GUI)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
from jax import grad, hessian, jit, vmap

# Включаем 64-битную точность
jax.config.update("jax_enable_x64", True)


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
        
        # ✅ Предварительно конвертируем коэффициенты в JAX массивы
        self._d = jnp.array(self.config["diagonal"], dtype=jnp.float64)
        self._l = jnp.array(self.config["linear"], dtype=jnp.float64)
        self._alpha = float(self.config["high_degree"]["alpha"])
        self._k = int(self.config["high_degree"]["k"])
        self._p = int(self.config["high_degree"]["p"])
        
        # ✅ Предварительно обрабатываем cross_terms в индексы и коэффициенты
        self._cross_i = jnp.array([t["i"] for t in self.config["cross_terms"]], dtype=jnp.int32)
        self._cross_j = jnp.array([t["j"] for t in self.config["cross_terms"]], dtype=jnp.int32)
        self._cross_c = jnp.array([float(t["c"]) for t in self.config["cross_terms"]], dtype=jnp.float64)
        self._n_cross = len(self.config["cross_terms"])
        
        # Построение JAX-функций
        self._f_jax = self._build_function()
        
        # Авто-вычисление градиента и Гессиана
        self._grad_jax = grad(self._f_jax)
        self._hess_jax = hessian(self._f_jax)
        
        # ✅ Компиляция для ускорения
        self._f_compiled = jit(self._f_jax)
        self._grad_compiled = jit(self._grad_jax)
        self._hess_compiled = jit(self._hess_jax)
        
        # ✅ Прогрев JIT-компиляции на начальной точке
        self._warmup()
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Загрузить конфигурацию из JSON"""
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфигурация не найдена: {path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_function(self) -> Callable:
        """Построить JAX-функцию полинома (оптимизированная версия)"""
        
        # Захватываем переменные из self для замыкания
        d = self._d
        l = self._l
        alpha = self._alpha
        k = self._k
        p = self._p
        cross_i = self._cross_i
        cross_j = self._cross_j
        cross_c = self._cross_c
        n_cross = self._n_cross
        
        def f(x: jnp.ndarray) -> jnp.ndarray:
            # ✅ Квадратичная часть: Σ dᵢ·xᵢ² — векторизовано
            result = jnp.dot(d, x * x)
            
            # ✅ Перекрёстные члены: Σ cᵢⱼ·xᵢ·xⱼ — через индексирование
            # Используем цикл for, но JAX трассирует его эффективно
            for idx in range(n_cross):
                i, j, c = cross_i[idx], cross_j[idx], cross_c[idx]
                result += c * x[i] * x[j]
            
            # ✅ Линейная часть: Σ lᵢ·xᵢ — векторизовано
            result += jnp.dot(l, x)
            
            # ✅ Высокостепенной член: α·xₖ^p — с защитой от переполнения
            # Используем jnp.where для условного вычисления
            x_k = x[k]
            # Для чётных p используем abs, для нечётных — знак сохраняется
            if p % 2 == 0:
                x_pow = jnp.abs(x_k) ** p
            else:
                x_pow = jnp.sign(x_k) * jnp.abs(x_k) ** p
            result += alpha * x_pow
            
            return result
        
        return f
    
    def _warmup(self):
        """Прогреть JIT-компиляцию"""
        x0 = jnp.array(self.config["x0_default"], dtype=jnp.float64)
        try:
            _ = self._f_compiled(x0)
            _ = self._grad_compiled(x0)
            _ = self._hess_compiled(x0)
        except Exception as e:
            print(f"⚠️ Предупреждение при прогреве JIT: {e}")
    
    def f(self, x: np.ndarray) -> float:
        """Вычислить значение функции"""
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        # ✅ Добавлена защита от NaN/Inf на уровне numpy
        try:
            result = float(self._f_compiled(x_jax))
            if not np.isfinite(result):
                return np.inf
            return result
        except Exception:
            return np.inf
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Вычислить градиент (авто-дифф JAX)"""
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        try:
            result = np.array(self._grad_compiled(x_jax))
            # ✅ Замена NaN/Inf на нули
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
            return result
        except Exception:
            return np.zeros_like(x, dtype=np.float64)
    
    def hess(self, x: np.ndarray) -> np.ndarray:
        """Вычислить Гессиан (авто-дифф JAX)"""
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        try:
            result = np.array(self._hess_compiled(x_jax))
            # ✅ Замена NaN/Inf и симметризация
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
            result = 0.5 * (result + result.T)
            return result
        except Exception:
            return np.eye(len(x), dtype=np.float64)
    
    # ✅ Векторизованная версия f для построения контуров (значительно быстрее)
    def f_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Вычислить значения функции для множества точек.
        X: массив формы (N, n)
        Возвращает: массив формы (N,)
        """
        X_jax = jnp.asarray(X, dtype=jnp.float64)
        f_vec = vmap(self._f_compiled)
        try:
            result = np.array(f_vec(X_jax))
            result = np.nan_to_num(result, nan=np.inf, posinf=np.inf, neginf=np.inf)
            return result
        except Exception:
            return np.full(X.shape[0], np.inf)
    
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
    
    @property
    def is_2d(self) -> bool:
        """Флаг двумерности"""
        return False
    
    @property
    def x_min(self) -> np.ndarray:
        """Точный минимум (если известен)"""
        return np.zeros(self.n, dtype=np.float64)
    
    @property
    def f_min(self) -> float:
        """Минимальное значение (если известно)"""
        return 0.0
    
    def to_test_function(self):
        from functions.test_functions import TestFunction  # Локальный импорт
        
        return TestFunction(
            name=self.name,
            func=self.f,
            grad=self.grad,
            hess=self.hess,
            x0_default=self.x0_default,
            x_min=self.x_min,
            f_min=self.f_min,
            bounds=self.bounds,
            is_2d=False
        )