import numpy as np

from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TestFunction:
    """Тестовая функция для оптимизации"""
    name: str
    func: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hess: Optional[Callable[[np.ndarray], np.ndarray]]
    x0_default: np.ndarray
    x_min: np.ndarray
    f_min: float
    bounds: Tuple[np.ndarray, np.ndarray]
    is_2d: bool

class TestFunctions:
    """Коллекция тестовых функций для оптимизации"""

    @staticmethod
    def himmelblau() -> TestFunction:
        """Функция Химмельблау"""

        def f(x):
            return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        def grad(x):
            g = np.zeros(2)
            g[0] = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
            g[1] = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
            return g

        def hess(x):
            h = np.zeros((2, 2))
            h[0, 0] = 12 * x[0] ** 2 + 4 * x[1] - 42
            h[0, 1] = 4 * x[0] + 4 * x[1]
            h[1, 0] = 4 * x[0] + 4 * x[1]
            h[1, 1] = 4 * x[0] + 12 * x[1] ** 2 - 26
            return h

        return TestFunction(
            name="Химмельблау",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([0, 0]),
            x_min=np.array([3.0, 2.0]),
            f_min=0.0,
            bounds=(np.array([-5, -5]), np.array([5, 5])),
            is_2d=True
        )

    @staticmethod
    def rosenbrock() -> TestFunction:
        """Функция Розенброка"""

        def f(x):
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        def grad(x):
            g = np.zeros(2)
            g[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
            g[1] = 200 * (x[1] - x[0] ** 2)
            return g

        def hess(x):
            h = np.zeros((2, 2))
            h[0, 0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
            h[0, 1] = -400 * x[0]
            h[1, 0] = -400 * x[0]
            h[1, 1] = 200
            return h

        return TestFunction(
            name="Розенброк",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([-1.2, 1.0]),
            x_min=np.array([1.0, 1.0]),
            f_min=0.0,
            bounds=(np.array([-2, -1]), np.array([2, 3])),
            is_2d=True
        )

    @staticmethod
    def booth() -> TestFunction:
        """Функция Бута"""

        def f(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        def grad(x):
            g = np.zeros(2)
            g[0] = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
            g[1] = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
            return g

        def hess(x):
            return np.array([[10, 8], [8, 20]])

        return TestFunction(
            name="Бут",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([0.0, 0.0]),
            x_min=np.array([1.0, 3.0]),
            f_min=0.0,
            bounds=(np.array([-10, -10]), np.array([10, 10])),
            is_2d=True
        )

    @staticmethod
    def beale() -> TestFunction:
        """Функция Била"""

        def f(x):
            return (1.5 - x[0] + x[0] * x[1]) ** 2 + \
                (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + \
                (2.625 - x[0] + x[0] * x[1] ** 3) ** 2

        def grad(x):
            g = np.zeros(2)
            t1 = 1.5 - x[0] + x[0] * x[1]
            t2 = 2.25 - x[0] + x[0] * x[1] ** 2
            t3 = 2.625 - x[0] + x[0] * x[1] ** 3
            g[0] = 2 * t1 * (-1 + x[1]) + 2 * t2 * (-1 + x[1] ** 2) + 2 * t3 * (-1 + x[1] ** 3)
            g[1] = 2 * t1 * x[0] + 2 * t2 * 2 * x[0] * x[1] + 2 * t3 * 3 * x[0] * x[1] ** 2
            return g

        def hess(x):
            h = np.zeros((2, 2))
            t1 = 1.5 - x[0] + x[0] * x[1]
            t2 = 2.25 - x[0] + x[0] * x[1] ** 2
            t3 = 2.625 - x[0] + x[0] * x[1] ** 3
            h[0, 0] = 2 * (-1 + x[1]) ** 2 + 2 * (-1 + x[1] ** 2) ** 2 + 2 * (-1 + x[1] ** 3) ** 2
            h[0, 1] = 2 * (-1 + x[1]) * x[0] + 2 * t1 + 2 * (-1 + x[1] ** 2) * 2 * x[0] * x[1] + \
                      2 * t2 * 2 * x[1] + 2 * (-1 + x[1] ** 3) * 3 * x[0] * x[1] ** 2 + 2 * t3 * 3 * x[1] ** 2
            h[1, 0] = h[0, 1]
            h[1, 1] = 2 * x[0] ** 2 + 2 * t2 * 2 * x[0] + 2 * 2 * x[0] * x[1] * 2 * x[0] * x[1] + \
                      2 * t3 * 6 * x[0] * x[1] + 2 * 3 * x[0] * x[1] ** 2 * 3 * x[0] * x[1] ** 2
            return h

        return TestFunction(
            name="Бил",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([0.0, 0.0]),
            x_min=np.array([3.0, 0.5]),
            f_min=0.0,
            bounds=(np.array([-4.5, -4.5]), np.array([4.5, 4.5])),
            is_2d=True
        )

    @staticmethod
    def matyas() -> TestFunction:
        """Функция Матьяс"""

        def f(x):
            return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

        def grad(x):
            g = np.zeros(2)
            g[0] = 0.52 * x[0] - 0.48 * x[1]
            g[1] = 0.52 * x[1] - 0.48 * x[0]
            return g

        def hess(x):
            return np.array([[0.52, -0.48], [-0.48, 0.52]])

        return TestFunction(
            name="Матьяс",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([5.0, 5.0]),
            x_min=np.array([0.0, 0.0]),
            f_min=0.0,
            bounds=(np.array([-10, -10]), np.array([10, 10])),
            is_2d=True
        )

    @staticmethod
    def three_hump_camel() -> TestFunction:
        """Функция трёхгорбый верблюд"""

        def f(x):
            return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2

        def grad(x):
            g = np.zeros(2)
            g[0] = 4 * x[0] - 4.2 * x[0] ** 3 + x[0] ** 5 + x[1]
            g[1] = x[0] + 2 * x[1]
            return g

        def hess(x):
            h = np.zeros((2, 2))
            h[0, 0] = 4 - 12.6 * x[0] ** 2 + 5 * x[0] ** 4
            h[0, 1] = 1
            h[1, 0] = 1
            h[1, 1] = 2
            return h

        return TestFunction(
            name="Трёхгорбый верблюд",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([2.0, 2.0]),
            x_min=np.array([0.0, 0.0]),
            f_min=0.0,
            bounds=(np.array([-3, -3]), np.array([3, 3])),
            is_2d=True
        )

    @staticmethod
    def sphere() -> TestFunction:
        """Функция Сфера"""

        def f(x):
            return np.sum(x ** 2)

        def grad(x):
            return 2 * x

        def hess(x):
            return 2 * np.eye(len(x))

        return TestFunction(
            name="Сфера",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([5.0, 5.0]),
            x_min=np.array([0.0, 0.0]),
            f_min=0.0,
            bounds=(np.array([-10, -10]), np.array([10, 10])),
            is_2d=True
        )

    @staticmethod
    def rastrigin() -> TestFunction:
        """Функция Растригин"""

        def f(x):
            return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

        def grad(x):
            g = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
            return g

        def hess(x):
            h = np.diag(2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * x))
            return h

        return TestFunction(
            name="Растригин",
            func=f, grad=grad, hess=hess,
            x0_default=np.array([2.5, 2.5]),
            x_min=np.array([0.0, 0.0]),
            f_min=0.0,
            bounds=(np.array([-5.12, -5.12]), np.array([5.12, 5.12])),
            is_2d=True
        )
    

    # @staticmethod
    # def polynomial_10d() -> TestFunction:
    #     """
    #     Полином 10-й степени от 10 переменных (индивидуальный вариант)
        
    #     f(x) = Σ dᵢ·xᵢ² + Σ cᵢⱼ·xᵢ·xⱼ + Σ lᵢ·xᵢ + α·xₖ^p
    #     """
    #     n = 10
        
    #     # Диагональные коэффициенты (все > 0)
    #     d = np.array([3.2, 2.5, 4.6, 1.8, 3.9, 2.1, 4.3, 1.5, 3.7, 2.8])
        
    #     # Перекрёстные коэффициенты (3 пары): (i, j, coefficient)
    #     cross_terms = [
    #         (0, 1, 0.5),    # 0.5 * x₀ * x₁
    #         (2, 4, -0.3),   # -0.3 * x₂ * x₄
    #         (5, 7, 0.4)     # 0.4 * x₅ * x₇
    #     ]
        
    #     # Линейные коэффициенты
    #     l = np.array([0.1, -0.2, 0.0, 0.3, 0.0, -0.1, 0.0, 0.2, 0.0, -0.15])
        
    #     # Высокостепенной член: 0.002 * x₃⁸
    #     alpha = 0.002
    #     k = 3  # индекс переменной (x₃)
    #     p = 8  # степень
        
    #     # Начальная точка
    #     x0_default = np.array([-1.3, 2.4, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    #     def f(x):
    #         result = np.sum(d * x ** 2)
    #         for i, j, c in cross_terms:
    #             result += c * x[i] * x[j]
    #         result += np.sum(l * x)
    #         result += alpha * x[k] ** p
    #         return result
        
    #     def grad(x):
    #         g = np.zeros(n)
    #         g += 2 * d * x
    #         for i, j, c in cross_terms:
    #             g[i] += c * x[j]
    #             g[j] += c * x[i]
    #         g += l
    #         g[k] += alpha * p * x[k] ** (p - 1)
    #         return g
        
    #     def hess(x):
    #         H = np.zeros((n, n))
    #         np.fill_diagonal(H, 2 * d)
    #         for i, j, c in cross_terms:
    #             H[i, j] = c
    #             H[j, i] = c
    #         H[k, k] += alpha * p * (p - 1) * x[k] ** (p - 2)
    #         return H
        
    #     return TestFunction(
    #         name="Полином 10D (вариант)",
    #         func=f,
    #         grad=grad,
    #         hess=hess,
    #         x0_default=x0_default,
    #         x_min=np.zeros(n),
    #         f_min=0.0,
    #         bounds=(np.full(n, -3.0), np.full(n, 3.0)),
    #         is_2d=False
    #     )

    @classmethod
    def get_all_2d(cls) -> List[TestFunction]:
        """Получить все 2D функции"""
        return [
            cls.himmelblau(), cls.rosenbrock(), cls.booth(),
            cls.beale(), cls.matyas(), cls.three_hump_camel(),
            cls.sphere(), cls.rastrigin()
        ]

    @classmethod
    def get_by_name(cls, name: str) -> TestFunction:

        """Получить функцию по имени"""
        functions = {
            'himmelblau': cls.himmelblau,
            'rosenbrock': cls.rosenbrock,
            'booth': cls.booth,
            'beale': cls.beale,
            'matyas': cls.matyas,
            'three_hump_camel': cls.three_hump_camel,
            'sphere': cls.sphere,
            'rastrigin': cls.rastrigin
        }
        return functions[name.lower()]()
    
    @staticmethod
    def rosenbrock_nd(n: int = 10) -> TestFunction:
        """n-мерная функция Розенброка (овраг)"""
        
        def f(x):
            return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n-1))
        
        def grad(x):
            g = np.zeros(n)
            for i in range(n-1):
                g[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
                g[i+1] += 200*(x[i+1] - x[i]**2)
            return g
        
        def hess(x):
            H = np.zeros((n, n))
            for i in range(n-1):
                H[i, i] += 1200*x[i]**2 - 400*x[i+1] + 2
                H[i, i+1] = -400*x[i]
                H[i+1, i] = -400*x[i]
                H[i+1, i+1] += 200
            return H
        
        return TestFunction(
            name=f"Розенброк {n}D",
            func=f, grad=grad, hess=hess,
            x0_default=np.full(n, 0.0),
            x_min=np.ones(n),
            f_min=0.0,
            bounds=(np.full(n, -5.0), np.full(n, 10.0)),
            is_2d=False
        )