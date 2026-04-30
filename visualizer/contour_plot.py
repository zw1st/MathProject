import numpy as np
import os
from typing import List, Tuple, Optional
import matplotlib
# ✅ Принудительно используем интерактивный бэкенд
matplotlib.use('TkAgg')  # Должен быть ДО импорта pyplot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from functions.test_functions import TestFunction
from models.optimization_result import OptimizationResult


def plot_contour_with_tracks(
    func: TestFunction,
    results: List[OptimizationResult],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = True,
    figure: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    slice_dims: Tuple[int, int] = (0, 1),
    slice_fixed_values: Optional[np.ndarray] = None,
    use_x0_for_slice: bool = True,
    n_levels: int = 30,
    use_line_colors: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Построить контурный график с траекториями методов.
    
    Параметры:
    ----------
    func : TestFunction
        Тестовая функция
    results : List[OptimizationResult]
        Результаты оптимизации
    title : str, optional
        Заголовок графика
    figsize : tuple, optional
        Размер фигуры (по умолчанию (12, 10))
    dpi : int, optional
        Разрешение (по умолчанию 150)
    save_path : str, optional
        Путь для сохранения
    show : bool, optional
        Показывать график (по умолчанию True)
    figure : Figure, optional
        Существующая фигура (для GUI)
    ax : Axes, optional
        Существующие оси (для GUI)
    slice_dims : tuple, optional
        Какие переменные показать для 10D (по умолчанию (0, 1))
    slice_fixed_values : ndarray, optional
        Фиксированные значения для остальных переменных
    use_x0_for_slice : bool, optional
        Использовать x⁰ для фиксированных переменных
    n_levels : int, optional
        Количество уровней контуров (по умолчанию 30)
    use_line_colors : bool, optional
        Если True — цвет линий показывает высоту (по умолчанию True)
    
    Возвращает:
    -----------
    Tuple[Figure, Axes]
        Фигура и оси
    """
    
    # Проверка размерности
    n_dims = len(func.x0_default) if hasattr(func, 'x0_default') else len(results[0].x)
    
    if func.is_2d:
        slice_dims = (0, 1)
        slice_fixed_values_bound = None
    elif n_dims > 2:
        if slice_fixed_values is None:
            if use_x0_for_slice and hasattr(func, 'x0_default'):
                slice_fixed_values_bound = func.x0_default.copy()
            else:
                slice_fixed_values_bound = np.zeros(n_dims)
        else:
            slice_fixed_values_bound = slice_fixed_values.copy()
    else:
        raise ValueError(f"Некорректная размерность: {n_dims}")
    
    # Создание или использование существующей фигуры
    created_figure = False
    if figure is None or ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        created_figure = True
    else:
        fig = figure
        ax.clear()
    
    # Настройка шрифтов
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })
    
    # Границы для выбранных измерений
    if func.is_2d:
        x1_min, x1_max = func.bounds[0][0], func.bounds[1][0]
        x2_min, x2_max = func.bounds[0][1], func.bounds[1][1]
    else:
        x1_min, x1_max = func.bounds[0][slice_dims[0]], func.bounds[1][slice_dims[0]]
        x2_min, x2_max = func.bounds[0][slice_dims[1]], func.bounds[1][slice_dims[1]]
    
    # Отступы 15%
    x1_range = x1_max - x1_min
    x2_range = x2_max - x2_min
    x1_min -= 0.15 * x1_range
    x1_max += 0.15 * x1_range
    x2_min -= 0.15 * x2_range
    x2_max += 0.15 * x2_range
    
    # Сетка
    n_grid = 200 if not func.is_2d else 500
    x1 = np.linspace(x1_min, x1_max, n_grid)
    x2 = np.linspace(x2_min, x2_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Вычисление Z
    Z = np.full_like(X1, np.nan, dtype=float)
    
    if func.is_2d:
        points = np.stack([X1.ravel(), X2.ravel()], axis=1)
        for idx in range(points.shape[0]):
            try:
                val = func.func(points[idx])
                if np.isfinite(val):
                    Z.ravel()[idx] = val
            except Exception:
                pass
    else:
        for i in range(n_grid):
            for j in range(n_grid):
                try:
                    point = slice_fixed_values_bound.copy()
                    point[slice_dims[0]] = X1[j, i]
                    point[slice_dims[1]] = X2[j, i]
                    val = func.func(point)
                    if np.isfinite(val):
                        Z[j, i] = val
                except Exception:
                    pass
    
    Z_masked = np.ma.masked_invalid(Z)
    
    # Проверка на валидность
    if np.all(Z_masked.mask):
        ax.text(0.5, 0.5, "Не удалось построить график",
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        if show and created_figure:
            plt.show()
            plt.close(fig)
        return fig, ax
    
    # Уровни контуров
    z_valid = Z_masked.compressed()
    if len(z_valid) == 0:
        ax.text(0.5, 0.5, "Нет допустимых значений функции",
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        if show and created_figure:
            plt.show()
            plt.close(fig)
        return fig, ax
    
    z_min, z_max = np.min(z_valid), np.max(z_valid)
    
    if np.isclose(z_min, z_max):
        z_min -= 1.0
        z_max += 1.0
    
    levels = np.linspace(z_min, z_max, n_levels)
    
    # КОНТУРЫ
    if use_line_colors:
        contour = ax.contour(
            X1, X2, Z_masked, levels=levels,
            cmap='viridis',
            linewidths=2.0,
            alpha=0.9
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f', colors='black')
        
        ax.contourf(
            X1, X2, Z_masked, levels=levels,
            cmap='viridis', alpha=0.15
        )
    else:
        contour = ax.contour(
            X1, X2, Z_masked, levels=levels,
            colors='black', linewidths=1.0, alpha=0.7
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        ax.contourf(
            X1, X2, Z_masked, levels=levels,
            cmap='viridis', alpha=0.5
        )
    
    # Цвета для методов
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    markers = ['o', 's', '^', 'D', 'p']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Траектории методов
    for idx, result in enumerate(results):
        # СТРОГАЯ валидация истории
        if 'x' not in result.history:
            continue
            
        x_history_raw = result.history['x']
        
        if isinstance(x_history_raw, list):
            if len(x_history_raw) == 0:
                continue
            
            # Безопасная конвертация с проверкой каждого элемента
            try:
                x_history_list = []
                for elem in x_history_raw:
                    if isinstance(elem, (np.ndarray, list)):
                        arr = np.asarray(elem, dtype=np.float64)
                        if arr.ndim == 1 and len(arr) >= 2:
                            x_history_list.append(arr)
                
                if len(x_history_list) == 0:
                    continue
                    
                x_history = np.array(x_history_list)
                
            except Exception:
                continue
        elif isinstance(x_history_raw, np.ndarray):
            x_history = x_history_raw.astype(np.float64)
            if x_history.ndim != 2 or x_history.shape[1] < 2:
                continue
        else:
            continue
        
        # Финальная проверка на NaN/Inf
        if np.any(np.isnan(x_history[:, :2])) or np.any(np.isinf(x_history[:, :2])):
            valid_rows = ~(np.isnan(x_history[:, 0]) | np.isnan(x_history[:, 1]) |
                          np.isinf(x_history[:, 0]) | np.isinf(x_history[:, 1]))
            x_history = x_history[valid_rows]
            if len(x_history) == 0:
                continue
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        line_style = line_styles[idx % len(line_styles)]
        
        # ✅ ИСПРАВЛЕНИЕ: markevery должен быть int или None
        n_points = len(x_history)
        if n_points > 1:
            markevery = max(1, n_points // 12)
        else:
            markevery = None  # Для 1-2 точек не используем markevery
        
        try:
            # ✅ Явное преобразование координат в 1D массивы
            x_coords = np.asarray(x_history[:, 0], dtype=np.float64).ravel()
            y_coords = np.asarray(x_history[:, 1], dtype=np.float64).ravel()
            
            # Линия траектории
            if markevery is not None and n_points > 1:
                ax.plot(
                    x_coords, y_coords,
                    line_style, color=color, linewidth=2.5,
                    label=f"{result.method_name} ({result.n_iterations} итер.)",
                    alpha=0.85, marker=marker, markersize=8,
                    markevery=markevery,
                    markeredgecolor='black', markeredgewidth=1.2
                )
            else:
                # Без маркеров для коротких траекторий
                ax.plot(
                    x_coords, y_coords,
                    line_style, color=color, linewidth=2.5,
                    label=f"{result.method_name} ({result.n_iterations} итер.)",
                    alpha=0.85,
                    markeredgecolor='black', markeredgewidth=1.2
                )
            
            # Начальная точка
            ax.plot(
                x_coords[0], y_coords[0],
                'k*', markersize=18, markeredgecolor='gold', markeredgewidth=2.5,
                label='Начальная точка' if idx == 0 else '', zorder=10
            )
            
            # Найденный минимум
            ax.plot(
                x_coords[-1], y_coords[-1],
                marker, color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=2.5, zorder=10
            )
            
            # Подпись метода
            ax.annotate(
                f'{result.method_name}',
                (x_coords[-1], y_coords[-1]),
                textcoords="offset points", xytext=(15, 15),
                fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=color, alpha=0.85)
            )
        except Exception as e:
            print(f"❌ Ошибка построения для {result.method_name}: {e}")
            continue
    
    # Точный минимум
    if func.x_min is not None and len(func.x_min) >= 2:
        try:
            x_min_arr = np.asarray(func.x_min, dtype=np.float64).ravel()
            if len(x_min_arr) >= 2 and np.isfinite(x_min_arr[:2]).all():
                ax.plot(
                    x_min_arr[0], x_min_arr[1], 'mX', markersize=15,
                    markeredgecolor='black', markeredgewidth=2.5,
                    label='Точный минимум', zorder=10
                )
        except Exception:
            pass
    
    # Оформление
    if func.is_2d:
        ax.set_xlabel(r'$x_1$', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(r'$x_2$', fontsize=14, fontweight='bold', labelpad=10)
    else:
        ax.set_xlabel(rf'$x_{{{slice_dims[0]+1}}}$', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(rf'$x_{{{slice_dims[1]+1}}}$', fontsize=14, fontweight='bold', labelpad=10)
    
    base_title = title or f"Оптимизация: {func.name}"
    if not func.is_2d:
        base_title += f"\n(сечение: x[{slice_dims[0]+1}], x[{slice_dims[1]+1}])"
    ax.set_title(base_title, fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.set_aspect('auto')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    if created_figure or (figure is not None):
        try:
            plt.ion()
        except Exception:
            pass
    
    # Сохранение
    if save_path:
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 График сохранён: {save_path}")
    
    # Показ
    if show and created_figure:
        plt.show()
    
    return fig, ax