"""
График сходимости методов оптимизации
"""

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Optional, Tuple, Literal
import os

from models.optimization_result import OptimizationResult




def plot_convergence(
    results: List[OptimizationResult],
    metric: Literal['f', 'grad', 'both'] = 'f',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = True,
    figure: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Построить график сходимости методов.
    
    Параметры:
    ----------
    results : List[OptimizationResult]
        Результаты оптимизации
    metric : {'f', 'grad', 'both'}, optional
        Что отображать: 'f' — функция, 'grad' — градиент, 'both' — оба (по умолчанию 'f')
    title : str, optional
        Заголовок
    figsize : tuple, optional
        Размер фигуры (по умолчанию (12, 8))
    dpi : int, optional
        Разрешение (по умолчанию 150)
    save_path : str, optional
        Путь для сохранения
    show : bool, optional
        Показывать график (по умолчанию True)
    log_scale : bool, optional
        Логарифмическая шкала Y (по умолчанию True)
    figure : Figure, optional
        Существующая фигура (для GUI)
    ax : Axes, optional
        Существующие оси (для GUI)
    
    Возвращает:
    -----------
    Tuple[Figure, Axes]
        Фигура и оси
    """
    
    if not results:
        raise ValueError("Список результатов не может быть пустым")
    
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
    
    # Цвета для методов
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    # Данные для построения
    if metric == 'both':
        metrics = ['f', 'grad']
    else:
        metrics = [metric]
    
    for plot_idx, m in enumerate(metrics):
        # Для 'both' создаём подплоты
        if metric == 'both':
            if plot_idx == 0:
                current_ax = fig.add_subplot(1, 2, 1)
            else:
                current_ax = fig.add_subplot(1, 2, 2)
        else:
            current_ax = ax
        
        # Подписи осей
        if m == 'f':
            current_ax.set_ylabel(r'$f(x^k)$', fontsize=13, fontweight='bold')
        else:
            current_ax.set_ylabel(r'$\|\nabla f(x^k)\|$', fontsize=13, fontweight='bold')
        
        current_ax.set_xlabel('Итерация k', fontsize=13, fontweight='bold')
        
        # Логарифмическая шкала
        if log_scale:
            current_ax.set_yscale('log')
        
        # Построение кривых для каждого метода
        for idx, result in enumerate(results):
            color = colors[idx % len(colors)]
            line_style = line_styles[idx % len(line_styles)]
            
            k = np.array(result.history['k'])
            
            if m == 'f':
                y_data = np.array(result.history['f'])
            else:
                y_data = np.array(result.history['grad_norm'])

            if len(k) == 0 or len(y_data) == 0:
                continue
            
            # Фильтрация невалидных значений
            valid = np.isfinite(y_data)
            k_valid = k[valid]
            y_valid = y_data[valid]
            
            if len(k_valid) == 0 or len(y_valid) == 0:
                continue
            
            # Основная линия
            current_ax.plot(
                k_valid, y_valid,
                line_style, color=color, linewidth=2.5,
                label=f"{result.method_name} ({result.n_iterations} итер.)",
                alpha=0.9, marker='o', markersize=4,
                markevery=max(1, len(k_valid) // 15),
                markerfacecolor=color, markeredgecolor='black',
                markeredgewidth=1.0
            )
            
            # Точка сходимости
            if result.converged and len(k_valid) > 0:
                current_ax.plot(
                    k_valid[-1], y_valid[-1],
                    'o', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=2, zorder=10
                )
        
        # Сетка и оформление
        current_ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0,
                       which='both' if log_scale else 'major')
        current_ax.set_facecolor('#fafafa')
        
        # Заголовок подплота
        if metric == 'both':
            subplot_title = 'Значение функции' if m == 'f' else 'Норма градиента'
            current_ax.set_title(subplot_title, fontsize=14, fontweight='bold', pad=10)
    
    # Общий заголовок
    plot_title = title or "Сходимость методов оптимизации"
    if metric == 'both':
        fig.suptitle(plot_title, fontsize=16, fontweight='bold', y=1.02)
    else:
        ax.set_title(plot_title, fontsize=15, fontweight='bold', pad=10)
    
    # Легенда (только на первом подплоте)
    if results:
        if metric == 'both':
            fig.axes[0].legend(loc='best', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
        else:
            ax.legend(loc='best', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
    
    # Фон фигуры
    fig.patch.set_facecolor('white')
    
    # ✅ Адаптивные отступы для масштабирования
    plt.tight_layout()
    if metric == 'both':
        plt.subplots_adjust(top=0.88)
    
    # Сохранение
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 График сходимости сохранён: {save_path}")
    
    # Показ (только для CLI режима)
    if show and created_figure:
        plt.show()
        plt.close(fig)
    
    return fig, ax