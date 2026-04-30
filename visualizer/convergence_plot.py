"""
График сходимости методов оптимизации
"""

from matplotlib.figure import Figure
import numpy as np
import matplotlib
# ✅ Принудительно используем интерактивный бэкенд
matplotlib.use('TkAgg')
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
        # Очищаем все оси фигуры
        for existing_ax in fig.axes:
            existing_ax.clear()
        ax = fig.axes[0] if len(fig.axes) > 0 else fig.add_subplot(111)
    
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
    
    # ✅ Определяем метрики и соответствующие ключи в history
    if metric == 'both':
        metrics_config = [
            {'key': 'f', 'ylabel': r'$f(x^k)$', 'title': 'Значение функции'},
            {'key': 'grad_norm', 'ylabel': r'$\|\nabla f(x^k)\|$', 'title': 'Норма градиента'}
        ]
        # Пересоздаём подплоты
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        axes_list = [ax1, ax2]
    elif metric == 'f':
        metrics_config = [
            {'key': 'f', 'ylabel': r'$f(x^k)$', 'title': None}
        ]
        axes_list = [ax]
    elif metric == 'grad':
        metrics_config = [
            {'key': 'grad_norm', 'ylabel': r'$\|\nabla f(x^k)\|$', 'title': None}
        ]
        axes_list = [ax]
    else:
        raise ValueError(f"Неизвестная метрика: {metric}")
    
    has_any_legend = False
    
    for cfg, current_ax in zip(metrics_config, axes_list):
        history_key = cfg['key']
        ylabel = cfg['ylabel']
        subplot_title = cfg['title']
        
        # Подписи осей
        current_ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        current_ax.set_xlabel('Итерация k', fontsize=13, fontweight='bold')
        
        # Логарифмическая шкала
        if log_scale:
            current_ax.set_yscale('log')
        
        # Построение кривых для каждого метода
        for idx, result in enumerate(results):
            # Проверка наличия ключей
            if 'k' not in result.history or history_key not in result.history:
                continue
            
            # Безопасная конвертация в numpy массивы
            try:
                k = np.asarray(result.history['k'], dtype=np.float64).ravel()
                y_data = np.asarray(result.history[history_key], dtype=np.float64).ravel()
            except Exception:
                continue

            if len(k) == 0 or len(y_data) == 0:
                continue
            
            # Выравнивание длин
            min_len = min(len(k), len(y_data))
            if min_len == 0:
                continue
            k = k[:min_len]
            y_data = y_data[:min_len]
            
            # Фильтрация невалидных значений
            valid = np.isfinite(y_data) & np.isfinite(k)
            k_valid = k[valid]
            y_valid = y_data[valid]
            
            if len(k_valid) == 0 or len(y_valid) == 0:
                continue
            
            # Фильтрация для log-шкалы
            if log_scale:
                positive = y_valid > 0
                k_valid = k_valid[positive]
                y_valid = y_valid[positive]
                if len(k_valid) == 0:
                    continue
            
            color = colors[idx % len(colors)]
            line_style = line_styles[idx % len(line_styles)]
            
            # markevery
            n_points = len(k_valid)
            if n_points > 1:
                markevery = max(1, n_points // 15)
            else:
                markevery = None
            
            try:
                # Явное преобразование в 1D массивы
                x_plot = np.asarray(k_valid, dtype=np.float64).ravel()
                y_plot = np.asarray(y_valid, dtype=np.float64).ravel()
                
                # Основная линия
                if markevery is not None and n_points > 1:
                    current_ax.plot(
                        x_plot, y_plot,
                        line_style, color=color, linewidth=2.5,
                        label=f"{result.method_name} ({result.n_iterations} итер.)",
                        alpha=0.9, marker='o', markersize=4,
                        markevery=markevery,
                        markerfacecolor=color, markeredgecolor='black',
                        markeredgewidth=1.0
                    )
                else:
                    current_ax.plot(
                        x_plot, y_plot,
                        line_style, color=color, linewidth=2.5,
                        label=f"{result.method_name} ({result.n_iterations} итер.)",
                        alpha=0.9
                    )
                
                has_any_legend = True
                
                # Точка сходимости
                if result.converged and len(x_plot) > 0:
                    current_ax.plot(
                        x_plot[-1], y_plot[-1],
                        'o', color=color, markersize=10,
                        markeredgecolor='black', markeredgewidth=2, zorder=10
                    )
            except Exception as e:
                print(f"❌ Ошибка построения графика сходимости для {result.method_name}: {e}")
                continue
        
        # Сетка и оформление
        current_ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0,
                       which='both' if log_scale else 'major')
        current_ax.set_facecolor('#fafafa')
        
        # Заголовок подплота
        if subplot_title is not None:
            current_ax.set_title(subplot_title, fontsize=14, fontweight='bold', pad=10)
    
    # Общий заголовок
    plot_title = title or "Сходимость методов оптимизации"
    if metric == 'both':
        fig.suptitle(plot_title, fontsize=16, fontweight='bold', y=1.02)
    else:
        ax.set_title(plot_title, fontsize=15, fontweight='bold', pad=10)
    
    # Легенда
    if has_any_legend:
        if metric == 'both':
            first_ax = axes_list[0]
            first_ax.legend(loc='best', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
        else:
            ax.legend(loc='best', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
    
    # Фон фигуры
    fig.patch.set_facecolor('white')
    
    # Адаптивные отступы
    plt.tight_layout()
    if metric == 'both':
        plt.subplots_adjust(top=0.88)
    
    # Включаем интерактивность для GUI
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
        print(f"📊 График сходимости сохранён: {save_path}")
    
    # Показ (только для CLI режима)
    if show and created_figure:
        plt.show()
    
    return fig, ax