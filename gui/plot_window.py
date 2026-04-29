"""
Отдельное окно для отображения графиков оптимизации
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from typing import List, Optional


from functions.test_functions import TestFunction
from models.optimization_result import OptimizationResult
from visualizer import plot_contour_with_tracks, plot_convergence


class PlotWindow:
    """
    Окно для просмотра графиков оптимизации.
    """
    
    def __init__(self, parent: tk.Tk, func: TestFunction, results: List[OptimizationResult]):
        """
        Инициализация окна графиков.
        """
        self.parent = parent
        self.func = func
        self.results = results
        
        # ✅ Проверка: есть ли данные для отображения
        if not results:
            messagebox.showwarning("Предупреждение", "Нет результатов для отображения!")
            return
        
        # Проверка: есть ли история в результатах
        has_history = any(
            len(r.history.get('k', [])) > 0 for r in results
        )
        if not has_history:
            messagebox.showwarning("Предупреждение", 
                "Результаты не содержат истории итераций!\n"
                "Убедитесь, что методы оптимизации корректно заполняют history.")
            return
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title("Графики оптимизации")
        self.window.geometry("1400x900")
        self.window.transient(parent)
        
        # ✅ Обработчик закрытия окна графиков
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Создание вкладок
        self._create_notebook()
        
        # Панель кнопок
        self._create_button_panel()
    
    def _create_notebook(self):
        """Создание вкладок с графиками"""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка 1: Контурный график
        contour_frame = ttk.Frame(notebook)
        notebook.add(contour_frame, text="🗺 Контурный график")
        self._create_contour_plot(contour_frame)
        
        # Вкладка 2: График сходимости
        conv_frame = ttk.Frame(notebook)
        notebook.add(conv_frame, text="📈 Сходимость")
        self._create_convergence_plot(conv_frame)
        
        # ✅ Переключиться на первую вкладку
        notebook.select(0)
    
    def _create_contour_plot(self, parent):
        """Создание контурного графика"""
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        try:
            # ✅ Проверка данных перед передачей
            valid_results = [
                r for r in self.results 
                if len(r.history.get('x', [])) > 0
            ]
            
            if not valid_results:
                ax.text(0.5, 0.5, "Нет данных для отображения\n(проверьте историю методов)",
                       ha='center', va='center', fontsize=14, color='red')
                ax.set_axis_off()
            else:
                # Для 10D функций передаём параметры сечения
                if not self.func.is_2d:
                    plot_contour_with_tracks(
                        func=self.func,
                        results=valid_results,  # ✅ Без deepcopy
                        figure=fig,
                        ax=ax,
                        show=False,
                        slice_dims=(0, 1),
                        use_x0_for_slice=True
                    )
                else:
                    plot_contour_with_tracks(
                        func=self.func,
                        results=valid_results,  # ✅ Без deepcopy
                        figure=fig,
                        ax=ax,
                        show=False
                    )
                
                # Проверка: действительно ли что-то нарисовано
                if len(ax.lines) == 0 and len(ax.collections) == 0:
                    ax.text(0.5, 0.5, "Нет данных для отображения",
                           ha='center', va='center', fontsize=14, color='red')
                    ax.set_axis_off()
            
        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f"Ошибка построения:\n{str(e)}",
                   ha='center', va='center', fontsize=14, color='red')
            ax.set_axis_off()
            import traceback
            traceback.print_exc()
        
        # Canvas для matplotlib
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Панель навигации
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack()
        
        self.contour_canvas = canvas
        self.contour_fig = fig
    
    def _create_convergence_plot(self, parent):
        """Создание графика сходимости"""
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        try:
            # ✅ Проверка данных перед передачей
            valid_results = [
                r for r in self.results 
                if len(r.history.get('k', [])) > 0 
                and len(r.history.get('f', [])) > 0
            ]
            
            if not valid_results:
                ax.text(0.5, 0.5, "Нет данных для графика сходимости",
                       ha='center', va='center', fontsize=14, color='red')
                ax.set_axis_off()
            else:
                plot_convergence(
                    results=valid_results,  # ✅ Без deepcopy
                    metric='both',
                    figure=fig,
                    ax=ax,
                    show=False
                )
                
                # Проверка: действительно ли что-то нарисовано
                if len(ax.lines) == 0:
                    ax.clear()
                    ax.text(0.5, 0.5, "Нет данных для графика сходимости\n"
                                     "(проверьте, что history['k'] и history['f'] заполнены)",
                           ha='center', va='center', fontsize=14, color='red')
                    ax.set_axis_off()
            
        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f"Ошибка построения:\n{str(e)}",
                   ha='center', va='center', fontsize=14, color='red')
            ax.set_axis_off()
            import traceback
            traceback.print_exc()
        
        # Canvas
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack()
        
        self.convergence_canvas = canvas
        self.convergence_fig = fig
    
    def _create_button_panel(self):
        """Панель кнопок для сохранения"""
        toolbar = ttk.Frame(self.window)
        toolbar.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            toolbar, text="💾 Сохранить оба графика",
            command=self._save_both
        ).pack(side='left', padx=5)
        
        ttk.Button(
            toolbar, text="🗺 Только контурный",
            command=lambda: self._save_plot('contour')
        ).pack(side='left', padx=5)
        
        ttk.Button(
            toolbar, text="📈 Только сходимость",
            command=lambda: self._save_plot('convergence')
        ).pack(side='left', padx=5)
        
        ttk.Button(
            toolbar, text="❌ Закрыть",
            command=self.window.destroy
        ).pack(side='right', padx=5)
    
    def _save_both(self):
        """Сохранение обоих графиков"""
        base_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Сохранить графики"
        )
        
        if base_path:
            import os
            base, ext = os.path.splitext(base_path)
            try:
                self.contour_canvas.figure.savefig(
                    f"{base}_contour{ext}", dpi=300, bbox_inches='tight'
                )
                self.convergence_canvas.figure.savefig(
                    f"{base}_convergence{ext}", dpi=300, bbox_inches='tight'
                )
                messagebox.showinfo(
                    "Успех",
                    f"Графики сохранены:\n{base}_contour{ext}\n{base}_convergence{ext}"
                )
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")
    
    def _save_plot(self, plot_type: str):
        """Сохранение одного графика"""
        canvas = (
            self.contour_canvas if plot_type == 'contour'
            else self.convergence_canvas
        )
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title=f"Сохранить {'контурный график' if plot_type == 'contour' else 'график сходимости'}"
        )
        
        if file_path:
            try:
                canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", f"График сохранён:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")
    
    def _on_close(self):
        """Обработчик закрытия окна графиков"""
        plt.close('all')
        self.window.destroy()
        import gc
        gc.collect()