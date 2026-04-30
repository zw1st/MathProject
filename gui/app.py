"""
Главное окно приложения для многомерной оптимизации
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime
from typing import List, Optional, Dict

from gui.plot_window import PlotWindow
from methods.bfgs import BFGS
from models.optimization_result import OptimizationResult
from functions.test_functions import TestFunctions
from methods.fletcher_reeves import FletcherReeves
from methods.polak_ribiere import PolakRibiere
from methods.newton import NewtonMethod

from methods.lbfgs import LBFGS


class OptimizationApp:
    """
    Главное приложение для запуска оптимизации.
    
    Компоненты:
    - Выбор тестовой функции
    - Гиперпараметры методов (ε, итерации, x⁰, рестарт)
    - Выбор методов для запуска (чекбоксы)
    - Таблица сводных результатов
    - Кнопки: запуск, графики, экспорт
    - Статус бар
    """
    
    def __init__(self, root: tk.Tk):
        """
        Инициализация приложения.
        
        Параметры:
        - root: главное окно tkinter
        """
        self.root = root
        self.root.title("Многомерная оптимизация: FR, PR, Newton, BFGS, L-BFGS")
        self.root.geometry("1300x800")
        self.root.minsize(1100, 700)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        
        # ==================== ДАННЫЕ ====================
        self.test_functions = TestFunctions()
        self.available_functions = [
            "Химмельблау", "Розенброк", "Бут", "Бил",
            "Матьяс", "Трёхгорбый верблюд", "Сфера",
            "Растригин", "Полином 10D"
        ]
        self.function_map = {
            "Химмельблау": "himmelblau",
            "Розенброк": "rosenbrock",
            "Бут": "booth",
            "Бил": "beale",
            "Матьяс": "matyas",
            "Трёхгорбый верблюд": "three_hump_camel",
            "Сфера": "sphere",
            "Растригин": "rastrigin",
            "Полином 10D": "polynomial_10d"
        }
        self.polynomial_10d = None
        
        # Результаты оптимизации
        self.results: List[OptimizationResult] = []
        self.current_func = None
        self.plot_window: Optional[PlotWindow] = None
        
        # ==================== ИНТЕРФЕЙС ====================
        self._create_menu()
        self._create_main_frame()
        self._create_function_panel()
        self._create_hyperparams_panel()
        self._create_methods_panel()
        self._create_buttons_panel()
        self._create_results_table()
        self._create_status_bar()
        
        # Загрузка значений по умолчанию
        self._load_default_x0()

    def _on_close(self):
        """Обработчик закрытия приложения"""
        # Закрыть окно графиков если открыто
        if self.plot_window and hasattr(self.plot_window, 'window'):
            try:
                self.plot_window.window.destroy()
            except:
                pass
        
        # Закрыть все matplotlib фигуры
        plt.close('all')
        
        # Уничтожить главное окно
        self.root.destroy()
        
        # Принудительная очистка памяти
        import gc
        gc.collect()
    
    # ========== МЕНЮ ==========
    
    def _create_menu(self):
        """Создание верхнего меню (Файл, Справка)"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Экспорт результатов", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Меню Справка
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self._show_about)
    
    # ========== ОСНОВНОЙ ФРЕЙМ ==========
    
    def _create_main_frame(self):
        """Создание основного контейнера"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)  # Таблица результатов растягивается
        
        self.main_frame = main_frame
    
    # ========== ВЫБОР ФУНКЦИИ ==========
    
    def _create_function_panel(self):
        """Панель выбора тестовой функции и начальной точки"""
        frame = ttk.LabelFrame(self.main_frame, text="1. Тестовая функция", padding="10")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        
        # Выбор функции
        ttk.Label(frame, text="Функция:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.func_var = tk.StringVar(value=self.available_functions[0])
        self.func_combo = ttk.Combobox(
            frame, textvariable=self.func_var,
            values=self.available_functions, width=35, state="readonly"
        )
        self.func_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.func_combo.bind('<<ComboboxSelected>>', self._on_function_change)
        
        # Начальная точка x⁰
        ttk.Label(frame, text="Начальная точка x⁰:").grid(row=0, column=2, sticky="w", padx=20, pady=5)
        self.x0_entry = ttk.Entry(frame, width=40)
        self.x0_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        ttk.Button(frame, text="По умолчанию", command=self._load_default_x0).grid(
            row=0, column=4, padx=10, pady=5
        )
    
    # ========== ГИПЕРПАРАМЕТРЫ ==========
    
    def _create_hyperparams_panel(self):
        """Панель гиперпараметров оптимизации"""
        frame = ttk.LabelFrame(self.main_frame, text="2. Гиперпараметры", padding="10")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        
        # Точность ε
        ttk.Label(frame, text="Точность ε:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.epsilon_var = tk.StringVar(value="0.001")
        self.epsilon_entry = ttk.Entry(frame, textvariable=self.epsilon_var, width=15)
        self.epsilon_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Макс. итераций
        ttk.Label(frame, text="Макс. итераций:").grid(row=0, column=2, sticky="w", padx=20, pady=5)
        self.max_iter_var = tk.StringVar(value="500")
        self.max_iter_entry = ttk.Entry(frame, textvariable=self.max_iter_var, width=15)
        self.max_iter_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Частота рестарта (для FR/PR)
        ttk.Label(frame, text="Рестарт (FR/PR):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.restart_var = tk.StringVar(value="auto")
        self.restart_entry = ttk.Entry(frame, textvariable=self.restart_var, width=15)
        self.restart_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(frame, text="(или 'auto' = n)").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        
        # Вывод в консоль
        self.verbose_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame, text="📄 Вывод итераций в консоль (+ экспорт в файл)",
            variable=self.verbose_var
        ).grid(row=1, column=3, sticky="w", padx=20, pady=5)
    
    # ========== ВЫБОР МЕТОДОВ ==========
    
    def _create_methods_panel(self):
        """Панель выбора методов для запуска (чекбоксы)"""
        frame = ttk.LabelFrame(self.main_frame, text="3. Методы оптимизации", padding="10")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        # Чекбоксы для каждого метода
        self.method_vars = {}
        methods = [
            ("Флетчер-Ривз", "fr", True),
            ("Полак-Рибьер", "pr", True),
            ("Ньютон", "newton", True),
            ("BFGS", "bfgs", True),
            ("L-BFGS", "lbfgs", False),  # По умолчанию выключен
        ]
        
        for i, (name, key, default) in enumerate(methods):
            var = tk.BooleanVar(value=default)
            self.method_vars[key] = var
            ttk.Checkbutton(frame, text=name, variable=var).grid(
                row=0, column=i, padx=15, pady=5, sticky="w"
            )
    
    # ========== КНОПКИ УПРАВЛЕНИЯ ==========
    
    def _create_buttons_panel(self):
        """Панель кнопок управления"""
        frame = ttk.Frame(self.main_frame)
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        
        self.run_btn = ttk.Button(
            frame, text="▶ Запустить оптимизацию",
            command=self._run_optimization
        )
        self.run_btn.grid(row=0, column=0, padx=10)
        
        self.plot_btn = ttk.Button(
            frame, text="📊 Показать графики",
            command=self._show_plots,
            state="disabled"  # Неактивна до запуска
        )
        self.plot_btn.grid(row=0, column=1, padx=10)
        
        self.export_btn = ttk.Button(
            frame, text="💾 Экспорт результатов",
            command=self._export_results,
            state="disabled"
        )
        self.export_btn.grid(row=0, column=2, padx=10)
        
        self.clear_btn = ttk.Button(
            frame, text="🗑 Очистить",
            command=self._clear_results
        )
        self.clear_btn.grid(row=0, column=3, padx=10)
    
    # ========== ТАБЛИЦА РЕЗУЛЬТАТОВ ==========
    
    def _create_results_table(self):
        """Таблица сводных результатов оптимизации"""
        frame = ttk.LabelFrame(self.main_frame, text="4. Результаты", padding="10")
        frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Колонки таблицы
        columns = (
            "Метод", "Итераций", "f(x*)", "||grad||",
            "Сходимость", "Вызовов f", "Вызовов ∇f", "x₁", "x₂"
        )
        
        self.results_tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=8
        )
        
        # Настройка колонок
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=90, anchor="center")
        
        self.results_tree.column("Метод", width=120)
        self.results_tree.column("f(x*)", width=130)
        self.results_tree.column("||grad||", width=110)
        
        # Скроллбары
        v_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.results_tree.yview)
        h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
    
    # ========== СТАТУС БАР ==========
    
    def _create_status_bar(self):
        """Строка состояния внизу окна"""
        self.status_var = tk.StringVar(value="✓ Готов к работе")
        status_label = ttk.Label(
            self.main_frame, textvariable=self.status_var,
            foreground="green", relief="sunken", anchor="w"
        )
        status_label.grid(row=5, column=0, sticky="ew", pady=(5, 0))
    
    # ========== ОБРАБОТЧИКИ СОБЫТИЙ ==========
    
    def _on_function_change(self, event=None):
        """Обработка смены функции в dropdown"""
        self._load_default_x0()
        self._clear_results()
        self.status_var.set(f"✓ Выбрана функция: {self.func_var.get()}")
    
    def _load_default_x0(self):
        """Загрузка начальной точки для выбранной функции"""
        func_name = self.function_map.get(self.func_var.get())
        if func_name:
            try:
                test_func = self.test_functions.get_by_name(func_name)
                x0_str = ", ".join([f"{x:.2f}" for x in test_func.x0_default])
                self.x0_entry.delete(0, tk.END)
                self.x0_entry.insert(0, x0_str)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить x⁰: {e}")
    
    def _parse_x0(self) -> Optional[np.ndarray]:
        """Парсинг начальной точки из текстового поля"""
        try:
            x0_str = self.x0_entry.get()
            x0 = np.array([float(x.strip()) for x in x0_str.split(',')])
            return x0
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверный формат x⁰: {e}")
            return None
    
    def _parse_hyperparams(self) -> Optional[Dict]:
        """Парсинг гиперпараметров из полей ввода"""
        try:
            params = {
                'epsilon': float(self.epsilon_var.get()),
                'max_iterations': int(self.max_iter_var.get()),
            }
            
            restart_str = self.restart_var.get().strip().lower()
            params['restart_frequency'] = None if restart_str == 'auto' else int(restart_str)
            
            return params
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверный формат параметров: {e}")
            return None
    
    def _get_selected_methods(self) -> List[str]:
        """Получение списка выбранных методов"""
        selected = []
        method_names = {
            'fr': 'Флетчер-Ривз',
            'pr': 'Полак-Рибьер',
            'newton': 'Ньютон',
            'bfgs': 'BFGS',
            'lbfgs': 'L-BFGS'
        }
        for key, var in self.method_vars.items():
            if var.get():
                selected.append(key)
        return selected
    
    # ========== ЗАПУСК ОПТИМИЗАЦИИ ==========
    
    def _run_optimization(self):
        """Запуск выбранных методов оптимизации"""
        # Парсинг входных данных
        x0 = self._parse_x0()
        if x0 is None:
            return
        
        params = self._parse_hyperparams()
        if params is None:
            return
        
        selected_methods = self._get_selected_methods()
        if not selected_methods:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один метод!")
            return
        
        # Загрузка функции
        func_name = self.function_map.get(self.func_var.get())
        if not func_name:
            messagebox.showerror("Ошибка", "Функция не найдена")
            return
        
        try:
            self.status_var.set("⏳ Запуск оптимизации...")
            self.root.update()
            
            test_func = self.test_functions.get_by_name(func_name)
            self.current_func = test_func
            self.results = []
            
            verbose = self.verbose_var.get()
            export_file = None
            
            # Если verbose включён — создаём файл для экспорта
            if verbose:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_file = f"results/optimization_{func_name}_{timestamp}.txt"
                os.makedirs("results", exist_ok=True)
                print(f"\n{'='*80}")
                print(f"ЭКСПОРТ ИТЕРАЦИЙ: {export_file}")
                print(f"{'='*80}\n")
            
            # Запуск каждого выбранного метода
            for method_key in selected_methods:
                result = self._run_single_method(
                    method_key, test_func, x0, params, verbose, export_file
                )
                if result:
                    self.results.append(result)
            
            # Обновление интерфейса
            self._update_results_table()
            self.plot_btn.config(state="normal")
            self.export_btn.config(state="normal")
            
            self.status_var.set(f"✓ Завершено! Найдено минимумов: {len(self.results)}")
            messagebox.showinfo(
                "Готово",
                f"Оптимизация завершена!\n\n"
                f"Методов выполнено: {len(self.results)}\n"
                f"Функция: {test_func.name}\n\n"
                f"Нажмите 'Показать графики' для визуализации."
            )
            
        except Exception as e:
            self.status_var.set("✗ Ошибка при оптимизации")
            messagebox.showerror("Ошибка", f"Ошибка при оптимизации:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _run_single_method(self, method_key: str, test_func, x0, params, verbose, export_file):
        """Запуск одного метода оптимизации"""
        method_map = {
            'fr': FletcherReeves(
                epsilon=params['epsilon'],
                max_iterations=params['max_iterations'],
                restart_frequency=params['restart_frequency'],
                verbose=verbose
            ),
            'pr': PolakRibiere(
                epsilon=params['epsilon'],
                max_iterations=params['max_iterations'],
                restart_frequency=params['restart_frequency'],
                verbose=verbose
            ),
            'newton': NewtonMethod(
                epsilon=params['epsilon'],
                max_iterations=params['max_iterations'],
                verbose=verbose
            ),
            'bfgs': BFGS(
                epsilon=params['epsilon'],
                max_iterations=params['max_iterations'],
                verbose=verbose
            ),
            'lbfgs': LBFGS(
                epsilon=params['epsilon'],
                max_iterations=params['max_iterations'],
                m=10,
                verbose=verbose
            ),
        }
        
        method = method_map.get(method_key)
        if method is None:
            return None
        
        # Для Ньютона нужен Гессиан
        hess = test_func.hess if method_key == 'newton' else None
        
        try:
            return method.optimize(test_func.func, test_func.grad, x0, hess)
        except Exception as e:
            messagebox.showerror("Ошибка метода", f"{method.method_name}:\n{str(e)}")
            return None
    
    # ========== ОБНОВЛЕНИЕ ТАБЛИЦЫ ==========
    
    def _update_results_table(self):
        """Обновление таблицы результатов"""
        # Очистка
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Добавление результатов
        for result in self.results:
            x1 = result.x[0] if len(result.x) > 0 else 0.0
            x2 = result.x[1] if len(result.x) > 1 else 0.0

            
            values = (
                result.method_name,
                result.n_iterations,
                f"{result.f_val:.10e}",
                f"{result.grad_norm:.10e}",
                "✓ Да" if result.converged else "✗ Нет",
                result.n_function_evals,
                result.n_grad_evals,
                f"{x1:.6f}",
                f"{x2:.6f}"
            )
            self.results_tree.insert("", tk.END, values=values)
    
    # ========== ГРАФИКИ ==========
    
    def _show_plots(self):
        if not self.results or not self.current_func:
            messagebox.showwarning("Предупреждение", "Сначала запустите оптимизацию!")
            return
        
        if self.plot_window and self.plot_window.window.winfo_exists():
            self.plot_window.window.destroy()
        
        # Для многомерных функций передаём начальную точку для среза
        slice_fixed = None
        if not self.current_func.is_2d and len(self.results) > 0:
            hist = self.results[0].history.get('x', [])
            slice_fixed = np.array(hist[0]) if len(hist) > 0 else None
        
        self.plot_window = PlotWindow(
            self.root, self.current_func, self.results,
            slice_fixed_values=slice_fixed
        )
    
    # ========== ЭКСПОРТ ==========
    
    def _export_results(self):
        """Экспорт результатов в файл"""
        if not self.results:
            messagebox.showwarning("Предупреждение", "Нет результатов для экспорта")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="Экспорт результатов"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Заголовок
                    f.write("=" * 80 + "\n")
                    f.write("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Функция: {self.current_func.name}\n")
                    f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Начальная точка: {self.x0_entry.get()}\n")
                    f.write(f"Точность ε: {self.epsilon_var.get()}\n")
                    f.write(f"Макс. итераций: {self.max_iter_var.get()}\n\n")
                    
                    # Таблица результатов
                    f.write("-" * 80 + "\n")
                    f.write("СВОДНАЯ ТАБЛИЦА\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Метод':<20} {'Итераций':<12} {'f(x*)':<20} {'||grad||':<18} {'Сходимость':<12}\n")
                    f.write("-" * 80 + "\n")
                    
                    for result in self.results:
                        f.write(f"{result.method_name:<20} {result.n_iterations:<12} "
                               f"{result.f_val:<20.10e} {result.grad_norm:<18.10e} "
                               f"{'✓' if result.converged else '✗':<12}\n")
                    
                    # Детали по каждому методу
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("ДЕТАЛИ ПО МЕТОДАМ\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for result in self.results:
                        f.write(f"Метод: {result.method_name}\n")
                        f.write(f"  Найденный минимум: {result.x}\n")
                        f.write(f"  f(x*) = {result.f_val:.10e}\n")
                        f.write(f"  ||grad|| = {result.grad_norm:.10e}\n")
                        f.write(f"  Итераций: {result.n_iterations}\n")
                        f.write(f"  Вызовов функции: {result.n_function_evals}\n")
                        f.write(f"  Вызовов градиента: {result.n_grad_evals}\n")
                        f.write(f"  Сходимость: {'Да' if result.converged else 'Нет'}\n\n")
                
                messagebox.showinfo("Успех", f"Результаты сохранены в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")
    
    # ========== ОЧИСТКА ==========
    
    def _clear_results(self):
        """Очистка всех результатов"""
        self.results = []
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.current_func = None
        
        if self.plot_window and self.plot_window.window.winfo_exists():
            self.plot_window.window.destroy()
        
        self.plot_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.status_var.set("✓ Результаты очищены")
    
    # ========== О ПРОГРАММЕ ==========
    
    def _show_about(self):
        """Окно 'О программе'"""
        about_text = """
        Многомерная оптимизация
        Версия 1.0
        
        Методы:
        • Флетчер-Ривз (сопряжённые градиенты)
        • Полак-Рибьер (сопряжённые градиенты)
        • Ньютон (с модификацией Гессиана)
        • BFGS (квазиньютоновский)
        • L-BFGS (с ограниченной памятью)
        
        Тестовые функции:
        • Химмельблау, Розенброк, Бут, Бил
        • Матьяс, Трёхгорбый верблюд, Сфера, Растригин
        • Полином 10-й степени (10 переменных)
        
        Требования:
        • Python 3.8+
        • numpy, matplotlib
        • jax, jaxlib (для полинома 10D)
        """
        messagebox.showinfo("О программе", about_text)


def run_gui():
    """Точка входа для запуска GUI"""
    root = tk.Tk()
    
    # Настройка стиля
    style = ttk.Style()
    style.theme_use('clam')
    
    # Шрифты
    style.configure('TLabel', font=('Arial', 10))
    style.configure('TButton', font=('Arial', 10, 'bold'))
    style.configure('TLabelframe', font=('Arial', 11, 'bold'))
    style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))
    style.configure('Treeview', font=('Arial', 10), rowheight=25)
    style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
    
    app = OptimizationApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()