import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading

# Импортируем необходимые функции и классы из первого файла (simulation.py)
from method_pz1 import TSpaceCraft, EulerIntegrator, RungeKutta4Integrator, plot_results, plot_results_3d

def run_simulation():
    try:
        # Чтение значений из полей ввода
        x0 = float(entry_x0.get())
        y0 = float(entry_y0.get())
        z0 = float(entry_z0.get())
        vx0 = float(entry_vx0.get())
        vy0 = float(entry_vy0.get())
        vz0 = float(entry_vz0.get())
        t0 = float(entry_t0.get())
        tk = float(entry_tk.get())
        h = float(entry_h.get())
    except ValueError:
        messagebox.showerror("Ошибка", "Проверьте корректность введённых числовых значений!")
        return

    # Формирование вектора начального состояния
    initial_state = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Создание модели движения КА
    model = TSpaceCraft()

    # Выбор метода интегрирования
    method = integrator_method.get()
    if method == "Эйлера":
        integrator = EulerIntegrator(model, t0, tk, h)
    else:
        integrator = RungeKutta4Integrator(model, t0, tk, h)

    # Функция, выполняющая интегрирование
    def simulation_thread():
        times, states = integrator.move_to(initial_state)
        # Планируем вызов функции построения графиков в главном потоке
        root.after(0, lambda: plot_results(times, states))
        # Если нужно строить 3D график, можно вместо этого вызвать:
        # root.after(0, lambda: plot_results_3d(times, states))

    # Запуск интегрирования в отдельном потоке
    threading.Thread(target=simulation_thread, daemon=True).start()

    
# Создание главного окна Tkinter
root = tk.Tk()
root.title("Симуляция движения космического аппарата")

# Фрейм для ввода параметров
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Начальные координаты
ttk.Label(frame, text="Начальные координаты (м):").grid(row=0, column=0, columnspan=2, sticky=tk.W)
ttk.Label(frame, text="x0:").grid(row=1, column=0, sticky=tk.W)
entry_x0 = ttk.Entry(frame)
entry_x0.insert(0, "6878000")
entry_x0.grid(row=1, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="y0:").grid(row=2, column=0, sticky=tk.W)
entry_y0 = ttk.Entry(frame)
entry_y0.insert(0, "0")
entry_y0.grid(row=2, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="z0:").grid(row=3, column=0, sticky=tk.W)
entry_z0 = ttk.Entry(frame)
entry_z0.insert(0, "0")
entry_z0.grid(row=3, column=1, sticky=(tk.W, tk.E))

# Начальные скорости
ttk.Label(frame, text="Начальные скорости (м/с):").grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
ttk.Label(frame, text="vx0:").grid(row=5, column=0, sticky=tk.W)
entry_vx0 = ttk.Entry(frame)
entry_vx0.insert(0, "0")
entry_vx0.grid(row=5, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="vy0:").grid(row=6, column=0, sticky=tk.W)
entry_vy0 = ttk.Entry(frame)
entry_vy0.insert(0, "8000")
entry_vy0.grid(row=6, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="vz0:").grid(row=7, column=0, sticky=tk.W)
entry_vz0 = ttk.Entry(frame)
entry_vz0.insert(0, "0")
entry_vz0.grid(row=7, column=1, sticky=(tk.W, tk.E))

# Параметры интегрирования
ttk.Label(frame, text="Параметры интегрирования:").grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
ttk.Label(frame, text="Начальное время t0 (с):").grid(row=9, column=0, sticky=tk.W)
entry_t0 = ttk.Entry(frame)
entry_t0.insert(0, "0.0")
entry_t0.grid(row=9, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Конечное время tk (с):").grid(row=10, column=0, sticky=tk.W)
entry_tk = ttk.Entry(frame)
entry_tk.insert(0, "5400.0")
entry_tk.grid(row=10, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Шаг интегрирования h (с):").grid(row=11, column=0, sticky=tk.W)
entry_h = ttk.Entry(frame)
entry_h.insert(0, "10.0")
entry_h.grid(row=11, column=1, sticky=(tk.W, tk.E))

# Выбор метода интегрирования
ttk.Label(frame, text="Метод интегрирования:").grid(row=12, column=0, sticky=tk.W, pady=(10, 0))
integrator_method = tk.StringVar(value="Рунге-Кутты 4-го порядка")
methods = ["Эйлера", "Рунге-Кутты 4-го порядка"]
combo_method = ttk.Combobox(frame, textvariable=integrator_method, values=methods, state="readonly")
combo_method.grid(row=12, column=1, sticky=(tk.W, tk.E))

# Кнопка запуска симуляции
button_run = ttk.Button(frame, text="Запустить симуляцию", command=run_simulation)
button_run.grid(row=13, column=0, columnspan=2, pady=(15, 0))

# Настройка расширения столбцов
frame.columnconfigure(1, weight=1)


if __name__ == "__main__":
    root.mainloop()
