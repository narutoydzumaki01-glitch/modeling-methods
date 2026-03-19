import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Гравитационная постоянная Земли (м^3/с^2)
MU = 3.98603e14

# Абстрактный класс динамической модели
class TDynamicModel(ABC):
    @abstractmethod
    def Funcs(self, t, state):
        """
        Вычисляет правые части системы ОДУ.
        :param t: время
        :param state: вектор состояния [x, y, z, vx, vy, vz]
        :return: массив правых частей (dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt)
        """
        pass

# Модель движения космического аппарата в центральном гравитационном поле Земли
class TSpaceCraft(TDynamicModel):
    def Funcs(self, t, state):
        x, y, z, vx, vy, vz = state
        r = np.sqrt(x**2 + y**2 + z**2)
        # Если r слишком мал, выдаем предупреждение или корректируем значение
        if r < 1e-6:
            print("Предупреждение: расстояние r слишком мало, корректировка для избежания деления на ноль.")
            r = 1e-6
        ax = -MU * x / r**3
        ay = -MU * y / r**3
        az = -MU * z / r**3
        return np.array([vx, vy, vz, ax, ay, az])

# Абстрактный класс интегратора
class TAbstractIntegrator(ABC):
    def __init__(self, model, t0, tk, h):
        """
        :param model: объект динамической модели (TSpaceCraft)
        :param t0: начальное время интегрирования
        :param tk: конечное время интегрирования
        :param h: шаг интегрирования
        """
        self.model = model
        self.t0 = t0
        self.tk = tk
        self.h = h

    @abstractmethod
    def one_step(self, t, state):
        """
        Производит один шаг интегрирования.
        :param t: текущее время
        :param state: текущее состояние
        :return: новое состояние после шага
        """
        pass

    @abstractmethod
    def move_to(self, state):
        """
        Выполняет интегрирование до времени tk.
        :param state: начальное состояние
        :return: массив времен и массив состояний
        """
        pass

# Интегратор методом Эйлера
class EulerIntegrator(TAbstractIntegrator):
    def one_step(self, t, state):
        return state + self.h * self.model.Funcs(t, state)

    def move_to(self, state):
        t = self.t0
        times = [t]
        states = [state]
        while t < self.tk:
            state = self.one_step(t, state)
            t += self.h
            times.append(t)
            states.append(state)
        return np.array(times), np.array(states)

# Интегратор методом Рунге–Кутты 4-го порядка
class RungeKutta4Integrator(TAbstractIntegrator):
    def one_step(self, t, state):
        k1 = self.h * self.model.Funcs(t, state)
        k2 = self.h * self.model.Funcs(t + self.h/2, state + k1/2)
        k3 = self.h * self.model.Funcs(t + self.h/2, state + k2/2)
        k4 = self.h * self.model.Funcs(t + self.h, state + k3)
        return state + (k1 + 2*k2 + 2*k3 + k4) / 6

    def move_to(self, state):
        t = self.t0
        times = [t]
        states = [state]
        while t < self.tk:
            state = self.one_step(t, state)
            t += self.h
            times.append(t)
            states.append(state)
        return np.array(times), np.array(states)

# Функция для ввода параметров пользователем
def get_user_input():
    try:
        print("Введите начальные координаты (x, y, z) в метрах:")
        x0 = float(input("x0: "))
        y0 = float(input("y0: "))
        z0 = float(input("z0: "))
        print("Введите начальные скорости (vx, vy, vz) в м/с:")
        vx0 = float(input("vx0: "))
        vy0 = float(input("vy0: "))
        vz0 = float(input("vz0: "))
    except ValueError:
        print("Неверный ввод, используются значения по умолчанию.")
        x0, y0, z0 = 7e6, 0, 0
        vx0, vy0, vz0 = 0, 7.12e3, 0

    try:
        t0 = float(input("Начальное время t0 (с): "))
    except ValueError:
        t0 = 0.0

    try:
        tk = float(input("Конечное время tk (с): "))
    except ValueError:
        tk = 5400.0

    try:
        h = float(input("Шаг интегрирования h (с): "))
    except ValueError:
        h = 10.0

    print("Выберите метод интегрирования:")
    print("1 - Эйлера")
    print("2 - Рунге-Кутты 4-го порядка")
    method = input("Ваш выбор (1/2): ")
    if method not in ('1', '2'):
        print("Выбран метод по умолчанию: Рунге-Кутты 4-го порядка")
        method = '2'

    initial_state = np.array([x0, y0, z0, vx0, vy0, vz0])
    return initial_state, t0, tk, h, method

# Функция для построения графиков
def plot_results(times, states):
    # Распаковка состояний
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    vx = states[:, 3]
    vy = states[:, 4]
    vz = states[:, 5]

    plt.figure(figsize=(14, 10))

    # Эволюция координат
    plt.subplot(2, 2, 1)
    plt.plot(times, x, label='x')
    plt.plot(times, y, label='y')
    plt.plot(times, z, label='z')
    plt.xlabel("Время (с)")
    plt.ylabel("Координаты (м)")
    plt.title("Эволюция координат")
    plt.legend()
    plt.grid()

    # Эволюция скоростей
    plt.subplot(2, 2, 2)
    plt.plot(times, vx, label='vx')
    plt.plot(times, vy, label='vy')
    plt.plot(times, vz, label='vz')
    plt.xlabel("Время (с)")
    plt.ylabel("Скорости (м/с)")
    plt.title("Эволюция скоростей")
    plt.legend()
    plt.grid()

    # Проекция траектории XY
    plt.subplot(2, 2, 3)
    plt.plot(x, y)
    plt.xlabel("x (м)")
    plt.ylabel("y (м)")
    plt.title("Траектория в плоскости XY")
    plt.grid()

    # Проекции траектории YZ и XZ
    plt.subplot(2, 2, 4)
    plt.plot(y, z, label='YZ')
    plt.plot(x, z, label='XZ')
    plt.xlabel("Координаты (м)")
    plt.ylabel("Координата z (м)")
    plt.title("Траектории в плоскостях YZ и XZ")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_results_3d(times, states):
    # Распаковка состояний
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Построение траектории движения КА
    ax.plot(x, y, z, label='3D траектория', color='red', linewidth=2)

    # Параметры Земли
    earth_radius = 6.371e6  # Радиус Земли в метрах
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # Параметры для сферы
    X = earth_radius * np.outer(np.cos(u), np.sin(v))
    Y = earth_radius * np.outer(np.sin(u), np.sin(v))
    Z = earth_radius * np.outer(np.ones_like(u), np.cos(v))
    
    # Отображение сферы Земли с прозрачностью
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.3, rstride=4, cstride=4, linewidth=0)
    
    # Оформление графика
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_zlabel("Z (м)")
    ax.set_title("3D траектория движения КА с изображением Земли")
    ax.legend()
    
    # Опционально: задать равные масштабы по осям для корректного отображения сферы
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

def main():
    initial_state, t0, tk, h, method = get_user_input()

    # Создаём модель движения КА
    model = TSpaceCraft()

    # Выбор интегратора
    if method == '1':
        integrator = EulerIntegrator(model, t0, tk, h)
    else:
        integrator = RungeKutta4Integrator(model, t0, tk, h)

    times, states = integrator.move_to(initial_state)

    # Вывод графиков результатов
    plot_results(times, states)
    plot_results_3d(times, states)

if __name__ == '__main__':
    main()
