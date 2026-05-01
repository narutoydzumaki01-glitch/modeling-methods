from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


GRAPHICS_DIR = Path(__file__).resolve().parent / "graphics"


def density(y: float) -> float:
    """
    Плотность вероятности p(y).

    p(y) = 0.5, если 0 <= y <= 1
    p(y) = -0.25y + 0.75, если 1 < y <= 3
    p(y) = 0 иначе
    """
    if 0 <= y <= 1:
        return 0.5
    if 1 < y <= 3:
        return -0.25 * y + 0.75
    return 0.0


def distribution_function(y: float) -> float:
    """
    Теоретическая функция распределения F(y).

    F(y) = P(X <= y)
    """
    if y < 0:
        return 0.0

    if 0 <= y <= 1:
        return 0.5 * y

    if 1 < y <= 3:
        return -0.125 * y * y + 0.75 * y - 0.125

    return 1.0


def inverse_distribution_function(k: float) -> float:
    """
    Обратная функция распределения.

    Если 0 <= k <= 0.5:
        F(y) = 0.5y
        y = 2k

    Если 0.5 < k <= 1:
        F(y) = -0.125y^2 + 0.75y - 0.125
        y = 3 - sqrt(8 - 8k)
    """
    if not 0 <= k <= 1:
        raise ValueError("k должно быть в диапазоне [0, 1]")

    if k <= 0.5:
        return 2 * k

    return 3 - math.sqrt(8 - 8 * k)


def generate_by_inverse_method(n: int) -> list[float]:
    """
    Генерация выборки методом обратной функции.
    """
    sample = []

    for _ in range(n):
        k = random.random()
        y = inverse_distribution_function(k)
        sample.append(y)

    return sample


def generate_by_rejection_method(n: int) -> list[float]:
    """
    Генерация выборки методом исключения.

    Область по y: [0, 3]
    Максимум плотности M = 0.5
    """
    sample = []

    a = 0.0
    b = 3.0
    m = 0.5

    while len(sample) < n:
        y = random.uniform(a, b)
        z = random.random()

        if z < density(y) / m:
            sample.append(y)

    return sample


def empirical_distribution_function(sample: list[float], x: float) -> float:
    """
    Выборочная функция распределения F_n(x).

    F_n(x) = количество элементов выборки <= x / n
    """
    count = sum(1 for value in sample if value <= x)
    return count / len(sample)


def kolmogorov_statistic(sample: list[float], theoretical_f: Callable[[float], float]) -> tuple[float, float]:
    """
    Вычисление статистики Колмогорова.

    D_n = sup |F_n(x) - F(x)|
    K_n = sqrt(n) * D_n
    """
    sorted_sample = sorted(sample)
    n = len(sorted_sample)

    d_plus = 0.0
    d_minus = 0.0

    for i, x in enumerate(sorted_sample, start=1):
        theoretical_value = theoretical_f(x)

        d_plus = max(d_plus, i / n - theoretical_value)
        d_minus = max(d_minus, theoretical_value - (i - 1) / n)

    d_n = max(d_plus, d_minus)
    k_n = math.sqrt(n) * d_n

    return d_n, k_n


def plot_density() -> None:
    """
    Построение графика теоретической плотности.
    """
    xs = [i / 1000 * 3 for i in range(1001)]
    ys = [density(x) for x in xs]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, label="Теоретическая плотность p(y)")
    plt.title("График плотности вероятности")
    plt.xlabel("y")
    plt.ylabel("p(y)")
    plt.grid(True)
    plt.legend()
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(GRAPHICS_DIR / "density.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram(sample: list[float], title: str) -> None:
    """
    Построение гистограммы выборки.
    """
    xs = [i / 1000 * 3 for i in range(1001)]
    ys = [density(x) for x in xs]

    plt.figure(figsize=(8, 5))
    plt.hist(sample, bins=30, density=True, alpha=0.6, label="Гистограмма выборки")
    plt.plot(xs, ys, linewidth=2, label="Теоретическая плотность p(y)")
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("Плотность")
    plt.grid(True)
    plt.legend()
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    filename = "histogram_inverse.png" if "обратной функции" in title else "histogram_rejection.png"
    plt.savefig(GRAPHICS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_empirical_distribution(sample: list[float], title: str) -> None:
    """
    Построение выборочной функции распределения и сравнение с теоретической.
    """
    sorted_sample = sorted(sample)
    n = len(sorted_sample)

    empirical_y = [(i + 1) / n for i in range(n)]
    theoretical_y = [distribution_function(x) for x in sorted_sample]

    plt.figure(figsize=(8, 5))
    plt.step(sorted_sample, empirical_y, where="post", label="Выборочная функция F_n(y)")
    plt.plot(sorted_sample, theoretical_y, linewidth=2, label="Теоретическая функция F(y)")
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("F(y)")
    plt.grid(True)
    plt.legend()
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    filename = "cdf_inverse.png" if "обратной функции" in title else "cdf_rejection.png"
    plt.savefig(GRAPHICS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def print_kolmogorov_result(sample: list[float], method_name: str, alpha: float = 0.05) -> None:
    """
    Печать результата проверки по критерию Колмогорова.

    Для alpha = 0.05 используется асимптотическое критическое значение 1.36.
    """
    d_n, k_n = kolmogorov_statistic(sample, distribution_function)

    critical_value = 1.36

    print(f"\nМетод: {method_name}")
    print(f"Размер выборки n = {len(sample)}")
    print(f"D_n = {d_n:.6f}")
    print(f"K_n = sqrt(n) * D_n = {k_n:.6f}")
    print(f"Критическое значение K_кр при alpha = {alpha}: {critical_value}")

    if k_n <= critical_value:
        print("Вывод: гипотеза о соответствии теоретическому распределению НЕ отвергается.")
    else:
        print("Вывод: гипотеза о соответствии теоретическому распределению отвергается.")


def main() -> None:
    n = 1000

    inverse_sample = generate_by_inverse_method(n)
    rejection_sample = generate_by_rejection_method(n)

    plot_density()

    plot_histogram(
        inverse_sample,
        "Гистограмма выборки: метод обратной функции",
    )

    plot_empirical_distribution(
        inverse_sample,
        "Выборочная функция распределения: метод обратной функции",
    )

    print_kolmogorov_result(
        inverse_sample,
        "Метод обратной функции",
    )

    plot_histogram(
        rejection_sample,
        "Гистограмма выборки: метод исключения",
    )

    plot_empirical_distribution(
        rejection_sample,
        "Выборочная функция распределения: метод исключения",
    )

    print_kolmogorov_result(
        rejection_sample,
        "Метод исключения",
    )


if __name__ == "__main__":
    main()
