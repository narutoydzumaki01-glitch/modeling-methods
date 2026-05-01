import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_covariance_matrix(K: np.ndarray) -> None:
    print("Проверка ковариационной матрицы K_X")
    print("K_X симметрична:", np.allclose(K, K.T))

    eigenvalues = np.linalg.eigvalsh(K)
    print("Собственные значения K_X:")
    print(eigenvalues)

    if not np.all(eigenvalues > 0):
        raise ValueError("Матрица K_X не является положительно определённой")

    print("K_X положительно определена: True")


def generate_sample(
    mean: np.ndarray,
    covariance: np.ndarray,
    sample_size: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # K_X = A A^T
    A = np.linalg.cholesky(covariance)

    # U — независимые стандартные нормальные случайные величины
    U = rng.standard_normal(size=(sample_size, len(mean)))

    X = np.einsum("ij,kj->ik", U, A) + mean

    if not np.all(np.isfinite(X)):
        raise ValueError("В выборке X есть inf или nan")

    return X, U, A


def calculate_statistics(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_mean = X.mean(axis=0)

    # rowvar=False: каждая колонка — отдельная случайная величина X_i
    sample_covariance = np.cov(X, rowvar=False, bias=False)

    return sample_mean, sample_covariance


def save_tables(
    X: np.ndarray,
    U: np.ndarray,
    A: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_covariance: np.ndarray,
    sample_mean: np.ndarray,
    sample_covariance: np.ndarray,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df_u = pd.DataFrame(U, columns=["U1", "U2", "U3", "U4"])
    df_x = pd.DataFrame(X, columns=["X1", "X2", "X3", "X4"])

    df_u.head(20).to_csv(
        os.path.join(output_dir, "first_20_U.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    df_x.head(20).to_csv(
        os.path.join(output_dir, "first_20_X.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(A).to_csv(
        os.path.join(output_dir, "matrix_A_cholesky.csv"),
        index=False,
        header=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        "component": ["X1", "X2", "X3", "X4"],
        "theoretical_mean": theoretical_mean,
        "sample_mean": sample_mean,
        "difference": sample_mean - theoretical_mean,
    }).to_csv(
        os.path.join(output_dir, "mean_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(theoretical_covariance).to_csv(
        os.path.join(output_dir, "theoretical_covariance.csv"),
        index=False,
        header=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(sample_covariance).to_csv(
        os.path.join(output_dir, "sample_covariance.csv"),
        index=False,
        header=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(sample_covariance - theoretical_covariance).to_csv(
        os.path.join(output_dir, "covariance_difference.csv"),
        index=False,
        header=False,
        encoding="utf-8-sig",
    )


def plot_scatter_pairs(X: np.ndarray, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    pairs = [
        (0, 1, "X1", "X2"),
        (0, 2, "X1", "X3"),
        (1, 2, "X2", "X3"),
        (0, 3, "X1", "X4"),
        (1, 3, "X2", "X4"),
        (2, 3, "X3", "X4"),
    ]

    for i, j, name_i, name_j in pairs:
        plt.figure(figsize=(6, 5))
        plt.scatter(X[:, i], X[:, j], s=8, alpha=0.35)
        plt.xlabel(name_i)
        plt.ylabel(name_j)
        plt.title(f"Диаграмма рассеяния {name_i} и {name_j}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"scatter_{name_i}_{name_j}.png"),
            dpi=150,
        )
        plt.close()


def print_report(
    theoretical_mean: np.ndarray,
    theoretical_covariance: np.ndarray,
    A: np.ndarray,
    sample_mean: np.ndarray,
    sample_covariance: np.ndarray,
    U: np.ndarray,
    X: np.ndarray,
) -> None:
    np.set_printoptions(precision=4, suppress=True)

    print("\nТеоретическое математическое ожидание m_X:")
    print(theoretical_mean)

    print("\nТеоретическая ковариационная матрица K_X:")
    print(theoretical_covariance)

    print("\nМатрица A из разложения Холецкого, K_X = A A^T:")
    print(A)

    print("\nПроверка A A^T:")
    print(A @ A.T)

    if not np.all(np.isfinite(U)):
        print("\nВ U есть inf или nan")
    if not np.all(np.isfinite(A)):
        print("\nВ A есть inf или nan")
    if not np.all(np.isfinite(X)):
        print("\nВ X есть inf или nan")

    print("\nВыборочное математическое ожидание:")
    print(sample_mean)

    print("\nРазность: выборочное m_X - теоретическое m_X:")
    print(sample_mean - theoretical_mean)

    print("\nВыборочная ковариационная матрица:")
    print(sample_covariance)

    print("\nРазность: выборочная K_X - теоретическая K_X:")
    print(sample_covariance - theoretical_covariance)

    print("\nПервые 10 сгенерированных векторов X:")
    print(pd.DataFrame(X[:10], columns=["X1", "X2", "X3", "X4"]))


def main() -> None:
    # -----------------------------
    # Вариант 1
    # -----------------------------

    n = 4

    m_x = np.array([1, 0, 1, 0], dtype=float)

    K_x = np.array([
        [3, 2, 1, 0],
        [2, 8, 3, 0],
        [1, 3, 4, 0],
        [0, 0, 0, 9],
    ], dtype=float)

    sample_size = 10_000
    seed = 42

    output_dir = "results_lr2"

    if len(m_x) != n:
        raise ValueError("Размерность m_X не совпадает с n")

    if K_x.shape != (n, n):
        raise ValueError("Размерность K_X не совпадает с n x n")

    check_covariance_matrix(K_x)

    X, U, A = generate_sample(
        mean=m_x,
        covariance=K_x,
        sample_size=sample_size,
        seed=seed,
    )

    sample_mean, sample_covariance = calculate_statistics(X)

    print_report(
        theoretical_mean=m_x,
        theoretical_covariance=K_x,
        A=A,
        sample_mean=sample_mean,
        sample_covariance=sample_covariance,
        U=U,
        X=X,
    )

    save_tables(
        X=X,
        U=U,
        A=A,
        theoretical_mean=m_x,
        theoretical_covariance=K_x,
        sample_mean=sample_mean,
        sample_covariance=sample_covariance,
        output_dir=output_dir,
    )

    plot_scatter_pairs(X, output_dir=output_dir)

    print(f"\nФайлы сохранены в папку: {output_dir}")


if __name__ == "__main__":
    main()
