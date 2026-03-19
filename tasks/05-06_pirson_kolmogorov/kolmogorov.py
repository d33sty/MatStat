import numpy as np
from scipy.stats import norm, kstest, kstwobign
import matplotlib.pyplot as plt

# дано
I = list(
    map(lambda x: x / 1000, [141, 144, 157, 147, 151, 165, 142, 150, 163, 161])
)  # ток в си в амперах


def main():
    n = len(I)
    mu, sigma = np.mean(I), np.std(I, ddof=1)

    # 1. Расчет статистики D и p-value
    stat, p_val = kstest(I, "norm", args=(mu, sigma))
    lam = stat * np.sqrt(n)  # Аргумент для функции Колмогорова

    # --- ГРАФИК 1: CDF ---
    plt.figure(1)
    plt.hist(
        I,
        bins=5,
        edgecolor="black",
        cumulative=True,
        density=True,
        label="Эмпирическая CDF",
    )
    x_cdf = np.linspace(min(I), max(I), 100)
    plt.plot(x_cdf, norm.cdf(x_cdf, mu, sigma), "r-", label="Теоретическая CDF")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- ГРАФИК 2: Функция Колмогорова K(lambda) ---
    plt.figure(2)
    l_space = np.linspace(0, 2.5, 300)
    k_vals = kstwobign.cdf(l_space)  # Значения K(lambda)

    plt.plot(
        l_space,
        k_vals,
        "b-",
        lw=2,
        label=r"$K(\lambda) = \sum_{k=-\infty}^{\infty} (-1)^k e^{-2k^2\lambda^2}$",
    )

    # Точка текущего значения
    k_current = kstwobign.cdf(lam)
    plt.scatter([lam], [k_current], color="red", zorder=5)

    # Аннотация с p-value
    # p-value здесь это 1 - K(lambda)
    plt.annotate(
        f"$\lambda$ = {lam:.3f}\n$K(\lambda)$ = {k_current:.3f}\n$p$-value = {p_val:.3f}",
        xy=(lam, k_current),
        xytext=(lam + 0.2, k_current - 0.2),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        bbox=dict(boxstyle="round", fc="ghostwhite", ec="blue", alpha=0.5),
    )

    plt.title("Распределение Колмогорова и расчетный $p$-value")
    plt.xlabel("$\lambda$ (Статистика)")
    plt.ylabel("Вероятность $K(\lambda)$")
    plt.axhline(1.0, color="black", lw=0.5, ls="--")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")

    # Блокирующий вызов в самом конце
    plt.show()


main()
