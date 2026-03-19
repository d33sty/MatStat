import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# данные измерений темнового тока (А)
X0 = [
    -2.47779e-11,
    -2.48390e-11,
    -2.48732e-11,
    -2.49078e-11,
    -2.49546e-11,
    -2.49546e-11,
    -2.49489e-11,
    -2.49330e-11,
    -2.49634e-11,
    -2.49153e-11,
    -2.48980e-11,
    -2.48304e-11,
    -2.50186e-11,
    -2.50640e-11,
    -2.49451e-11,
    -2.50279e-11,
    -2.50096e-11,
    -2.48821e-11,
    -2.48304e-11,
    -2.49670e-11,
    -2.48384e-11,
    -2.48677e-11,
    -2.49263e-11,
    -2.49318e-11,
    -2.48630e-11,
    -2.49104e-11,
    -2.49060e-11,
    -2.49383e-11,
    -2.48665e-11,
    -2.49377e-11,
    -2.49369e-11,
    -2.48982e-11,
    -2.48058e-11,
    -2.48948e-11,
    -2.48858e-11,
    -2.48231e-11,
    -2.48180e-11,
    -2.49383e-11,
    -2.46490e-11,
    -2.49801e-11,
    -2.50328e-11,
    -2.49705e-11,
    -2.49776e-11,
    -2.50668e-11,
    -2.49098e-11,
    -2.49125e-11,
    -2.48439e-11,
    -2.49503e-11,
    -2.47525e-11,
    -2.49157e-11,
    -2.48620e-11,
    -2.48160e-11,
    -2.49609e-11,
    -2.48970e-11,
    -2.49333e-11,
    -2.48891e-11,
    -2.48551e-11,
    -2.49153e-11,
    -2.49359e-11,
    -2.48630e-11,
    -2.49491e-11,
    -2.48939e-11,
    -2.49306e-11,
    -2.48795e-11,
    -2.48337e-11,
    -2.49082e-11,
    -2.50165e-11,
    -2.48907e-11,
    -2.48738e-11,
    -2.47683e-11,
    -2.48636e-11,
    -2.49913e-11,
    -2.49841e-11,
    -2.49566e-11,
    -2.50086e-11,
    -2.48406e-11,
    -2.48687e-11,
    -2.48367e-11,
    -2.49279e-11,
    -2.49750e-11,
    -2.49279e-11,
    -2.48160e-11,
    -2.49066e-11,
    -2.48730e-11,
    -2.49029e-11,
    -2.49699e-11,
    -2.48695e-11,
    -2.50356e-11,
    -2.49499e-11,
    -2.49387e-11,
    -2.50189e-11,
    -2.50008e-11,
    -2.49157e-11,
    -2.49147e-11,
    -2.49418e-11,
    -2.49029e-11,
    -2.50106e-11,
    -2.49542e-11,
    -2.48976e-11,
    -2.50242e-11,
    -2.49328e-11,
    -2.48524e-11,
    -2.48522e-11,
    -2.49583e-11,
    -2.49454e-11,
    -2.48852e-11,
    -2.48964e-11,
    -2.47999e-11,
    -2.49754e-11,
    -2.49571e-11,
    -2.49007e-11,
    -2.50365e-11,
    -2.49383e-11,
    -2.49866e-11,
    -2.49837e-11,
]

# уровень значимости
alpha = 0.05

print("=" * 70)
print("Анализ распределения темнового тока детектора Keithley")
print("=" * 70)
print(f"количество измерений: n = {len(X0)}")
print(f"уровень значимости: alpha = {alpha}")
print()

# преобразуем в массив numpy для удобства
X = np.array(X0)

# основные статистические характеристики
mean_X = np.mean(X)
std_X = np.std(X, ddof=1)
var_X = std_X**2

print("Основные статистические характеристики:")
print(f"среднее значение: mean = {mean_X:.3e} А")
print(f"СКО: std = {std_X:.3e} А")
print(f"дисперсия: var = {var_X:.3e} А^2")
print()

# ============================================================
# 1. Проверка гипотезы о нормальном распределении
# ============================================================

print("-" * 70)
print("1. Проверка гипотезы о нормальном распределении")
print("-" * 70)

# 1.1 Критерий Пирсона (хи-квадрат)
print("1.1 Критерий Пирсона (chi^2)")

# количество интервалов (правило Стерджесса)
k = int(1 + np.log2(len(X)))
print(f"количество интервалов (правило Стерджесса): k = {k}")

# наблюдаемые частоты
observed, bins = np.histogram(X, bins=k)

# теоретические вероятности для нормального распределения
cdf_vals = stats.norm.cdf(bins, mean_X, std_X)
probabilities = np.diff(cdf_vals)

# ожидаемые частоты
expected = probabilities * len(X)

# объединяем интервалы с ожидаемой частотой < 5
min_expected = 5
while np.any(expected < min_expected) and len(expected) > 1:
    # находим интервал с минимальной ожидаемой частотой
    min_idx = np.argmin(expected)
    if min_idx == 0:
        # объединяем с правым соседом
        observed[min_idx + 1] += observed[min_idx]
        expected[min_idx + 1] += expected[min_idx]
        observed = np.delete(observed, min_idx)
        expected = np.delete(expected, min_idx)
    elif min_idx == len(expected) - 1:
        # объединяем с левым соседом
        observed[min_idx - 1] += observed[min_idx]
        expected[min_idx - 1] += expected[min_idx]
        observed = np.delete(observed, min_idx)
        expected = np.delete(expected, min_idx)
    else:
        # объединяем с меньшим из соседей
        if expected[min_idx - 1] < expected[min_idx + 1]:
            observed[min_idx - 1] += observed[min_idx]
            expected[min_idx - 1] += expected[min_idx]
        else:
            observed[min_idx + 1] += observed[min_idx]
            expected[min_idx + 1] += expected[min_idx]
        observed = np.delete(observed, min_idx)
        expected = np.delete(expected, min_idx)

k_after = len(observed)
print(f"количество интервалов после объединения: k' = {k_after}")

# статистика хи-квадрат
chi2_stat = np.sum((observed - expected) ** 2 / expected)

# степени свободы (k' - 1 - число оцененных параметров)
# для нормального распределения оцениваем 2 параметра (mean и std)
df = k_after - 1 - 2

# критическое значение
chi2_critical = stats.chi2.ppf(1 - alpha, df)

# p-value
p_value_chi2 = 1 - stats.chi2.cdf(chi2_stat, df)

print(f"статистика chi^2 = {chi2_stat:.4f}")
print(f"степени свободы: df = {df}")
print(f"критическое значение (alpha=0.05): chi^2_кр = {chi2_critical:.4f}")
print(f"p-value = {p_value_chi2:.4f}")

if chi2_stat < chi2_critical:
    print(f"chi^2 < chi^2_кр -> гипотеза о нормальном распределении ПРИНИМАЕТСЯ")
else:
    print(f"chi^2 >= chi^2_кр -> гипотеза о нормальном распределении ОТВЕРГАЕТСЯ")

if p_value_chi2 > alpha:
    print(f"p-value > {alpha} -> гипотеза о нормальном распределении ПРИНИМАЕТСЯ")
else:
    print(f"p-value <= {alpha} -> гипотеза о нормальном распределении ОТВЕРГАЕТСЯ")
print()

# 1.2 Критерий Колмогорова-Смирнова
print("1.2 Критерий Колмогорова-Смирнова")

n = len(X)
mu, sigma = np.mean(X), np.std(X, ddof=1)

# расчет статистики D и p-value
stat, p_val = stats.kstest(X, "norm", args=(mu, sigma))
lam = stat * np.sqrt(n)  # аргумент для функции Колмогорова

print(f"статистика D = {stat:.4f}")
print(f"лямбда = {lam:.4f}")
print(f"p-value = {p_val:.4f}")

if p_val > alpha:
    print(f"p-value > {alpha} -> гипотеза о нормальном распределении ПРИНИМАЕТСЯ")
else:
    print(f"p-value <= {alpha} -> гипотеза о нормальном распределении ОТВЕРГАЕТСЯ")
print()

# ============================================================
# 2. Проверка гипотезы о равномерном распределении
# ============================================================

print("-" * 70)
print("2. Проверка гипотезы о равномерном распределении")
print("-" * 70)

# параметры равномерного распределения (min и max выборки)
a = np.min(X)
b = np.max(X)
print(f"параметры равномерного распределения: a = {a:.3e}, b = {b:.3e}")

# 2.1 Критерий Пирсона (хи-квадрат) для равномерного распределения
print("2.1 Критерий Пирсона (chi^2)")

# количество интервалов (правило Стерджесса)
k = int(1 + np.log2(len(X)))
print(f"количество интервалов: k = {k}")

# наблюдаемые частоты
observed_unif, bins_unif = np.histogram(X, bins=k)

# теоретические вероятности для равномерного распределения
# для равномерного распределения вероятность попасть в каждый интервал одинакова
prob_unif = 1 / k

# ожидаемые частоты
expected_unif = np.full(k, prob_unif * len(X))

# статистика хи-квадрат
chi2_stat_unif = np.sum((observed_unif - expected_unif) ** 2 / expected_unif)

# степени свободы (k - 1 - число оцененных параметров)
# для равномерного распределения оцениваем 2 параметра (a и b)
df_unif = k - 1 - 2

# критическое значение
chi2_critical_unif = stats.chi2.ppf(1 - alpha, df_unif)

# p-value
p_value_chi2_unif = 1 - stats.chi2.cdf(chi2_stat_unif, df_unif)

print(f"статистика chi^2 = {chi2_stat_unif:.4f}")
print(f"степени свободы: df = {df_unif}")
print(f"критическое значение (alpha=0.05): chi^2_кр = {chi2_critical_unif:.4f}")
print(f"p-value = {p_value_chi2_unif:.4f}")

if chi2_stat_unif < chi2_critical_unif:
    print(f"chi^2 < chi^2_кр -> гипотеза о равномерном распределении ПРИНИМАЕТСЯ")
else:
    print(f"chi^2 >= chi^2_кр -> гипотеза о равномерном распределении ОТВЕРГАЕТСЯ")

if p_value_chi2_unif > alpha:
    print(f"p-value > {alpha} -> гипотеза о равномерном распределении ПРИНИМАЕТСЯ")
else:
    print(f"p-value <= {alpha} -> гипотеза о равномерном распределении ОТВЕРГАЕТСЯ")
print()

# 2.2 Критерий Колмогорова-Смирнова для равномерного распределения
print("2.2 Критерий Колмогорова-Смирнова")

# для равномерного распределения нужно передать параметры
# stats.uniform.cdf(x, loc=a, scale=b-a)
stat_unif, p_val_unif = stats.kstest(X, "uniform", args=(a, b - a))
lam_unif = stat_unif * np.sqrt(n)

print(f"статистика D = {stat_unif:.4f}")
print(f"лямбда = {lam_unif:.4f}")
print(f"p-value = {p_val_unif:.4f}")

if p_val_unif > alpha:
    print(f"p-value > {alpha} -> гипотеза о равномерном распределении ПРИНИМАЕТСЯ")
else:
    print(f"p-value <= {alpha} -> гипотеза о равномерном распределении ОТВЕРГАЕТСЯ")
print()

# ============================================================
# 3. Визуализация (создание всех фигур)
# ============================================================

print("-" * 70)
print("3. Визуализация результатов")
print("-" * 70)

# Фигура 1: Гистограмма с наложением теоретических распределений
fig1 = plt.figure(1, figsize=(12, 6))

# гистограмма эмпирических данных
counts, bins_hist, patches = plt.hist(
    X,
    bins=20,
    density=True,
    alpha=0.6,
    color="skyblue",
    edgecolor="black",
    label="эмпирические данные",
)

# нормальное распределение
x_norm = np.linspace(a, b, 100)
y_norm = stats.norm.pdf(x_norm, mean_X, std_X)
plt.plot(x_norm, y_norm, "r-", linewidth=2, label="нормальное распределение")

# равномерное распределение
y_unif = stats.uniform.pdf(x_norm, a, b - a)
plt.plot(x_norm, y_unif, "g-", linewidth=2, label="равномерное распределение")

plt.xlabel("ток, А")
plt.ylabel("плотность вероятности")
plt.title("Гистограмма эмпирических данных и теоретические распределения")
plt.legend()
plt.grid(True, alpha=0.3)

# Фигура 2: График для критерия Колмогорова (нормальное распределение)
fig2 = plt.figure(2, figsize=(12, 6))

# эмпирическая и теоретическая CDF
plt.subplot(1, 2, 1)
plt.hist(
    X,
    bins=20,
    edgecolor="black",
    cumulative=True,
    density=True,
    alpha=0.6,
    color="skyblue",
    label="эмпирическая CDF",
)
x_cdf = np.linspace(min(X), max(X), 100)
plt.plot(
    x_cdf,
    stats.norm.cdf(x_cdf, mean_X, std_X),
    "r-",
    linewidth=2,
    label="теоретическая CDF (нормальное)",
)
plt.xlabel("ток, А")
plt.ylabel("вероятность")
plt.title("CDF: эмпирическая vs нормальное")
plt.legend()
plt.grid(True, alpha=0.3)

# функция Колмогорова K(lambda)
plt.subplot(1, 2, 2)
l_space = np.linspace(0, 2.5, 300)
k_vals = stats.kstwobign.cdf(l_space)

plt.plot(
    l_space,
    k_vals,
    "b-",
    lw=2,
    label=r"$K(\lambda) = \sum_{k=-\infty}^{\infty} (-1)^k e^{-2k^2\lambda^2}$",
)

# точка текущего значения для нормального распределения
k_current = stats.kstwobign.cdf(lam)
plt.scatter([lam], [k_current], color="red", zorder=5, s=50)

plt.annotate(
    f"$\lambda$ = {lam:.3f}\n$K(\lambda)$ = {k_current:.3f}\n$p$-value = {p_val:.3f}",
    xy=(lam, k_current),
    xytext=(lam + 0.2, k_current - 0.2),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
    bbox=dict(boxstyle="round", fc="ghostwhite", ec="blue", alpha=0.5),
)

plt.title("Распределение Колмогорова (нормальное)")
plt.xlabel("$\lambda$")
plt.ylabel("$K(\lambda)$")
plt.axhline(1.0, color="black", lw=0.5, ls="--")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

# Фигура 3: График для критерия Колмогорова (равномерное распределение)
fig3 = plt.figure(3, figsize=(12, 6))

# эмпирическая и теоретическая CDF
plt.subplot(1, 2, 1)
plt.hist(
    X,
    bins=20,
    edgecolor="black",
    cumulative=True,
    density=True,
    alpha=0.6,
    color="lightgreen",
    label="эмпирическая CDF",
)
plt.plot(
    x_cdf,
    stats.uniform.cdf(x_cdf, a, b - a),
    "g-",
    linewidth=2,
    label="теоретическая CDF (равномерное)",
)
plt.xlabel("ток, А")
plt.ylabel("вероятность")
plt.title("CDF: эмпирическая vs равномерное")
plt.legend()
plt.grid(True, alpha=0.3)

# функция Колмогорова K(lambda)
plt.subplot(1, 2, 2)
plt.plot(l_space, k_vals, "b-", lw=2, label=r"$K(\lambda)$")

# точка текущего значения для равномерного распределения
k_current_unif = stats.kstwobign.cdf(lam_unif)
plt.scatter([lam_unif], [k_current_unif], color="red", zorder=5, s=50)

plt.annotate(
    f"$\lambda$ = {lam_unif:.3f}\n$K(\lambda)$ = {k_current_unif:.3f}\n$p$-value = {p_val_unif:.3f}",
    xy=(lam_unif, k_current_unif),
    xytext=(lam_unif + 0.2, k_current_unif - 0.2),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
    bbox=dict(boxstyle="round", fc="ghostwhite", ec="green", alpha=0.5),
)

plt.title("Распределение Колмогорова (равномерное)")
plt.xlabel("$\lambda$")
plt.ylabel("$K(\lambda)$")
plt.axhline(1.0, color="black", lw=0.5, ls="--")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

# ============================================================
# 4. Итоговый вывод
# ============================================================

print("-" * 70)
print("4. Итоговый вывод")
print("-" * 70)

print("Критерий Пирсона (chi^2):")
print(
    f"  нормальное распределение: p-value = {p_value_chi2:.4f} -> "
    + ("принимается" if p_value_chi2 > alpha else "отвергается")
)
print(
    f"  равномерное распределение: p-value = {p_value_chi2_unif:.4f} -> "
    + ("принимается" if p_value_chi2_unif > alpha else "отвергается")
)
print()

print("Критерий Колмогорова-Смирнова:")
print(
    f"  нормальное распределение: p-value = {p_val:.4f} -> "
    + ("принимается" if p_val > alpha else "отвергается")
)
print(
    f"  равномерное распределение: p-value = {p_val_unif:.4f} -> "
    + ("принимается" if p_val_unif > alpha else "отвергается")
)
print()


plt.show()
