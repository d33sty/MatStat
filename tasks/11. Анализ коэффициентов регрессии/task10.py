"""Взять произвольное значение обуславливающего фактора ( в условиях которого
измерение исследуемой величины не проводилось) и при этом значении
обуславливающего фактора
1)
Оценить среднее значение исследуемой величины ( в скалярном исчислении и в
матричном виде)
2)
Оценить дисперсию для такого среднего ( в скалярном исчислении и в матричном
виде)
3)
Написать доверительный интервал для такого среднего
4)
Оценить значение исследуемой величины , которое может быть результатом
одного измерения.
5)
Оценить дисперсию одного измерения в этой точке
6)
Написать прогнозный интервал для одного измерения
7)
Повторить пункты 1-6 для любого значения обуславливающего фактора ( написать
функциональные зависимости доверительных и прогонозных интервалов от
значения обуславливающего фактора)
8)
Представить на одном графике: экспериментальные данные, фитированную
зависимость, доверительный интервал, прогнозный интрвал."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Исходные данные
# ============================================================
m = np.array([15, 20, 30, 40, 50, 60], dtype=float)
N = np.array([1305, 1457, 2380, 3074, 3615, 4420], dtype=float)
n = len(m)

print("Исходные данные")
print(f"масса U, г:          {m}")
print(f"количество импульсов:{N}")
print()

# ============================================================
# ЧАСТЬ 1. Расчёт коэффициентов регрессии (равноточные измерения)
# ============================================================
print("=" * 70)
print("РАСЧЁТ КОЭФФИЦИЕНТОВ ЛИНЕЙНОЙ ЗАВИСИМОСТИ N = a*m + b")
print("=" * 70)

# ---------- Способ 1: scipy.stats.linregress ----------
slope, intercept, r_value, p_value, std_err = stats.linregress(m, N)

print("\nСпособ 1 (linregress):")
print(f"  a = {slope:.4f} имп/г")
print(f"  b = {intercept:.4f} имп")
print(f"  r  = {r_value:.6f},   r² = {r_value**2:.6f}")

# ---------- Способ 2: матричная форма МНК ----------
X = np.column_stack((np.ones(n), m))
theta = np.linalg.inv(X.T @ X) @ (X.T @ N)
b_matr, a_matr = theta[0], theta[1]

print("\nСпособ 2 (матричная форма):")
print(f"  a = {a_matr:.4f} имп/г")
print(f"  b = {b_matr:.4f} имп")

a, b = a_matr, b_matr

# ---------- 1) Оценённая дисперсия измерений ----------
N_pred = X @ theta
residuals = N - N_pred
p = 2
S2_y = np.sum(residuals**2) / (n - p)
S_y = np.sqrt(S2_y)

print(f"\n1) Оценённая дисперсия измерений:")
print(f"   S²_y = {S2_y:.1f}")
print(f"   S_y  = {S_y:.1f} имп")

# ---------- 2) Дисперсии и СКО оценок параметров ----------
m_mean = np.mean(m)
Sxx = np.sum((m - m_mean) ** 2)

S2_a = S2_y / Sxx
S2_b = S2_y * (1 / n + m_mean**2 / Sxx)

S_a = np.sqrt(S2_a)
S_b = np.sqrt(S2_b)

print(f"\n2) Стандартные ошибки коэффициентов:")
print(f"   S_a = {S_a:.2f} имп/г")
print(f"   S_b = {S_b:.1f} имп")

# ---------- 3) Ковариационная матрица ----------
C = S2_y * np.linalg.inv(X.T @ X)

print(f"\n3) Ковариационная матрица параметров Cov([b, a]):")
print(f"   [  S²_b      Cov(b,a) ]   [ {C[0,0]:8.0f}   {C[0,1]:8.2f} ]")
print(f"   [ Cov(a,b)   S²_a     ] = [ {C[1,0]:8.2f}   {C[1,1]:8.2f} ]")
print(f"   (порядок: b (свободный член), a (наклон))")

# ---------- 4) Доверительные интервалы для a и b ----------
alpha = 0.05
t_cr = stats.t.ppf(1 - alpha / 2, n - p)

delta_a = t_cr * S_a
delta_b = t_cr * S_b

print(f"\n4) Доверительные интервалы для параметров (α = {alpha}, t_кр = {t_cr:.3f}):")
print(f"   a : [{a - delta_a:.2f};  {a + delta_a:.2f}] имп/г")
print(f"   b : [{b - delta_b:.0f};  {b + delta_b:.0f}] имп")

# ---------- 5) Проверка гипотез ----------
t_a = a / S_a
t_b = b / S_b

p_a = 2 * (1 - stats.t.cdf(abs(t_a), n - p))
p_b = 2 * (1 - stats.t.cdf(abs(t_b), n - p))

print(f"\n5) Проверка гипотез о значимости коэффициентов (α = {alpha}):")
print(
    f"   H0: a = 0  |  t = {t_a:.2f},  p-value = {p_a:.6f}  →  {'ОТВЕРГАЕТСЯ' if p_a < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)
print(
    f"   H0: b = 0  |  t = {t_b:.2f},  p-value = {p_b:.6f}  →  {'ОТВЕРГАЕТСЯ' if p_b < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)
print()

# ============================================================
# ЧАСТЬ 2. ПРОГНОЗ ДЛЯ ПРОИЗВОЛЬНОГО ЗНАЧЕНИЯ m0
# ============================================================
m0 = 25.0  # масса 25 г — значение между 20 и 30, где измерений не было

print("=" * 70)
print(f"ПРОГНОЗ ДЛЯ m0 = {m0} г")
print("=" * 70)

# ---------- 1) Оценка среднего значения ----------
N0 = a * m0 + b
print(f"\n1) Оценка среднего значения:")
print(f"   N0 = a*m0 + b = {a:.2f}*{m0} + {b:.0f} = {N0:.0f} имп")

x0 = np.array([1, m0])
N0_matr = x0 @ theta
print(f"   (матричная форма: N0 = {N0_matr:.0f} имп)")

# ---------- 2) Дисперсия среднего значения ----------
var_N0_scalar = S2_y * (1 / n + (m0 - m_mean) ** 2 / Sxx)
var_N0_matrix = x0 @ C @ x0

print(f"\n2) Дисперсия среднего значения:")
print(f"   Var(N0) = S2_y*(1/n + (m0-m_mean)^2/Sxx) = {var_N0_scalar:.1f}")
print(f"   S(N0) = {np.sqrt(var_N0_scalar):.1f} имп")

# ---------- 3) Доверительный интервал для среднего ----------
delta_N0_di = t_cr * np.sqrt(var_N0_scalar)
ci_lower = N0 - delta_N0_di
ci_upper = N0 + delta_N0_di

print(f"\n3) Доверительный интервал для среднего (α = {alpha}):")
print(f"   Δ = {delta_N0_di:.1f} имп")
print(f"   CI: [{ci_lower:.0f}; {ci_upper:.0f}] имп")

# ---------- 4) Прогнозное значение для одного измерения ----------
N0_pred = N0
print(f"\n4) Прогнозное значение для одного измерения:")
print(f"   N0_pred = N0 = {N0_pred:.0f} имп")

# ---------- 5) Дисперсия прогноза ----------
var_pred_scalar = S2_y + var_N0_scalar
var_pred_matrix = S2_y + x0 @ C @ x0

print(f"\n5) Дисперсия прогноза для одного измерения:")
print(
    f"   Var(N0_pred) = S2_y + Var(N0) = {S2_y:.1f} + {var_N0_scalar:.1f} = {var_pred_scalar:.1f}"
)
print(f"   S(N0_pred) = {np.sqrt(var_pred_scalar):.1f} имп")

# ---------- 6) Прогнозный интервал для одного измерения ----------
delta_N0_pi = t_cr * np.sqrt(var_pred_scalar)
pi_lower = N0_pred - delta_N0_pi
pi_upper = N0_pred + delta_N0_pi

print(f"\n6) Прогнозный интервал для одного измерения (α = {alpha}):")
print(f"   Δ = {delta_N0_pi:.1f} имп")
print(f"   PI: [{pi_lower:.0f}; {pi_upper:.0f}] имп")

# ============================================================
# ЧАСТЬ 3. Функциональные зависимости интервалов
# ============================================================
print("\n" + "=" * 70)
print("ФУНКЦИОНАЛЬНЫЕ ЗАВИСИМОСТИ")
print("=" * 70)

print(f"\nN(m) = {a:.2f}·m + {b:.0f}")
print(f"Var_сред(m) = S2_y·(1/n + (m - {m_mean:.1f})² / {Sxx:.1f})")
print(f"Var_прогн(m) = S2_y + Var_сред(m)")

# ============================================================
# ЧАСТЬ 4. ГРАФИКИ
# ============================================================
m_plot = np.linspace(0, 80, 200)
N_plot = a * m_plot + b

# Доверительные интервалы для среднего
var_mean_plot = S2_y * (1 / n + (m_plot - m_mean) ** 2 / Sxx)
ci_lower_plot = N_plot - t_cr * np.sqrt(var_mean_plot)
ci_upper_plot = N_plot + t_cr * np.sqrt(var_mean_plot)

# Прогнозные интервалы для одного измерения
var_pred_plot = S2_y + var_mean_plot
pi_lower_plot = N_plot - t_cr * np.sqrt(var_pred_plot)
pi_upper_plot = N_plot + t_cr * np.sqrt(var_pred_plot)

fig, ax = plt.subplots(figsize=(12, 7))

# Экспериментальные данные
ax.scatter(m, N, color="blue", s=80, zorder=5, label="экспериментальные данные")

# Линия регрессии
ax.plot(
    m_plot, N_plot, "r-", lw=2, label=f"линейная регрессия: N = {a:.2f}·m + {b:.0f}"
)

# Доверительный интервал для среднего
ax.fill_between(
    m_plot,
    ci_lower_plot,
    ci_upper_plot,
    color="red",
    alpha=0.15,
    label="доверительный интервал для среднего (95%)",
)

# Прогнозный интервал для одного измерения
ax.fill_between(
    m_plot,
    pi_lower_plot,
    pi_upper_plot,
    color="gray",
    alpha=0.25,
    label="прогнозный интервал для одного измерения (95%)",
)

# Точка m0 и её интервалы
ax.scatter(
    [m0],
    [N0],
    color="green",
    s=100,
    zorder=10,
    marker="s",
    label=f"прогноз при m = {m0} г",
)

# Вертикальная линия для m0
ax.axvline(x=m0, color="green", linestyle="--", alpha=0.5)

# Отображение интервалов в точке m0
ax.plot([m0, m0], [ci_lower, ci_upper], "r-", lw=3, alpha=0.7)
ax.plot([m0, m0], [pi_lower, pi_upper], "gray", lw=3, alpha=0.7)

# Оформление графика
ax.set_xlabel("масса U, г")
ax.set_ylabel("количество импульсов")
ax.set_title("Линейная регрессия с доверительными и прогнозными интервалами")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 80)
ax.set_ylim(0, 6000)

plt.tight_layout()

# ============================================================
# ЧАСТЬ 5. Сравнение интервалов для разных m
# ============================================================
print("\n" + "=" * 70)
print("СРАВНЕНИЕ ИНТЕРВАЛОВ ДЛЯ РАЗНЫХ ЗНАЧЕНИЙ m")
print("=" * 70)

test_points = [10, 25, 35, 55, 70]
print(
    f"\n{'m, г':<8} {'N(m)':<8} {'95% CI для среднего':<28} {'95% PI для одного':<28}"
)
print("-" * 72)

for m_test in test_points:
    N_test = a * m_test + b
    var_mean_test = S2_y * (1 / n + (m_test - m_mean) ** 2 / Sxx)
    var_pred_test = S2_y + var_mean_test
    ci_delta = t_cr * np.sqrt(var_mean_test)
    pi_delta = t_cr * np.sqrt(var_pred_test)

    ci_str = f"[{N_test - ci_delta:.0f}; {N_test + ci_delta:.0f}]"
    pi_str = f"[{N_test - pi_delta:.0f}; {N_test + pi_delta:.0f}]"

    print(f"{m_test:<8.0f} {N_test:<8.0f} {ci_str:<28} {pi_str:<28}")

print("\nПримечания:")
print(
    "- CI (доверительный интервал): интервал для среднего значения при фиксированном m"
)
print(
    "- PI (прогнозный интервал): интервал для отдельного измерения при фиксированном m"
)
print("- Интервалы расширяются при удалении от центра выборки (m_mean ≈ 35.8)")

plt.show()
