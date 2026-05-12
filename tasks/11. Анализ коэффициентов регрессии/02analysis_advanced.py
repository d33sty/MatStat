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
# ЧАСТЬ 1. Без учёта погрешностей (равноточные измерения)
# ============================================================
print("=" * 70)
print("ЧАСТЬ 1. Без учёта погрешностей")
print("=" * 70)

# ---------- Способ 1: scipy.stats.linregress ----------
slope, intercept, r_value, p_value, std_err = stats.linregress(m, N)

print("\nСпособ 1 (linregress):")
print(f"  a = {slope:.2f} имп/г")
print(f"  b = {intercept:.0f} имп")
print(f"  r  = {r_value:.6f},   r2 = {r_value**2:.6f}")

# ---------- Способ 2: матричная форма МНК ----------
X = np.column_stack((np.ones(n), m))
theta = np.linalg.inv(X.T @ X) @ (X.T @ N)
b_matr, a_matr = theta[0], theta[1]

print("\nСпособ 2 (матричная форма):")
print(f"  a = {a_matr:.2f} имп/г")
print(f"  b = {b_matr:.0f} имп")

a, b = a_matr, b_matr

# ---------- 1) Оценённая дисперсия ----------
N_pred = X @ theta
residuals = N - N_pred
p = 2
S2_y = np.sum(residuals**2) / (n - p)
S_y = np.sqrt(S2_y)

print(f"\n1) Оценённая дисперсия измерений:  S2_y = {S2_y:.1f}")
print(f"   Среднеквадратическое отклонение: S_y  = {S_y:.1f}")

# ---------- 2) Дисперсии оценок параметров ----------
m_mean = np.mean(m)
Sxx = np.sum((m - m_mean) ** 2)

S2_a = S2_y / Sxx
S2_b = S2_y * (1 / n + m_mean**2 / Sxx)

S_a = np.sqrt(S2_a)
S_b = np.sqrt(S2_b)

print(f"\n2) Дисперсии оценок параметров (скалярное исчисление):")
print(f"   S_a = {S_a:.2f} имп/г,   S2_a = {S2_a:.2f}")
print(f"   S_b = {S_b:.1f} имп,     S2_b = {S2_b:.0f}")

# ---------- 3) Ковариационная матрица ----------
C = S2_y * np.linalg.inv(X.T @ X)

print(f"\n3) Ковариационная матрица Cov([b, a]):")
print(f"   [  S2_b      Cov(b,a) ]   [ {C[0,0]:8.0f}   {C[0,1]:8.2f} ]")
print(f"   [ Cov(a,b)   S2_a     ] = [ {C[1,0]:8.2f}   {C[1,1]:8.2f} ]")
print(f"   (порядок: b (свободный член), a (наклон))")

# ---------- 4) Доверительные интервалы ----------
alpha = 0.05
t_cr = stats.t.ppf(1 - alpha / 2, n - p)

delta_a = t_cr * S_a
delta_b = t_cr * S_b

print(f"\n4) Доверительные интервалы (α = {alpha}, t_кр = {t_cr:.3f}):")
print(f"   a : [{a - delta_a:.2f};  {a + delta_a:.2f}] имп/г")
print(f"   b : [{b - delta_b:.0f};  {b + delta_b:.0f}] имп")

# ---------- 5) Проверка гипотез ----------
t_a = a / S_a
t_b = b / S_b

p_a = 2 * (1 - stats.t.cdf(abs(t_a), n - p))
p_b = 2 * (1 - stats.t.cdf(abs(t_b), n - p))

print(f"\n5) Проверка гипотез (α = {alpha}):")
print(
    f"   H0: a = 0  |  t = {t_a:.2f},  p-value = {p_a:.4f}    {'ОТВЕРГАЕТСЯ' if p_a < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)
print(
    f"   H0: b = 0  |  t = {t_b:.2f},  p-value = {p_b:.4f}    {'ОТВЕРГАЕТСЯ' if p_b < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)

# ---------- График 1 ----------
N_fit = a * m + b
fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.scatter(m, N, color="blue", s=60, zorder=5, label="экспериментальные данные")
ax1.plot(m, N_fit, "r-", lw=2, label=f"МНК: N = {a:.1f}·m + {b:.0f}")
ax1.set_xlabel("масса U, г")
ax1.set_ylabel("количество импульсов")
ax1.set_title("Зависимость количества импульсов от массы U")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# ============================================================
# ЧАСТЬ 2. С учётом погрешности 15 %
# ============================================================
print("\n" + "=" * 70)
print("ЧАСТЬ 2. С учётом погрешности 15 %")
print("=" * 70)

delta_N_15 = 0.15 * N
w = 1.0 / delta_N_15**2
W = np.diag(w)

# ---------- Взвешенный МНК (матричная форма) ----------
theta_w = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ N)
b_w, a_w = theta_w[0], theta_w[1]

print(f"\nВзвешенный МНК (матричная форма):")
print(f"  a_w = {a_w:.2f} имп/г")
print(f"  b_w = {b_w:.0f} имп")

# ---------- 1) Оценённая дисперсия (взвешенные остатки) ----------
N_pred_w = X @ theta_w
res_w = N - N_pred_w
S2_y_w = np.sum(w * res_w**2) / (n - p)

print(f"\n1) Взвешенная дисперсия:  S2_y_w = {S2_y_w:.3f}  (при нормировке ≈ 1)")

# ---------- 2) Дисперсии оценок параметров ----------
C_w = np.linalg.inv(X.T @ W @ X)

S2_a_w = C_w[1, 1]
S2_b_w = C_w[0, 0]
S_a_w = np.sqrt(S2_a_w)
S_b_w = np.sqrt(S2_b_w)

print(f"\n2) Дисперсии оценок параметров (взвешенный МНК):")
print(f"   S_a_w = {S_a_w:.2f} имп/г,   S2_a_w = {S2_a_w:.2f}")
print(f"   S_b_w = {S_b_w:.1f} имп,     S2_b_w = {S2_b_w:.0f}")

# ---------- 3) Ковариационная матрица ----------
print(f"\n3) Ковариационная матрица Cov_w([b, a]) = (X'WX)^{{-1}}:")
print(f"   [ {C_w[0,0]:8.0f}   {C_w[0,1]:8.2f} ]")
print(f"   [ {C_w[1,0]:8.2f}   {C_w[1,1]:8.2f} ]")

# ---------- 4) Доверительные интервалы ----------
delta_a_w = t_cr * S_a_w
delta_b_w = t_cr * S_b_w

print(f"\n4) Доверительные интервалы (α = {alpha}, t_кр = {t_cr:.3f}):")
print(f"   a_w : [{a_w - delta_a_w:.2f};  {a_w + delta_a_w:.2f}] имп/г")
print(f"   b_w : [{b_w - delta_b_w:.0f};  {b_w + delta_b_w:.0f}] имп")

# ---------- 5) Проверка гипотез ----------
t_aw = a_w / S_a_w
t_bw = b_w / S_b_w

p_aw = 2 * (1 - stats.t.cdf(abs(t_aw), n - p))
p_bw = 2 * (1 - stats.t.cdf(abs(t_bw), n - p))

print(f"\n5) Проверка гипотез (α = {alpha}):")
print(
    f"   H0: a_w = 0  |  t = {t_aw:.2f},  p-value = {p_aw:.4f}    {'ОТВЕРГАЕТСЯ' if p_aw < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)
print(
    f"   H0: b_w = 0  |  t = {t_bw:.2f},  p-value = {p_bw:.4f}    {'ОТВЕРГАЕТСЯ' if p_bw < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)

# ---------- График 2 ----------
N_fit_w = a_w * m + b_w
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.errorbar(
    m,
    N,
    yerr=delta_N_15,
    fmt="bo",
    capsize=6,
    markersize=7,
    label="данные (погрешность 15 %)",
)
ax2.plot(m, N_fit_w, "r-", lw=2, label=f"взвешенный МНК: N = {a_w:.1f}·m + {b_w:.0f}")
ax2.set_xlabel("масса U, г")
ax2.set_ylabel("количество импульсов")
ax2.set_title("Зависимость количества импульсов от массы U (с учётом погрешностей)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# ============================================================
# Итоговое сравнение
# ============================================================
print("\n" + "=" * 70)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 70)


# Функция для форматирования с учётом погрешности
def fmt_with_error(val, err):
    if err == 0:
        return f"{val:.2f}"
    # определяем порядок погрешности
    order = int(np.floor(np.log10(abs(err))))
    # округляем погрешность до 1-2 значащих цифр
    err_rounded = np.round(err, -order + 1)
    # округляем значение до той же точности
    if order >= 0:
        val_rounded = np.round(val, -order + 1)
        return f"{val_rounded:.0f} ± {err_rounded:.0f}"
    else:
        decimals = -order + 1
        val_rounded = np.round(val, decimals)
        err_rounded = np.round(err, decimals)
        return f"{val_rounded:.{decimals}f} ± {err_rounded:.{decimals}f}"


print(f"\n{'Параметр':<20} {'без погрешности':>28} {'с погр. 15%':>28}")
print("-" * 76)

print(f"{'a (имп/г)':<20} {a:>28.2f} {a_w:>28.2f}")
print(f"{'b (имп)':<20} {b:>28.0f} {b_w:>28.0f}")
print(f"{'S_a (имп/г)':<20} {S_a:>28.2f} {S_a_w:>28.2f}")
print(f"{'S_b (имп)':<20} {S_b:>28.1f} {S_b_w:>28.1f}")

print(f"\n{'Доверительные интервалы (95%)':^76}")
print(
    f"{'a (имп/г)':<20} [{a-delta_a:>8.2f}; {a+delta_a:>8.2f}]        [{a_w-delta_a_w:>8.2f}; {a_w+delta_a_w:>8.2f}]"
)
print(
    f"{'b (имп)':<20} [{b-delta_b:>8.0f}; {b+delta_b:>8.0f}]        [{b_w-delta_b_w:>8.0f}; {b_w+delta_b_w:>8.0f}]"
)

plt.show()
