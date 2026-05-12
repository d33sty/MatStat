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
print(f"  a = {slope:.4f} имп/г")
print(f"  b = {intercept:.4f} имп")
print(f"  r  = {r_value:.6f},   r2 = {r_value**2:.6f}")

# ---------- Способ 2: матричная форма МНК ----------
X = np.column_stack((np.ones(n), m))  # матрица плана [1, m]
theta = np.linalg.inv(X.T @ X) @ (X.T @ N)
b_matr, a_matr = theta[0], theta[1]

print("\nСпособ 2 (матричная форма):")
print(f"  a = {a_matr:.4f} имп/г")
print(f"  b = {b_matr:.4f} имп")

# Используем параметры из матричной формы (совпадают с linregress)
a, b = a_matr, b_matr

# ---------- 1) Оценённая дисперсия исследуемой величины ----------
N_pred = X @ theta  # предсказанные значения
residuals = N - N_pred  # остатки
p = 2  # число параметров (a, b)
S2_y = np.sum(residuals**2) / (n - p)  # несмещённая оценка дисперсии

print(f"\n1) Оценённая дисперсия измерений:  S2_y = {S2_y:.4f}")
print(f"   Среднеквадратическое отклонение: S_y  = {np.sqrt(S2_y):.4f}")

# ---------- 2) Дисперсии оценок параметров (скалярное исчисление) ----------
m_mean = np.mean(m)
Sxx = np.sum((m - m_mean) ** 2)  # сумма квадратов отклонений x

S2_a = S2_y / Sxx  # дисперсия наклона
S2_b = S2_y * (1 / n + m_mean**2 / Sxx)  # дисперсия свободного члена

S_a = np.sqrt(S2_a)
S_b = np.sqrt(S2_b)

print(f"\n2) Дисперсии оценок параметров (скалярное исчисление):")
print(f"   S2_a = {S2_a:.6f},  S_a = {S_a:.4f}")
print(f"   S2_b = {S2_b:.4f},  S_b = {S_b:.4f}")

# ---------- 3) Ковариационная матрица параметров ----------
# Cov(θ) = S²_y * (X'X)^{-1}
C = S2_y * np.linalg.inv(X.T @ X)

print(f"\n3) Ковариационная матрица параметров Cov([b, a]):")
print(f"   [  S2_b      Cov(b,a) ]   [ {C[0,0]:10.4f}  {C[0,1]:10.4f} ]")
print(f"   [ Cov(a,b)   S2_a     ] = [ {C[1,0]:10.4f}  {C[1,1]:10.6f} ]")
print(f"   (порядок параметров: [b, a], т.е. [свободный член, наклон])")

# Проверка: диагональные элементы должны совпасть со скалярными оценками
print(f"\n   Проверка: C[0,0] = {C[0,0]:.4f} = S2_b = {S2_b:.4f}")
print(f"            C[1,1] = {C[1,1]:.6f} = S2_a = {S2_a:.6f}")

# ---------- 4) Доверительные интервалы ----------
alpha = 0.05  # уровень значимости
t_cr = stats.t.ppf(1 - alpha / 2, n - p)  # критическое значение t

delta_a = t_cr * S_a
delta_b = t_cr * S_b

print(f"\n4) Доверительные интервалы (α = {alpha}, t_кр = {t_cr:.4f}):")
print(f"   a : [{a - delta_a:.4f};  {a + delta_a:.4f}]")
print(f"   b : [{b - delta_b:.4f};  {b + delta_b:.4f}]")

# ---------- 5) Проверка гипотез H0: a=0 и H0: b=0 ----------
t_a = a / S_a  # t-статистика для наклона
t_b = b / S_b  # t-статистика для свободного члена

p_a = 2 * (1 - stats.t.cdf(abs(t_a), n - p))
p_b = 2 * (1 - stats.t.cdf(abs(t_b), n - p))

print(f"\n5) Проверка гипотез (α = {alpha}):")
print(
    f"   H0: a = 0  |  t = {t_a:.4f},  p-value = {p_a:.6f}    {'ОТВЕРГАЕТСЯ' if p_a < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)
print(
    f"   H0: b = 0  |  t = {t_b:.4f},  p-value = {p_b:.6f}    {'ОТВЕРГАЕТСЯ' if p_b < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)

# ---------- График 1 ----------
N_fit = a * m + b
fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.scatter(m, N, color="blue", s=60, zorder=5, label="экспериментальные данные")
ax1.plot(
    m, N_fit, "r-", lw=2, label=f"МНК: N = {a:.2f}·m + {b:.2f}\n(без погрешностей)"
)
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

delta_N_15 = 0.15 * N  # абсолютные погрешности
w = 1.0 / delta_N_15**2  # веса (обратно пропорциональны σ²)
W = np.diag(w)  # весовая матрица

# ---------- Взвешенный МНК (матричная форма) ----------
theta_w = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ N)
b_w, a_w = theta_w[0], theta_w[1]

print(f"\nВзвешенный МНК (матричная форма):")
print(f"  a_w = {a_w:.4f} имп/г")
print(f"  b_w = {b_w:.4f} имп")

# Проверка через np.polyfit (способ 1)
a_pf, b_pf = np.polyfit(m, N, 1, w=w)
print(f"\nПроверка (polyfit с весами):")
print(f"  a_w = {a_pf:.4f},  b_w = {b_pf:.4f}")

# ---------- 1) Оценённая дисперсия (взвешенные остатки) ----------
N_pred_w = X @ theta_w
res_w = N - N_pred_w
# Взвешенная сумма квадратов остатков
S2_y_w = np.sum(w * res_w**2) / (n - p)

print(f"\n1) Оценённая дисперсия (взвешенная):  S2_y_w = {S2_y_w:.6f}")
print(f"   (при корректной модели ≈ 1)")

# ---------- 2) Дисперсии оценок параметров ----------
# При взвешенном МНК Cov(θ) = (X'WX)^{-1}
C_w = np.linalg.inv(X.T @ W @ X)

S2_a_w = C_w[1, 1]
S2_b_w = C_w[0, 0]
S_a_w = np.sqrt(S2_a_w)
S_b_w = np.sqrt(S2_b_w)

print(f"\n2) Дисперсии оценок параметров (взвешенный МНК):")
print(f"   S²_a_w = {S2_a_w:.6f},  S_a_w = {S_a_w:.4f}")
print(f"   S²_b_w = {S2_b_w:.4f},  S_b_w = {S_b_w:.4f}")

# ---------- 3) Ковариационная матрица ----------
print(f"\n3) Ковариационная матрица Cov_w([b, a]) = (X'WX)^{{-1}}:")
print(f"   [ {C_w[0,0]:10.4f}  {C_w[0,1]:10.6f} ]")
print(f"   [ {C_w[1,0]:10.6f}  {C_w[1,1]:10.6f} ]")

# ---------- 4) Доверительные интервалы ----------
delta_a_w = t_cr * S_a_w
delta_b_w = t_cr * S_b_w

print(f"\n4) Доверительные интервалы (α = {alpha}, t_кр = {t_cr:.4f}):")
print(f"   a_w : [{a_w - delta_a_w:.4f};  {a_w + delta_a_w:.4f}]")
print(f"   b_w : [{b_w - delta_b_w:.4f};  {b_w + delta_b_w:.4f}]")

# ---------- 5) Проверка гипотез ----------
t_aw = a_w / S_a_w
t_bw = b_w / S_b_w

p_aw = 2 * (1 - stats.t.cdf(abs(t_aw), n - p))
p_bw = 2 * (1 - stats.t.cdf(abs(t_bw), n - p))

print(f"\n5) Проверка гипотез (α = {alpha}):")
print(
    f"   H0: a_w = 0  |  t = {t_aw:.4f},  p-value = {p_aw:.6f}    {'ОТВЕРГАЕТСЯ' if p_aw < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
)
print(
    f"   H0: b_w = 0  |  t = {t_bw:.4f},  p-value = {p_bw:.6f}    {'ОТВЕРГАЕТСЯ' if p_bw < alpha else 'НЕ ОТВЕРГАЕТСЯ'}"
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
ax2.plot(m, N_fit_w, "r-", lw=2, label=f"взвешенный МНК: N = {a_w:.2f}·m + {b_w:.2f}")
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
print(f"{'Параметр':<25} {'без погрешности':>18} {'с погр. 15%':>18}")
print("-" * 62)
print(f"{'a (имп/г)':<25} {a:>18.4f} {a_w:>18.4f}")
print(f"{'b (имп)':<25} {b:>18.4f} {b_w:>18.4f}")
print(f"{'S_a':<25} {S_a:>18.4f} {S_a_w:>18.4f}")
print(f"{'S_b':<25} {S_b:>18.4f} {S_b_w:>18.4f}")
print(f"{'ДИ a (нижн.)':<25} {a-delta_a:>18.4f} {a_w-delta_a_w:>18.4f}")
print(f"{'ДИ a (верхн.)':<25} {a+delta_a:>18.4f} {a_w+delta_a_w:>18.4f}")
print(f"{'ДИ b (нижн.)':<25} {b-delta_b:>18.4f} {b_w-delta_b_w:>18.4f}")
print(f"{'ДИ b (верхн.)':<25} {b+delta_b:>18.4f} {b_w+delta_b_w:>18.4f}")

plt.show()
