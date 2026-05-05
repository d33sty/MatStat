import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# исходные данные
m = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
N = np.array([52.0, 58.6, 60.0, 62.6, 65.4, 64.1, 72.6, 72.4, 82.4, 83.8])

print("Исходные данные")
print(f"масса U, г: {m}")
print(f"количество импульсов: {N}")
print()

# ================ 1. Без учета погрешностей ================

# способ 1: scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(m, N)

print("Способ 1 (linregress):")
print(f"коэффициент наклона (a) = {slope:.2f} имп/г")
print(f"свободный член (b) = {intercept:.2f} имп")
print(f"коэффициент корреляции r = {r_value:.4f}")
print(f"r^2 = {r_value**2:.4f}")
print()

# способ 2: матричная форма мнк
X = np.column_stack((np.ones(len(m)), m))
tetta = np.linalg.inv(X.T @ X) @ (X.T @ N)
a_matr = tetta[1]
b_matr = tetta[0]

print("Способ 2 (матричная форма):")
print(f"коэффициент наклона (a) = {a_matr:.2f} имп/г")
print(f"свободный член (b) = {b_matr:.2f} имп")
print("\n\n\nЧерез C_tetta")
print(f"{N - (a_matr*m + b_matr)=}")
Sy = (1 / (len(m) - 2)) * sum(map(lambda x: x**2, N - (slope * m + intercept)))
print(f"{Sy=}")
C_tetta = np.linalg.inv(X.T @ X) * Sy
print(C_tetta)
var_b = C_tetta[0][0]
var_a = C_tetta[1][1]
delta_b = stats.t.ppf(0.975, len(m) - 1) * var_b**0.5
delta_a = stats.t.ppf(0.975, len(m) - 1) * var_a**0.5
print(f"{var_a=}")
print(f"{var_b=}")
print(f"{delta_a=}")
print(f"{delta_b=}")
print(f"{p_value}")

print("\n\n\nФормулами Sa, Sb")
x_sred = np.mean(m)

Sx = (1 / (len(m) - 1)) * sum(map(lambda x: x**2, m - x_sred))
Sa_form = Sy / (Sx * (len(m) - 1))
Sb_form = Sy * (1 / len(m) + x_sred / sum(map(lambda x: x**2, m - x_sred)))
delta_b = stats.t.ppf(0.975, len(m) - 1) * Sb_form**0.5
delta_a = stats.t.ppf(0.975, len(m) - 1) * Sa_form**0.5

print(f"{delta_a=}")
print(f"{delta_b=}")

print("\n\n\nТочка 210")
var_q210 = var_a * 210**2 + var_b + 2 * 210 * C_tetta[1][0]
print(f"{var_q210=}\n")

column_ba = np.array([[b_matr], [a_matr]])
row_210 = [1, 210]
q_210 = row_210 @ column_ba
var_q210_matr = row_210 @ C_tetta @ row_210
print(f"{q_210=}")
print(f"{var_q210_matr=}")
# TODO построить график var_qX_matr(x)

print("\n\n\n")
# ================ 2. График без учета погрешностей ================

N_fit = slope * m + intercept

plt.figure(1, figsize=(10, 6))
plt.scatter(m, N, color="blue", s=50, label="экспериментальные данные")
plt.plot(
    m,
    N_fit,
    "r-",
    linewidth=2,
    label=f"линейная зависимость: N = {slope:.2f}*m + {intercept:.2f}",
)
plt.xlabel("масса U, г")
plt.ylabel("количество импульсов")
plt.title("зависимость количества импульсов от массы U")
plt.legend()
plt.grid(True, alpha=0.3)

# ================ 3. С учетом погрешности 15% ================

delta_N = 0.15 * N
gn = 1 / (delta_N**2)  # веса (обратно пропорциональны квадрату погрешности)

# способ 1: взвешенный метод через polyfit
slope_w, intercept_w = np.polyfit(m, N, 1, w=gn)

print("Учет погрешности 15% (взвешенный метод):")
print(f"коэффициент наклона (a) = {slope_w:.2f} имп/г")
print(f"свободный член (b) = {intercept_w:.2f} имп")
print()

# способ 2: матричная форма взвешенного метода
Gn = np.diag(gn)
tetta_w = np.linalg.inv(X.T @ Gn @ X) @ (X.T @ Gn @ N)
a_matr_w = tetta_w[1]
b_matr_w = tetta_w[0]

print("Взвешенный метод (матричная форма):")
print(f"коэффициент наклона (a) = {a_matr_w:.2f} имп/г")
print(f"свободный член (b) = {b_matr_w:.2f} имп")
print()

# ================ 4. График с учетом погрешностей ================

N_fit_w = slope_w * m + intercept_w

plt.figure(2, figsize=(10, 6))
plt.errorbar(
    m,
    N,
    yerr=delta_N,
    fmt="bo",
    capsize=5,
    markersize=8,
    label="экспериментальные данные (погрешность 15%)",
)
plt.plot(
    m,
    N_fit_w,
    "r-",
    linewidth=2,
    label=f"линейная зависимость: N = {slope_w:.2f}*m + {intercept_w:.2f}",
)
plt.xlabel("масса U, г")
plt.ylabel("количество импульсов")
plt.title("зависимость количества импульсов от массы U (с учетом погрешностей)")
plt.legend()
plt.grid(True, alpha=0.3)

# ================ 5. Сравнение результатов ================

print("=" * 70)
print("Сравнение результатов")
print("=" * 70)
print("Параметр                   без учета погрешности    с учетом погрешности 15%")
print("-" * 70)
print(f"a (имп/г)                  {slope:<25.2f} {slope_w:<30.2f}")
print(f"b (имп)                    {intercept:<25.2f} {intercept_w:<30.2f}")
print(f"r^2                        {r_value**2:<25.4f} {'(не применимо)':<30}")

plt.show()
