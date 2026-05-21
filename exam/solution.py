import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ============================================================
# ЭКЗАМЕНАЦИОННАЯ ЗАДАЧА
# Зависимость температуры Z (С) от:
#   X - процентное содержание компоненты А в теплоносителе, %
#   Y - температура окружающей среды, С
# Модель: Z = a0 + a1*X + a2*Y  (множественная линейная регрессия)
# n = 11 наблюдений
# ============================================================

# -- Исходные данные ------------------------------------------
X = np.array([1, 4, 9, 11, 3, 8, 5, 10, 2, 7, 6], dtype=float)
Y = np.array([8, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4], dtype=float)
Z = np.array([6, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5], dtype=float)

n = len(Z)
p = 3  # число параметров: a0, a1, a2

print("=" * 60)
print("МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ: Z = a0 + a1*X + a2*Y")
print("=" * 60)
print(f"n = {n},  p = {p},  df = n - p = {n - p}")
print()

# -- 1. Матрица плана и МНК-оценка ----------------------------
print("-" * 60)
print("1. МНК-оценка параметров (матричная форма)")
print("-" * 60)

# матрица плана: каждая строка -- [1, X_i, Y_i]
X_mat = np.column_stack((np.ones(n), X, Y))

# theta = (X'X)^{-1} X'Z
XtX = X_mat.T @ X_mat
XtZ = X_mat.T @ Z
theta = np.linalg.inv(XtX) @ XtZ
a0, a1, a2 = theta

print(f"a0 (свободный член) = {a0:.4f}")
print(f"a1 (коэф. при X)    = {a1:.4f}")
print(f"a2 (коэф. при Y)    = {a2:.4f}")
print()
print(f"Уравнение регрессии: Z = {a0:.4f} + ({a1:.4f})*X + ({a2:.4f})*Y")
print()

# -- 2. Оценка качества модели --------------------------------
print("-" * 60)
print("2. Оценка качества модели")
print("-" * 60)

Z_pred = X_mat @ theta
residuals = Z - Z_pred

SS_res = np.sum(residuals**2)
SS_tot = np.sum((Z - np.mean(Z)) ** 2)
R2 = 1 - SS_res / SS_tot
R2_adj = 1 - (1 - R2) * (n - 1) / (n - p)

S2_y = SS_res / (n - p)
S_y = np.sqrt(S2_y)

print(f"S2_y (дисперсия остатков) = {S2_y:.4f}")
print(f"S_y  (СКО остатков)       = {S_y:.4f} С")
print(f"R2   (коэф. детерминации) = {R2:.4f}  ({R2*100:.1f}%)")
print(f"R2_скорр.                 = {R2_adj:.4f}")
print()

# -- 3. Ковариационная матрица и СКО коэффициентов ------------
print("-" * 60)
print("3. Дисперсии и СКО коэффициентов")
print("-" * 60)

# Cov(theta) = S2_y * (X'X)^{-1}
Cov = S2_y * np.linalg.inv(XtX)
S_theta = np.sqrt(np.diag(Cov))
S_a0, S_a1, S_a2 = S_theta

print(f"S(a0) = {S_a0:.4f}")
print(f"S(a1) = {S_a1:.4f}")
print(f"S(a2) = {S_a2:.4f}")
print()
print("Ковариационная матрица Cov([a0, a1, a2]):")
for row in Cov:
    print("  " + "  ".join(f"{v:10.5f}" for v in row))
print()

# -- 4. Доверительные интервалы -------------------------------
print("-" * 60)
print("4. Доверительные интервалы (alpha = 0.05, двусторонние)")
print("-" * 60)

alpha = 0.05
t_cr = stats.t.ppf(1 - alpha / 2, df=n - p)
print(f"t_кр (df={n-p}, alpha={alpha}) = {t_cr:.4f}")
print()

for name, val, se in zip(["a0", "a1", "a2"], theta, S_theta):
    lo = val - t_cr * se
    hi = val + t_cr * se
    print(f"  {name}: [{lo:.4f};  {hi:.4f}]   ({val:.4f} +- {t_cr*se:.4f})")
print()

# -- 5. Проверка значимости коэффициентов (H0: a_i = 0) -------
print("-" * 60)
print("5. Проверка значимости коэффициентов  H0: a_i = 0")
print("-" * 60)
print(
    f"{'Параметр':<10} {'Оценка':>10} {'S':>8} {'t-стат':>10} {'p-value':>10} {'H0':>16}"
)
print("-" * 60)

for name, val, se in zip(["a0", "a1", "a2"], theta, S_theta):
    t_stat = val / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - p))
    verdict = "ОТВЕРГАЕТСЯ" if p_val < alpha else "НЕ отвергается"
    print(
        f"  {name:<8} {val:>10.4f} {se:>8.4f} {t_stat:>10.4f} {p_val:>10.4f}   {verdict}"
    )
print()

# -- 6. Остатки по наблюдениям --------------------------------
print("-" * 60)
print("6. Остатки по наблюдениям")
print("-" * 60)
print(f"{'N':<4} {'Z':>6} {'Z_pred':>8} {'остаток':>10}")
for i in range(n):
    print(f"  {i+1:<3} {Z[i]:>6.1f} {Z_pred[i]:>8.4f} {residuals[i]:>10.4f}")
print()

# -- 7. Графики -----------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Множественная линейная регрессия: Z = a0 + a1*X + a2*Y", fontsize=13)

# График 1: Z измеренное vs Z предсказанное
ax = axes[0]
ax.scatter(Z, Z_pred, color="blue", s=60, zorder=5)
lim = [min(Z.min(), Z_pred.min()) - 1, max(Z.max(), Z_pred.max()) + 1]
ax.plot(lim, lim, "r--", lw=1.5, label="идеальная линия")
for i in range(n):
    ax.annotate(
        str(i + 1),
        (Z[i], Z_pred[i]),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=8,
    )
ax.set_xlabel("Z измеренное, С")
ax.set_ylabel("Z предсказанное, С")
ax.set_title("Z измеренное vs Z предсказанное")
ax.legend()
ax.grid(True, alpha=0.3)

# График 2: остатки vs Z предсказанное
ax = axes[1]
ax.scatter(Z_pred, residuals, color="green", s=60, zorder=5)
ax.axhline(0, color="red", linestyle="--", lw=1.5)
ax.axhline(2 * S_y, color="gray", linestyle=":", lw=1, label=f"+-2*S_y = +-{2*S_y:.2f}")
ax.axhline(-2 * S_y, color="gray", linestyle=":", lw=1)
for i in range(n):
    ax.annotate(
        str(i + 1),
        (Z_pred[i], residuals[i]),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=8,
    )
ax.set_xlabel("Z предсказанное, С")
ax.set_ylabel("Остатки, С")
ax.set_title("Остатки регрессии")
ax.legend()
ax.grid(True, alpha=0.3)

# График 3: Z измеренное и предсказанное по наблюдениям
ax = axes[2]
obs_idx = np.arange(1, n + 1)
ax.plot(obs_idx, Z, "bo-", markersize=6, label="Z измеренное")
ax.plot(obs_idx, Z_pred, "rs--", markersize=6, label="Z предсказанное")
ax.set_xlabel("Номер наблюдения")
ax.set_ylabel("Z, С")
ax.set_title("Z измеренное и предсказанное")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
