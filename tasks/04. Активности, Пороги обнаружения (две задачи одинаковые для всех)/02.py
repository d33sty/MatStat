import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Исходные данные
mu_0 = 3.2  # среднее при нулевой гипотезе, пКи/г
sigma = 0.2  # стандартное отклонение, пКи/г
alpha = 0.01  # уровень значимости
Lc_given = 3.7  # заданное критическое значение для частей (b)-(e)

print("Задача: уран в почве")
print(f"среднее при нулевой гипотезе: mu_0 = {mu_0} пКи/г")
print(f"стандартное отклонение: sigma = {sigma} пКи/г")
print(f"уровень значимости: alpha = {alpha}")
print()

# Часть (a) - Критическое значение Lc

print("Часть (a)")

z_alpha = stats.norm.ppf(1 - alpha)  # квантиль для одностороннего теста
Lc = mu_0 + z_alpha * sigma

print(f"квантиль для alpha={alpha}: z_0.99 = {z_alpha:.3f}")
print(f"критическое значение: Lc = {mu_0} + {z_alpha:.3f} * {sigma} = {Lc:.3f} пКи/г")
print()

# Часть (b) - Ошибка II рода при mu = 3.4 и Lc = 3.7

print("Часть (b)")

mu_true_b = 3.4
z_b = (Lc_given - mu_true_b) / sigma
beta_b = stats.norm.cdf(z_b)
power_b = 1 - beta_b

print(f"истинное среднее: mu = {mu_true_b} пКи/г")
print(f"z = (Lc - mu) / sigma = ({Lc_given} - {mu_true_b}) / {sigma} = {z_b:.3f}")
print(f"beta = Phi({z_b:.3f}) = {beta_b:.4f}")
print(f"мощность = 1 - beta = {power_b:.4f}")
print()

# Часть (c) - Для других значений mu

print("Часть (c)")

mu_values = [3.6, 3.8, 4.0, 4.1]
results = []

print(f"при Lc = {Lc_given} пКи/г:")
print("  mu    z      beta    мощность")
for mu in mu_values:
    z = (Lc_given - mu) / sigma
    beta = stats.norm.cdf(z)
    power = 1 - beta
    results.append((mu, z, beta, power))
    print(f"  {mu:.1f}  {z:+.3f}  {beta:.4f}  {power:.4f}")
print()

# Часть (d) - Построение графика

print("Часть (d)")

# Добавляем точку из части (b)
all_mu = [3.4] + mu_values
all_power = [power_b] + [r[3] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(all_mu, all_power, "bo-", linewidth=2, markersize=8)
plt.xlabel("истинное среднее mu, пКи/г")
plt.ylabel("мощность (1 - beta)")
plt.title("зависимость мощности от истинного значения mu")
plt.grid(True, alpha=0.3)
plt.axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="мощность 0.8")
plt.axvline(x=3.5, color="g", linestyle="--", alpha=0.5, label="mu = 3.5")
plt.legend()
plt.show()

print("график построен")
print()

# Часть (e) - Мощность при mu = 3.5 по графику

print("Часть (e)")

mu_target = 3.5
z_target = (Lc_given - mu_target) / sigma
beta_target = stats.norm.cdf(z_target)
power_target = 1 - beta_target

print(f"при mu = {mu_target} пКи/г:")
print(f"z = ({Lc_given} - {mu_target}) / {sigma} = {z_target:.3f}")
print(f"beta = Phi({z_target:.3f}) = {beta_target:.4f}")
print(f"мощность = {power_target:.4f}")
