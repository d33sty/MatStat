import numpy as np
import scipy.stats as stats

# Исходные данные

t_b = 4  # время измерения фона, мин
t_s = 16  # время измерения образца + фон, мин
N_b = 120  # фоновые импульсы
N_sb = 584  # полные импульсы (фон + образец)
k = 2.36  # калибровочная постоянная, Бк / (имп/мин)

print("Задача: радиоактивный счет (фон + образец)")
print(f"время измерения фона: t_b = {t_b} мин")
print(f"время измерения образца+фон: t_s = {t_s} мин")
print(f"фоновые импульсы: N_b = {N_b}")
print(f"полные импульсы: N_sb = {N_sb}")
print(f"калибровочная постоянная: k = {k} Бк/(имп/мин)")
print()

# Часть (a) - Оценка скорости чистого счета и его ско

print("Часть (a)")

# скорости счета
R_b = N_b / t_b  # скорость фона, имп/мин
R_sb = N_sb / t_s  # скорость полного счета, имп/мин
R_s = R_sb - R_b  # скорость чистого счета, имп/мин

print(f"скорость фона: R_b = {R_b:.2f} имп/мин")
print(f"скорость полного счета: R_sb = {R_sb:.2f} имп/мин")
print(f"скорость чистого счета: R_s = {R_s:.2f} имп/мин")

# дисперсии числа импульсов (пуассоновские)
var_Nb = N_b
var_Nsb = N_sb

# дисперсии скоростей счета
var_Rb = var_Nb / t_b**2
var_Rsb = var_Nsb / t_s**2

# дисперсия и ско скорости чистого счета
var_Rs = var_Rsb + var_Rb
sigma_Rs = np.sqrt(var_Rs)

print(f"дисперсия фона: var[N_b] = {var_Nb}")
print(f"дисперсия полного счета: var[N_sb] = {var_Nsb}")
print(f"дисперсия скорости фона: var[R_b] = {var_Rb:.4f}")
print(f"дисперсия скорости полного счета: var[R_sb] = {var_Rsb:.4f}")
print(f"дисперсия скорости чистого счета: var[R_s] = {var_Rs:.4f}")
print(f"ско скорости чистого счета: sigma_Rs = {sigma_Rs:.4f} имп/мин")
print(f"R_s = {R_s:.2f} +- {sigma_Rs:.2f} имп/мин")
print()

# Часть (b) - Активность образца

print("Часть (b)")

A = k * R_s

print(f"активность образца: A = k * R_s = {k} * {R_s:.2f} = {A:.2f} Бк")
print()

# Часть (c) - Вероятность превышения при нулевой активности

print("Часть (c)")

# при At = 0 используем оценку фона mu_b = N_b
mu_b_est = N_b

# дисперсия чистого счета при At = 0
var_Rs_at0 = mu_b_est / t_b**2 + (4 * mu_b_est) / t_s**2
sigma_Rs_at0 = np.sqrt(var_Rs_at0)

# нормированное отклонение
z = R_s / sigma_Rs_at0

# вероятность превышения (односторонняя)
p_value = 1 - stats.norm.cdf(z)

print(f"оценка среднего фона: mu_b = {mu_b_est}")
print(f"дисперсия чистого счета при At=0: var[R_s] = {var_Rs_at0:.4f}")
print(f"ско чистого счета при At=0: sigma_Rs(0) = {sigma_Rs_at0:.4f} имп/мин")
print(f"нормированное отклонение: z = {z:.3f}")
print(f"вероятность превышения: p = {p_value:.4f} ({p_value*100:.2f}%)")

print()

# Часть (d) - Порог принятия решения Lc за 4 минуты

print("Часть (d)")

# при At=0 чистый счет за 4 мин: C = N_sb4 - Nb
# дисперсия C: var[C] = mu_b + mu_b = 2*mu_b
mu_b_est = N_b
sigma_C = np.sqrt(2 * mu_b_est)

# квантили нормального распределения
z_05 = stats.norm.ppf(0.95)  # для alpha = 0.05 (односторонний)
z_10 = stats.norm.ppf(0.90)  # для alpha = 0.10

Lc_05 = z_05 * sigma_C
Lc_10 = z_10 * sigma_C

print(f"оценка среднего фона: mu_b = {mu_b_est}")
print(f"ско чистого счета за 4 мин: sigma_C = sqrt(2*mu_b) = {sigma_C:.2f}")
print(f"квантиль для alpha=0.05: z_0.95 = {z_05:.3f}")
print(f"квантиль для alpha=0.10: z_0.90 = {z_10:.3f}")
print(f"Lc (alpha=0.05) = {z_05:.3f} * {sigma_C:.2f} = {Lc_05:.2f}")
print(f"Lc (alpha=0.10) = {z_10:.3f} * {sigma_C:.2f} = {Lc_10:.2f}")
print()

# Часть (e) - Минимальная значимая активность

print("Часть (e)")

t_measure = 4  # время измерения для Lc, мин

A_min_05 = k * (Lc_05 / t_measure)
A_min_10 = k * (Lc_10 / t_measure)

print(f"минимальная активность для alpha=0.05:")
print(f"  A_min = {k} * ({Lc_05:.2f} / {t_measure}) = {A_min_05:.2f} Бк")
print(f"минимальная активность для alpha=0.10:")
print(f"  A_min = {k} * ({Lc_10:.2f} / {t_measure}) = {A_min_10:.2f} Бк")
print(f"A_min(0.05) = {A_min_05:.2f} Бк, A_min(0.10) = {A_min_10:.2f} Бк")
print()
