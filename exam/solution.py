import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# директория, в которой лежит этот файл чтобы сохранить туда графики
_dir = os.path.dirname(os.path.abspath(__file__))

# Масса U, г
U = np.array([15, 20, 30, 40, 50, 60], dtype=float)

# Кол-во импульсов
N = np.array([1305, 1457, 2380, 3074, 3615, 4420], dtype=float)

n = len(U)


# I. Линейный МНК
# Модель: N = a*U + b

print("I. Линейный МНК")
print()

# a. Оценка параметров линейной зависимости двумя способами
# Способ 1: scipy.stats.linregress
# linregress возвращает наклон (slope=a), свободный член (intercept=b),
# коэффициент корреляции r, p-value и стандартную ошибку наклона
slope, intercept, r_value, p_value, std_err = stats.linregress(U, N)

# Способ 2: матричная форма МНК
# Матрица плана X: каждая строка [1, U_i] для модели N = b + a*U
# theta = (X^T * X)^(-1) * X^T * N
X = np.column_stack((np.ones(n), U))
theta = np.linalg.inv(X.T @ X) @ (X.T @ N)
b_m, a_m = theta[0], theta[1]

# Используем матричные оценки для дальнейших расчётов
a, b = a_m, b_m

# Предсказанные значения и остатки
N_pred = a * U + b
residuals = N - N_pred

# Несмещённая оценка дисперсии остатков: S^2 = sum(r_i^2) / (n - p), p=2 параметра
p = 2
S2 = np.sum(residuals**2) / (n - p)
S = np.sqrt(S2)

print("I.a. Построить линейную зависимость количества импульсов от массы U.")
print("Оценить параметры линейной зависимости двумя способами.")
print()
print("Способ 1 (linregress):")
print(f"a = {slope:.2f} имп/г")
print(f"b = {intercept:.2f} имп")
print(f"r = {r_value:.4f}")
print(f"r^2 = {r_value**2:.4f}")
print()
print("Способ 2 (матричная форма):")
print(f"a = {a_m:.2f} имп/г")
print(f"b = {b_m:.2f} имп")
print()
print(f"Стандартная ошибка регрессии: S = {S:.2f} имп")
print()

print("I.b. Экспериментальные данные и подогнанную зависимость представить на графике.")
print(f"График 1 (I. N(U)): экспериментальные данные и прямая N = {a:.2f}*U + {b:.2f}")
print()

# c. Дисперсия параметров
# Ковариационная матрица Сov = S^2 * (X^T * X)^(-1)
# Диагональные элементы — дисперсии b и a соответственно
Cov = S2 * np.linalg.inv(X.T @ X)
S2_b = Cov[0, 0]
S2_a = Cov[1, 1]
S_b = np.sqrt(S2_b)
S_a = np.sqrt(S2_a)

print("I.c. Оценить дисперсию полученных параметров.")
print(f"D(a) = {S2_a:.2f} (имп/г)^2")
print(f"D(b) = {S2_b:.2f} имп^2")
print(f"S(a) = {S_a:.2f} имп/г")
print(f"S(b) = {S_b:.2f} имп")
print()

# -d. 95% доверительный интервал для среднего значения N при U0 = 35 г
U0 = 35.0
x0 = np.array([1.0, U0])  # вектор-строка для новой точки

# Точечная оценка среднего при U0
N0 = a * U0 + b

# Дисперсия оценки среднего: D(N0) = S^2 * x0^T * (X^T*X)^(-1) * x0
# Это дисперсия линейной комбинации параметров x0^T * theta
var_N0 = S2 * (x0 @ np.linalg.inv(X.T @ X) @ x0)
S_N0 = np.sqrt(var_N0)

# Критическое значение t-распределения: df = n - p = 4, двусторонний alpha=0.05
t_cr = stats.t.ppf(0.975, df=n - p)

delta = t_cr * S_N0
ci_lower = N0 - delta
ci_upper = N0 + delta

print(
    "I.d. Построить 95% доверительный интервал для количества импульсов при массе U = 35 г."
)
print(f"Точечная оценка: N(35) = {N0:.2f} имп")
print(f"D(N0) = {var_N0:.2f}, S(N0) = {S_N0:.2f} имп")
print(f"t_кр (df={n-p}, p=0.95) = {t_cr:.2f}")
print(f"delta = {delta:.2f} имп")
print(f"95% ДИ: [{ci_lower:.2f}, {ci_upper:.2f}] имп")
print()

# График
U_plot = np.linspace(10, 65, 200)
N_plot = a * U_plot + b

plt.figure(1, figsize=(9, 6))
plt.scatter(U, N, color="blue", s=60, label="экспериментальные данные")
plt.plot(U_plot, N_plot, "r-", lw=2, label=f"N = {a:.2f}*U + {b:.2f}")
# точка прогноза I.d и её 95% ДИ
plt.errorbar(
    U0,
    N0,
    yerr=delta,
    fmt="go",
    capsize=6,
    markersize=7,
    label=f"I.d: N({U0:.0f}) = {N0:.0f}, ДИ = [{ci_lower:.0f}, {ci_upper:.0f}]",
)
plt.xlabel("масса U, г")
plt.ylabel("количество импульсов")
plt.title("График 1. I. Линейный МНК: N(U) — количество импульсов от массы U")
plt.legend()
plt.grid(True, alpha=0.3)


# II. Линейный МНК (инверсное решение)
# Модель: U = c*N + d


print("II. Линейный МНК (инверсное решение)")
print()

# a. Оценка параметров двумя способами

# Способ 1: scipy.stats.linregress (меняем местами аргументы: x=N, y=U)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(N, U)

# Способ 2: матричная форма МНК
# Матрица плана X2: каждая строка [1, N_i] для модели U = d + c*N
X2 = np.column_stack((np.ones(n), N))
theta2 = np.linalg.inv(X2.T @ X2) @ (X2.T @ U)
d_m, c_m = theta2[0], theta2[1]

c, d = c_m, d_m

# Остатки и дисперсия
U_pred = c * N + d
residuals2 = U - U_pred
p2 = 2
S2_2 = np.sum(residuals2**2) / (n - p2)
S2_reg = np.sqrt(S2_2)

print("II.a. Построить линейную зависимость массы U от количества импульсов.")
print("Оценить параметры линейной зависимости двумя способами.")
print()
print("Способ 1 (linregress):")
print(f"c = {slope2:.4f} г/имп")
print(f"d = {intercept2:.2f} г")
print(f"r = {r_value2:.4f}")
print(f"r^2 = {r_value2**2:.4f}")
print()
print("Способ 2 (матричная форма):")
print(f"c = {c_m:.4f} г/имп")
print(f"d = {d_m:.2f} г")
print()
print(f"Стандартная ошибка регрессии: S = {S2_reg:.2f} г")
print()

print(
    "II.b. Экспериментальные данные и подогнанную зависимость представить на графике."
)
print(f"График 2 (II. U(N)): экспериментальные данные и прямая U = {c:.4f}*N + {d:.2f}")
print()

#   c. Дисперсия параметров инверсной регрессии
# Ковариационная матрица: Cov2 = S2_2 * (X2^T * X2)^(-1)
Cov2 = S2_2 * np.linalg.inv(X2.T @ X2)
S2_d = Cov2[0, 0]
S2_c = Cov2[1, 1]
S_d = np.sqrt(S2_d)
S_c = np.sqrt(S2_c)

print("II.c. Оценить дисперсии полученных параметров.")
print(f"D(c) = {S2_c:.3e} (г/имп)^2")
print(f"D(d) = {S2_d:.2f} г^2")
print(f"S(c) = {S_c:.3e} г/имп")
print(f"S(d) = {S_d:.2f} г")
print()

#   d. 95% доверительный интервал для среднего значения U при N0 = 2500 имп
N0_2 = 2500.0
x0_2 = np.array([1.0, N0_2])

# Точечная оценка
U0_2 = c * N0_2 + d

# Дисперсия оценки среднего: D(U0) = S2_2 * x0^T * (X2^T*X2)^(-1) * x0
var_U0 = S2_2 * (x0_2 @ np.linalg.inv(X2.T @ X2) @ x0_2)
S_U0 = np.sqrt(var_U0)

# df = n - p2 = 4, двусторонний alpha=0.05
t_cr2 = stats.t.ppf(0.975, df=n - p2)

delta2 = t_cr2 * S_U0
ci2_lower = U0_2 - delta2
ci2_upper = U0_2 + delta2

print("II.d. Построить 95% доверительный интервал для массы урана при N = 2500 имп.")
print(f"Точечная оценка: U(2500) = {U0_2:.2f} г")
print(f"D(U0) = {var_U0:.2f}, S(U0) = {S_U0:.2f} г")
print(f"t_кр (df={n-p2}, p=0.95) = {t_cr2:.2f}")
print(f"delta = {delta2:.2f} г")
print(f"95% ДИ: [{ci2_lower:.2f}, {ci2_upper:.2f}] г")
print()

N_plot2 = np.linspace(1200, 4600, 200)
U_plot2 = c * N_plot2 + d

plt.figure(2, figsize=(9, 6))
plt.scatter(N, U, color="blue", s=60, label="экспериментальные данные")
plt.plot(N_plot2, U_plot2, "r-", lw=2, label=f"U = {c:.4f}*N + {d:.2f}")
# точка прогноза II.d и её 95% ДИ
plt.errorbar(
    N0_2,
    U0_2,
    yerr=delta2,
    fmt="go",
    capsize=6,
    markersize=7,
    label=f"II.d: U({N0_2:.0f}) = {U0_2:.2f}, ДИ = [{ci2_lower:.2f}, {ci2_upper:.2f}]",
)
plt.xlabel("количество импульсов")
plt.ylabel("масса U, г")
plt.title(
    "График 2. II. Линейный МНК (инверсное): U(N) — масса U от количества импульсов"
)
plt.legend()
plt.grid(True, alpha=0.3)


# III. Сравнение решений с разными факторами


print("III. Сравнение решений с разными факторами")
print()

#   a. Обе зависимости на одном рисунке
# Регрессия I: N = a*U + b (прямая)
# Регрессия II: U = c*N + d, инвертируем -> N = (U - d) / c
# чтобы обе кривые жили в одних осях (U по x, N по y)
N_from_II = (U_plot - d) / c

print("III.a. Построить графики зависимостей из заданий I и II на одном рисунке.")
print(f"График 3: I — N = {a:.2f}*U + {b:.2f}")
print(f"          II (инв.) — N = (U - ({d:.2f})) / {c:.4f}")
print()

plt.figure(3, figsize=(9, 6))
plt.scatter(U, N, color="blue", s=60, label="экспериментальные данные")
plt.plot(U_plot, N_plot, "r-", lw=2, label=f"I: N = {a:.2f}*U + {b:.2f}")
plt.plot(
    U_plot, N_from_II, "g--", lw=2, label=f"II (инв.): N = (U - {d:.2f}) / {c:.4f}"
)
plt.xlabel("масса U, г")
plt.ylabel("количество импульсов")
plt.title("График 3. III. Сравнение регрессий I и II")
plt.legend()
plt.grid(True, alpha=0.3)

#   b. Графики одинаковые?
# Наклон прямой I: a = Cov(U,N) / Var(U)
# Наклон инвертированной II: 1/c = Var(N) / Cov(U,N)
# Произведение наклонов равно r^2: a * (1/c) = r^2
# Чем ближе r к 1, тем ближе оба наклона друг к другу и тем больше прямые совпадают.
# При r = 1 прямые идентичны.
slope_I = a
slope_II_inv = 1.0 / c
r2_check = slope_I * c  # должно быть равно r^2

print("III.b. Графики одинаковые?")
print(f"Наклон прямой I:              a = {slope_I:.4f}")
print(f"Наклон инвертированной II: 1/c = {slope_II_inv:.4f}")
print(
    f"Произведение наклонов a*(1/c) = r^2: {r2_check:.4f} (r^2 из I = {r_value**2:.4f})"
)
print(f"r = {r_value:.4f} — близко к 1, поэтому прямые визуально совпадают.")
print("При r = 1 прямые идентичны; чем меньше r, тем сильнее расхождение.")
print()

#   c. Связь между параметрами I и II
# Связь 1: произведение наклонов равно r^2
#   a * c = r^2
# Связь 2: обе прямые проходят через точку средних (mean_U, mean_N),
#   что является фундаментальным свойством МНК
# Связь 3: свободные члены выражаются через средние и наклоны:
#   b = mean_N - a * mean_U  (из I)
#   d = mean_U - c * mean_N  (из II)
mean_U = np.mean(U)
mean_N = np.mean(N)

b_check = mean_N - a * mean_U  # должно совпасть с b
d_check = mean_U - c * mean_N  # должно совпасть с d
ac_product = a * c  # должно совпасть с r^2

print("III.c. Есть ли связь между параметрами, найденными в I и II?")
print(f"a * c = {ac_product:.4f},  r^2 = {r_value**2:.4f}  (должны совпадать)")
print(
    f"Обе прямые проходят через точку средних: (mean_U, mean_N) = ({mean_U:.1f}, {mean_N:.1f})"
)
print(f"b = mean_N - a*mean_U = {b_check:.2f}  (b из I = {b:.2f})")
print(f"d = mean_U - c*mean_N = {d_check:.2f}  (d из II = {d:.2f})")
print("Вывод: a*c = r^2; свободные члены однозначно определяются наклонами и средними.")
print()

#   d. Проверка гипотез о существовании зависимости
# H0: наклон = 0 (линейной зависимости нет)
# H1: наклон != 0 (зависимость существует)
# t-статистика: t = коэффициент / S(коэффициент), df = n - p
# Если |t| > t_кр, H0 отвергается

alpha_hyp = 0.05
t_cr_hyp = stats.t.ppf(0.975, df=n - p)

# Для регрессии I: проверяем значимость a
t_a = a / S_a
p_a = 2 * (1 - stats.t.cdf(abs(t_a), df=n - p))

# Для регрессии II: проверяем значимость c
t_c = c / S_c
p_c = 2 * (1 - stats.t.cdf(abs(t_c), df=n - p2))

print("III.d. Проверить гипотезы о существовании зависимости между параметрами I и II.")
print(f"t_кр (df={n-p}, alpha={alpha_hyp}) = {t_cr_hyp:.2f}")
print()
print("Регрессия I (H0: a = 0):")
print(f"t = a / S(a) = {a:.2f} / {S_a:.2f} = {t_a:.2f}")
print(f"p-value = {p_a:.4f}")
if abs(t_a) > t_cr_hyp:
    print(
        f"|t| = {abs(t_a):.2f} > t_кр = {t_cr_hyp:.2f} -> H0 отвергается, зависимость существует"
    )
else:
    print(f"|t| = {abs(t_a):.2f} <= t_кр = {t_cr_hyp:.2f} -> H0 не отвергается")
print()
print("Регрессия II (H0: c = 0):")
print(f"t = c / S(c) = {c:.4f} / {S_c:.3e} = {t_c:.2f}")
print(f"p-value = {p_c:.4f}")
if abs(t_c) > t_cr_hyp:
    print(
        f"|t| = {abs(t_c):.2f} > t_кр = {t_cr_hyp:.2f} -> H0 отвергается, зависимость существует"
    )
else:
    print(f"|t| = {abs(t_c):.2f} <= t_кр = {t_cr_hyp:.2f} -> H0 не отвергается")
print()


# IV. Взвешенный МНК


print("IV. Взвешенный МНК")
print()

# Веса для регрессии I: N имеет распределение Пуассона -> var(N_i) = N_i -> w_i = 1/N_i
w_N = 1.0 / N
W_N = np.diag(w_N)

# Веса для регрессии II: U измерена с относительной погрешностью 15%
# -> sigma_U_i = 0.15 * U_i -> var(U_i) = (0.15*U_i)^2 -> w_i = 1/(0.15*U_i)^2
w_U = 1.0 / (0.15 * U) ** 2
W_U = np.diag(w_U)


# IV.I. Взвешенный МНК: N = a_w*U + b_w (веса по Пуассону)

print("IV.I. Взвешенный МНК (N = a_w*U + b_w, веса w_i = 1/N_i)")
print()

# Способ 1: polyfit с весами
slope_w1, intercept_w1 = np.polyfit(U, N, 1, w=w_N)

# Способ 2: матричная форма взвешенного МНК
# theta = (X^T * W * X)^(-1) * X^T * W * y
theta_w1 = np.linalg.inv(X.T @ W_N @ X) @ (X.T @ W_N @ N)
b_w1, a_w1 = theta_w1[0], theta_w1[1]

# Взвешенные остатки и дисперсия: S^2 = sum(w_i * r_i^2) / (n - p)
N_pred_w1 = a_w1 * U + b_w1
res_w1 = N - N_pred_w1
S2_w1 = np.sum(w_N * res_w1**2) / (n - p)
S_w1 = np.sqrt(S2_w1)

print("IV.I.a. Оценить параметры двумя способами.")
print()
print("Способ 1 (polyfit с весами):")
print(f"a_w = {slope_w1:.2f} имп/г")
print(f"b_w = {intercept_w1:.2f} имп")
print()
print("Способ 2 (матричная форма):")
print(f"a_w = {a_w1:.2f} имп/г")
print(f"b_w = {b_w1:.2f} имп")
print()
print(f"Стандартная ошибка регрессии (взвешенная): S = {S_w1:.2f}")
print()

print("IV.I.b. График зависимости представлен на графике 4.")
print(
    f"График 4 (IV.I): экспериментальные данные и прямая N = {a_w1:.2f}*U + {b_w1:.2f}"
)
print()

# Ковариационная матрица взвешенных оценок
Cov_w1 = S2_w1 * np.linalg.inv(X.T @ W_N @ X)
S2_a_w1 = Cov_w1[1, 1]
S2_b_w1 = Cov_w1[0, 0]
S_a_w1 = np.sqrt(S2_a_w1)
S_b_w1 = np.sqrt(S2_b_w1)

print("IV.I.c. Дисперсии параметров.")
print(f"D(a_w) = {S2_a_w1:.2f} (имп/г)^2")
print(f"D(b_w) = {S2_b_w1:.2f} имп^2")
print(f"S(a_w) = {S_a_w1:.2f} имп/г")
print(f"S(b_w) = {S_b_w1:.2f} имп")
print()

# 95% ДИ при U0 = 35 г
x0_w1 = np.array([1.0, U0])
N0_w1 = a_w1 * U0 + b_w1
var_N0_w1 = S2_w1 * (x0_w1 @ np.linalg.inv(X.T @ W_N @ X) @ x0_w1)
S_N0_w1 = np.sqrt(var_N0_w1)
t_cr_w1 = stats.t.ppf(0.975, df=n - p)
delta_w1 = t_cr_w1 * S_N0_w1
ci_w1_lower = N0_w1 - delta_w1
ci_w1_upper = N0_w1 + delta_w1

print("IV.I.d. 95% доверительный интервал для количества импульсов при массе U = 35 г.")
print(f"Точечная оценка: N_w(35) = {N0_w1:.2f} имп")
print(f"D(N0) = {var_N0_w1:.2f}, S(N0) = {S_N0_w1:.2f} имп")
print(f"t_кр (df={n-p}, p=0.95) = {t_cr_w1:.2f}")
print(f"delta = {delta_w1:.2f} имп")
print(f"95% ДИ: [{ci_w1_lower:.2f}, {ci_w1_upper:.2f}] имп")
print()

U_plot_w = np.linspace(10, 65, 200)
N_plot_w1 = a_w1 * U_plot_w + b_w1

plt.figure(4, figsize=(9, 6))
plt.scatter(U, N, color="blue", s=60, label="экспериментальные данные")
plt.plot(U_plot_w, N_plot_w1, "r-", lw=2, label=f"N = {a_w1:.2f}*U + {b_w1:.2f}")
plt.errorbar(
    U0,
    N0_w1,
    yerr=delta_w1,
    fmt="go",
    capsize=6,
    markersize=7,
    label=f"IV.I.d: N({U0:.0f}) = {N0_w1:.0f}, ДИ = [{ci_w1_lower:.0f}, {ci_w1_upper:.0f}]",
)
plt.xlabel("масса U, г")
plt.ylabel("количество импульсов")
plt.title("График 4. IV.I. Взвешенный МНК N(U), веса 1/N_i (Пуассон)")
plt.legend()
plt.grid(True, alpha=0.3)


# IV.II. Взвешенный МНК: U = c_w*N + d_w (веса по погрешности 15%)
print("IV.II. Взвешенный МНК (U = c_w*N + d_w, веса w_i = 1/(0.15*U_i)^2)")
print()

# Способ 1: polyfit с весами
slope_w2, intercept_w2 = np.polyfit(N, U, 1, w=w_U)

# Способ 2: матричная форма
theta_w2 = np.linalg.inv(X2.T @ W_U @ X2) @ (X2.T @ W_U @ U)
d_w2, c_w2 = theta_w2[0], theta_w2[1]

U_pred_w2 = c_w2 * N + d_w2
res_w2 = U - U_pred_w2
S2_w2 = np.sum(w_U * res_w2**2) / (n - p2)
S_w2 = np.sqrt(S2_w2)

print("IV.II.a. Оценить параметры двумя способами.")
print()
print("Способ 1 (polyfit с весами):")
print(f"c_w = {slope_w2:.4f} г/имп")
print(f"d_w = {intercept_w2:.2f} г")
print()
print("Способ 2 (матричная форма):")
print(f"c_w = {c_w2:.4f} г/имп")
print(f"d_w = {d_w2:.2f} г")
print()
print(f"Стандартная ошибка регрессии (взвешенная): S = {S_w2:.2f}")
print()

print("IV.II.b. График зависимости представлен на графике 5.")
print(
    f"График 5 (IV.II): экспериментальные данные и прямая U = {c_w2:.4f}*N + {d_w2:.2f}"
)
print()

Cov_w2 = S2_w2 * np.linalg.inv(X2.T @ W_U @ X2)
S2_c_w2 = Cov_w2[1, 1]
S2_d_w2 = Cov_w2[0, 0]
S_c_w2 = np.sqrt(S2_c_w2)
S_d_w2 = np.sqrt(S2_d_w2)

print("IV.II.c. Дисперсии параметров.")
print(f"D(c_w) = {S2_c_w2:.3e} (г/имп)^2")
print(f"D(d_w) = {S2_d_w2:.2f} г^2")
print(f"S(c_w) = {S_c_w2:.3e} г/имп")
print(f"S(d_w) = {S_d_w2:.2f} г")
print()

x0_w2 = np.array([1.0, N0_2])
U0_w2 = c_w2 * N0_2 + d_w2
var_U0_w2 = S2_w2 * (x0_w2 @ np.linalg.inv(X2.T @ W_U @ X2) @ x0_w2)
S_U0_w2 = np.sqrt(var_U0_w2)
t_cr_w2 = stats.t.ppf(0.975, df=n - p2)
delta_w2 = t_cr_w2 * S_U0_w2
ci_w2_lower = U0_w2 - delta_w2
ci_w2_upper = U0_w2 + delta_w2

print("IV.II.d. 95% доверительный интервал для массы урана при N = 2500 имп.")
print(f"Точечная оценка: U_w(2500) = {U0_w2:.2f} г")
print(f"D(U0) = {var_U0_w2:.2f}, S(U0) = {S_U0_w2:.2f} г")
print(f"t_кр (df={n-p2}, p=0.95) = {t_cr_w2:.2f}")
print(f"delta = {delta_w2:.2f} г")
print(f"95% ДИ: [{ci_w2_lower:.2f}, {ci_w2_upper:.2f}] г")
print()

N_plot_w2 = np.linspace(1200, 4600, 200)
U_plot_w2 = c_w2 * N_plot_w2 + d_w2

plt.figure(5, figsize=(9, 6))
plt.scatter(N, U, color="blue", s=60, label="экспериментальные данные")
plt.plot(N_plot_w2, U_plot_w2, "r-", lw=2, label=f"U = {c_w2:.4f}*N + {d_w2:.2f}")
plt.errorbar(
    N0_2,
    U0_w2,
    yerr=delta_w2,
    fmt="go",
    capsize=6,
    markersize=7,
    label=f"IV.II.d: U({N0_2:.0f}) = {U0_w2:.2f}, ДИ = [{ci_w2_lower:.2f}, {ci_w2_upper:.2f}]",
)
plt.xlabel("количество импульсов")
plt.ylabel("масса U, г")
plt.title("График 5. IV.II. Взвешенный МНК U(N), веса 1/(0.15*U_i)^2")
plt.legend()
plt.grid(True, alpha=0.3)


# IV.III. Сравнение взвешенных решений

print("IV.III. Сравнение взвешенных решений")
print()

# Инвертированная IV.II для совместного графика в осях (U, N)
N_from_w2 = (U_plot_w - d_w2) / c_w2

print("IV.III.a. Графики IV.I и IV.II на одном рисунке (график 6).")
print(f"График 6 (IV.III): I_w — N = {a_w1:.2f}*U + {b_w1:.2f}")
print(f"                   II_w (инв.) — N = (U - {d_w2:.2f}) / {c_w2:.4f}")
print()

plt.figure(6, figsize=(9, 6))
plt.scatter(U, N, color="blue", s=60, label="экспериментальные данные")
plt.plot(U_plot_w, N_plot_w1, "r-", lw=2, label=f"IV.I: N = {a_w1:.2f}*U + {b_w1:.2f}")
plt.plot(
    U_plot_w,
    N_from_w2,
    "g--",
    lw=2,
    label=f"IV.II (инв.): N = (U - {d_w2:.2f}) / {c_w2:.4f}",
)
plt.xlabel("масса U, г")
plt.ylabel("количество импульсов")
plt.title("График 6. IV.III. Сравнение взвешенных регрессий I_w и II_w")
plt.legend()
plt.grid(True, alpha=0.3)

# IV.III.b — визуальное сравнение
ac_w = a_w1 * c_w2
print("IV.III.b. Графики одинаковые?")
print(f"a_w * c_w = {ac_w:.4f}")
print("Если близко к 1 — прямые совпадают. Отличие от невзвешенного случая")
print("обусловлено перераспределением влияния точек согласно их весам.")
print()

# IV.III.c — связь параметров
mean_U_w = np.average(U, weights=w_N)  # взвешенное среднее U по весам IV.I
mean_N_w = np.average(N, weights=w_U)  # взвешенное среднее N по весам IV.II
b_w1_check = np.average(N, weights=w_N) - a_w1 * np.average(U, weights=w_N)
d_w2_check = np.average(U, weights=w_U) - c_w2 * np.average(N, weights=w_U)
ac_w_product = a_w1 * c_w2

print("IV.III.c. Связь между параметрами IV.I и IV.II.")
print(f"a_w * c_w = {ac_w_product:.4f}")
print(f"b_w проверка: {b_w1_check:.2f} (b_w из IV.I = {b_w1:.2f})")
print(f"d_w проверка: {d_w2_check:.2f} (d_w из IV.II = {d_w2:.2f})")
print("Связь a_w*c_w = r^2 в общем случае не выполняется точно,")
print("так как каждая регрессия использует свои веса.")
print()

# IV.III.d — проверка гипотез
t_a_w = a_w1 / S_a_w1
p_a_w = 2 * (1 - stats.t.cdf(abs(t_a_w), df=n - p))
t_c_w = c_w2 / S_c_w2
p_c_w = 2 * (1 - stats.t.cdf(abs(t_c_w), df=n - p2))

print("IV.III.d. Проверка гипотез о существовании зависимости.")
print(f"t_кр (df={n-p}, alpha=0.05) = {t_cr_hyp:.2f}")
print()
print("IV.I (H0: a_w = 0):")
print(f"t = {t_a_w:.2f}, p-value = {p_a_w:.4f}")
if abs(t_a_w) > t_cr_hyp:
    print(f"|t| = {abs(t_a_w):.2f} > t_кр -> H0 отвергается, зависимость существует")
else:
    print(f"|t| = {abs(t_a_w):.2f} <= t_кр -> H0 не отвергается")
print()
print("IV.II (H0: c_w = 0):")
print(f"t = {t_c_w:.2f}, p-value = {p_c_w:.4f}")
if abs(t_c_w) > t_cr_hyp:
    print(f"|t| = {abs(t_c_w):.2f} > t_кр -> H0 отвергается, зависимость существует")
else:
    print(f"|t| = {abs(t_c_w):.2f} <= t_кр -> H0 не отвергается")
print()


# IV.b. Биссектриса угла двух зависимостей

# Наклон биссектрисы: угол биссектрисы = среднее углов двух прямых
# theta1 = arctan(a_w1), theta2 = arctan(1/c_w2)
# slope_bis = tan((theta1 + theta2) / 2)
# Точка пересечения двух прямых: линия I: N = a_w1*U + b_w1
#                                 линия II (инв.): N = U/c_w2 - d_w2/c_w2
# Пересечение: a_w1*U + b_w1 = U/c_w2 - d_w2/c_w2
# U_int = (-d_w2/c_w2 - b_w1) / (a_w1 - 1/c_w2)
theta1 = np.arctan(a_w1)
theta2 = np.arctan(1.0 / c_w2)
slope_bis = np.tan((theta1 + theta2) / 2.0)

slope_II_inv_w = 1.0 / c_w2
intercept_II_inv_w = -d_w2 / c_w2
U_int = (intercept_II_inv_w - b_w1) / (a_w1 - slope_II_inv_w)
N_int = a_w1 * U_int + b_w1
intercept_bis = N_int - slope_bis * U_int

N_bis = slope_bis * U_plot_w + intercept_bis

print("IV.b. Построить зависимость как биссектриссу угла двух зависимостей.")
print("Нанести на общий график (график 6).")
print()
print(
    f"Угол прямой IV.I:         theta1 = arctan({a_w1:.4f}) = {np.degrees(theta1):.2f} градусов"
)
print(
    f"Угол прямой IV.II (инв.): theta2 = arctan({slope_II_inv_w:.4f}) = {np.degrees(theta2):.2f} градусов"
)
print(
    f"Угол биссектрисы: (theta1 + theta2) / 2 = {np.degrees((theta1+theta2)/2):.2f} градусов"
)
print(f"Наклон биссектрисы: slope_bis = {slope_bis:.2f} имп/г")
print(f"Точка пересечения двух прямых: U = {U_int:.2f} г, N = {N_int:.2f} имп")
print(f"Биссектриса: N = {slope_bis:.2f}*U + {intercept_bis:.2f}")
print()

#   IV.c. 95% ДИ для N при U0 = 35 г по биссектрисе
# Точечная оценка по биссектрисе
N0_bis = slope_bis * U0 + intercept_bis

# Дисперсия: остатки биссектрисы относительно исходных данных,
# матрица плана X та же, что в регрессии I (отклик N, предиктор U)
res_bis = N - (slope_bis * U + intercept_bis)
S2_bis = np.sum(res_bis**2) / (n - p)
S_bis = np.sqrt(S2_bis)

x0_bis = np.array([1.0, U0])
var_N0_bis = S2_bis * (x0_bis @ np.linalg.inv(X.T @ X) @ x0_bis)
S_N0_bis = np.sqrt(var_N0_bis)

t_cr_bis = stats.t.ppf(0.975, df=n - p)
delta_bis = t_cr_bis * S_N0_bis
ci_bis_lower = N0_bis - delta_bis
ci_bis_upper = N0_bis + delta_bis

print(
    "IV.c. Построить 95% доверительный интервал для количества импульсов при U = 35 г (биссектриса)."
)
print(f"Точечная оценка: N_bis(35) = {N0_bis:.2f} имп")
print(f"S остатков биссектрисы: S = {S_bis:.2f} имп")
print(f"D(N0) = {var_N0_bis:.2f}, S(N0) = {S_N0_bis:.2f} имп")
print(f"t_кр (df={n-p}, p=0.95) = {t_cr_bis:.2f}")
print(f"delta = {delta_bis:.2f} имп")
print(f"95% ДИ: [{ci_bis_lower:.2f}, {ci_bis_upper:.2f}] имп")
print()

# Добавляем биссектрису и ДИ на график 6
plt.figure(6)
plt.plot(
    U_plot_w,
    N_bis,
    "m-",
    lw=2,
    label=f"биссектриса: N = {slope_bis:.2f}*U + {intercept_bis:.2f}",
)
plt.errorbar(
    U0,
    N0_bis,
    yerr=delta_bis,
    fmt="ms",
    capsize=6,
    markersize=7,
    label=f"IV.c: N_bis({U0:.0f}) = {N0_bis:.0f}, ДИ = [{ci_bis_lower:.0f}, {ci_bis_upper:.0f}]",
)
plt.legend()


# Сохранение и вывод всех графиков

for fig_num in [1, 2, 3, 4, 5, 6]:
    plt.figure(fig_num)
    plt.savefig(
        os.path.join(_dir, f"graph_{fig_num}.png"), dpi=150, bbox_inches="tight"
    )

plt.show()
