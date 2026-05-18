import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# исходные данные: номер канала и энергия частицы в кэВ
channel = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970])
energy = np.array([75.99, 91.97, 105.71, 123.20, 131.67, 150.70, 179.32, 203.21])

# число наблюдений и число параметров модели (a, b, c)
n = len(channel)
p = 3

print("Экспериментальные данные")
print(f"номер канала: {channel}")
print(f"энергия, кэВ: {energy}")
print()

# 1. Нелинейный МНК (квадратичная модель)
print("1. Нелинейный МНК (квадратичная модель E = a*k^2 + b*k + c)")
print()

# матрица плана X: каждая строка содержит [k^2, k, 1] для одного наблюдения
X = np.column_stack((channel**2, channel, np.ones(n)))

# МНК-оценка вектора параметров: theta = (X^T * X)^(-1) * X^T * y
theta = np.linalg.inv(X.T @ X) @ (X.T @ energy)
a, b, c = theta

print("Оцененные параметры:")
print(f"a = {a:.6f} кэВ/кан^2")
print(f"b = {b:.4f} кэВ/кан")
print(f"c = {c:.2f} кэВ")
print()

# предсказанные значения энергии по найденным коэффициентам
energy_pred = a * channel**2 + b * channel + c

# остатки регрессии: r_i = y_i - y_pred_i
residuals = energy - energy_pred

# несмещённая оценка дисперсии остатков: S^2 = sum(r_i^2) / (n - p)
S2_y = np.sum(residuals**2) / (n - p)

# стандартная ошибка регрессии (СКО остатков)
S_y = np.sqrt(S2_y)

print(f"Стандартная ошибка регрессии: S_y = {S_y:.3f} кэВ")
print(f"Максимальное абсолютное отклонение: {np.max(np.abs(residuals)):.3f} кэВ")
print()

# 2. Проверка пригодности модели
print("2. Проверка пригодности модели")
print()

# коэффициент множественной корреляции R -- теснота связи между y и y_pred
R_multiple = np.corrcoef(energy, energy_pred)[0, 1]

# сумма квадратов остатков: SS_res = sum(r_i^2)
SS_res = np.sum(residuals**2)

# полная сумма квадратов: SS_tot = sum((y_i - y_mean)^2)
SS_tot = np.sum((energy - np.mean(energy)) ** 2)

# коэффициент детерминации: R^2 = 1 - SS_res / SS_tot
R2 = 1 - SS_res / SS_tot

# скорректированный R^2: штрафует за лишние параметры, формула R^2_adj = 1 - (1 - R^2) * (n-1)/(n-p)
R2_adj = 1 - (1 - R2) * (n - 1) / (n - p)

print(f"Коэффициент множественной корреляции: R = {R_multiple:.6f}")
print(f"Коэффициент детерминации: R^2 = {R2:.6f}")
print(f"Скорректированный R^2 = {R2_adj:.6f}")
print(f"Модель объясняет {R2 * 100:.2f}% вариации энергии")
print()

# 3. Анализ остатков
print("3. Анализ остатков")
print()

# среднее остатков: у хорошей модели должно быть близко к нулю
print(f"Среднее остатков: {np.mean(residuals):.6f} кэВ")
print()

print("Остатки по точкам:")
for i, (ch, res) in enumerate(zip(channel, residuals)):
    print(f"  точка {i + 1}: канал {ch}, остаток = {res:.4f} кэВ")
print()

# число положительных и отрицательных остатков -- признак симметрии распределения
n_plus = np.sum(residuals > 0)
n_minus = np.sum(residuals < 0)

print(f"Положительных остатков: {n_plus}, отрицательных: {n_minus}")
if abs(n_plus - n_minus) <= 2:
    print("Распределение знаков остатков примерно симметрично")
else:
    print("Наблюдается заметное неравенство знаков остатков")
print()

# 4. Вывод о пригодности модели
print("4. Вывод о пригодности модели")
print()

is_model_good = True
reasons = []

if R2 < 0.95:
    is_model_good = False
    reasons.append(f"R^2 = {R2:.4f} < 0.95 (модель объясняет менее 95% дисперсии)")

if np.abs(np.mean(residuals)) > S_y / 2:
    is_model_good = False
    reasons.append(f"среднее остатков ({np.mean(residuals):.4f}) не близко к нулю")

if np.max(np.abs(residuals)) > 3 * S_y:
    is_model_good = False
    reasons.append(
        f"есть выбросы: максимальное отклонение {np.max(np.abs(residuals)):.3f} > 3*S_y"
    )

if is_model_good:
    print("Модель пригодна для описания экспериментальных данных")
    print(f"R^2 = {R2:.6f} -- модель объясняет {R2 * 100:.2f}% дисперсии данных")
    print(f"R = {R_multiple:.6f} -- очень тесная связь между факторами и откликом")
    print(f"S_y = {S_y:.3f} кэВ -- стандартная ошибка регрессии мала")
    print("Остатки распределены симметрично относительно нуля")
    print(f"Максимальное отклонение: {np.max(np.abs(residuals)):.3f} кэВ")
else:
    print("Модель не пригодна для описания экспериментальных данных")
    for reason in reasons:
        print(f"- {reason}")
print()

# 5. Графики
# набор точек по оси x для отрисовки плавной кривой регрессии
k_plot = np.linspace(1890, 1980, 200)
E_plot = a * k_plot**2 + b * k_plot + c

fig1 = plt.figure(1, figsize=(10, 6))
plt.scatter(channel, energy, color="blue", s=80, label="экспериментальные данные")
plt.plot(k_plot, E_plot, "r-", lw=2, label="квадратичная регрессия")
plt.xlabel("номер канала")
plt.ylabel("энергия частицы, кэВ")
plt.title("Зависимость энергии частицы от номера канала")
plt.legend()
plt.grid(True, alpha=0.3)

fig2 = plt.figure(2, figsize=(10, 5))
plt.scatter(channel, residuals, color="red", s=60)
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("номер канала")
plt.ylabel("остатки, кэВ")
plt.title("График остатков квадратичной модели")
plt.grid(True, alpha=0.3)

plt.show()
