import numpy as np
import scipy.stats as stats

# значение мощности дозы эталонного источника (Зв/ч)
H_et = 5e-7

# измерения фона (Зв/ч)
background = [
    6.579e-8,
    6.579e-8,
    8.459e-8,
    1.222e-7,
    9.399e-8,
    3.76e-8,
    1.034e-7,
    1.128e-7,
    7.908e-8,
    9.246e-8,
]

# измерения источника + фон (Зв/ч)
source_with_bg = [
    4.7000e-7,
    5.0760e-7,
    5.2640e-7,
    6.0150e-7,
    4.2300e-7,
    5.0760e-7,
    4.6060e-7,
    5.2640e-7,
    5.8270e-7,
    5.0730e-7,
    5.2200e-7,
    5.3120e-7,
    5.1730e-7,
    5.3000e-7,
    4.9430e-7,
    4.8750e-7,
    5.1190e-7,
    5.0930e-7,
    5.1120e-7,
    5.0730e-7,
]

# абсолютная погрешность эталона (по формуле delta = 2.3 * H / 100)
delta_et = 2.3 * H_et / 100

bg = np.array(background)
src = np.array(source_with_bg)

print("Исходные данные")
print(f"значение эталона: H_et = {H_et:.2e} Зв/ч")
print(f"погрешность эталона: delta_et = {delta_et:.2e} Зв/ч")
print(f"количество измерений фона: n_fon = {len(background)}")
print(f"количество измерений источника+фон: n_ist = {len(source_with_bg)}")
print()

# 1. Обработка фоновых измерений
print("1. Обработка фоновых измерений")

bg_mean = np.mean(bg)  # среднее
bg_std = np.std(bg, ddof=1)  # СКО
bg_std_sr = bg_std / np.sqrt(len(bg))  # СКО среднего
bg_disp = bg_std**2  # дисперсия
bg_disp_sr = bg_std_sr**2  # дисперсия среднего

# квантиль для доверительной вероятности 0.95
t_quantile_bg = stats.t.ppf(0.975, df=len(bg) - 1)
bg_delta_random = t_quantile_bg * bg_std_sr  # случайная погрешность фона

# инструментальная погрешность не учитывается
bg_delta = bg_delta_random

print(f"среднее значение фона: bg_mean = {bg_mean:.2e} Зв/ч")
print(f"СКО фона: bg_std = {bg_std:.2e} Зв/ч")
print(f"дисперсия фона: bg_disp = {bg_disp:.2e} (Зв/ч)^2")
print(f"СКО среднего фона: bg_std_sr = {bg_std_sr:.2e} Зв/ч")
print(f"дисперсия среднего фона: bg_disp_sr = {bg_disp_sr:.2e} (Зв/ч)^2")
print(f"случайная погрешность фона (p=0.95): bg_delta = {bg_delta:.2e} Зв/ч")
print()

# 2. Обработка измерений источника с фоном
print("2. Обработка измерений источника с фоном")

src_mean = np.mean(src)  # среднее
src_std = np.std(src, ddof=1)  # СКО
src_std_sr = src_std / np.sqrt(len(src))  # СКО среднего
src_disp = src_std**2  # дисперсия
src_disp_sr = src_std_sr**2  # дисперсия среднего

t_quantile_src = stats.t.ppf(0.975, df=len(src) - 1)  # квантиль
src_delta_random = t_quantile_src * src_std_sr  # случайная погрешность

print(f"среднее значение (источник+фон): src_mean = {src_mean:.2e} Зв/ч")
print(f"СКО: src_std = {src_std:.2e} Зв/ч")
print(f"дисперсия: src_disp = {src_disp:.2e} (Зв/ч)^2")
print(f"СКО среднего: src_std_sr = {src_std_sr:.2e} Зв/ч")
print(f"дисперсия среднего: src_disp_sr = {src_disp_sr:.2e} (Зв/ч)^2")
print(f"случайная погрешность (p=0.95): src_delta = {src_delta_random:.2e} Зв/ч")
print()

# 3. Вычисление чистого сигнала источника
print("3. Вычисление чистого сигнала источника")

# чистый сигнал источника (среднее значение)
H_ist = src_mean - bg_mean

# дисперсия чистого сигнала (сумма дисперсий средних)
D_ist = bg_disp_sr + src_disp_sr

# стандартная ошибка чистого сигнала (СКО среднего)
S_ist = np.sqrt(D_ist)

# погрешность чистого сигнала (метод переноса погрешностей)
delta_H_ist = np.sqrt(src_delta_random**2 + bg_delta**2)

print(f"чистый сигнал источника: H_ist = {H_ist:.2e} Зв/ч")
print(f"дисперсия чистого сигнала: D_ist = {D_ist:.2e} (Зв/ч)^2")
print(f"стандартная ошибка чистого сигнала: S_ist = {S_ist:.2e} Зв/ч")
print(f"погрешность чистого сигнала: delta_H_ist = {delta_H_ist:.2e} Зв/ч")
print()

# 4. Оценка систематической погрешности прибора
print("4. Оценка систематической погрешности прибора")

# систематическая погрешность (разность между измеренным и эталонным значением)
delta_sist = abs(H_ist - H_et)

# параметры для расчета delta_izm
n = 20
n1 = 10  # Стьюдент
n2 = 10  # Пирсон

# квантили
t_student = stats.t.ppf(
    0.975, df=n1 - 1
)  # квантиль Стьюдента с df=9 (фиксированное значение)
chi2_value = stats.chi2.ppf(
    0.975, df=n2 - 1
)  # квантиль хи-квадрат с df=19 (фиксированное значение)

# погрешность измерения с учетом максимизации
# используется фиксированное значение n_fixed, не зависящее от расчетов выше
delta_izm = S_ist * np.sqrt(n - 1) * t_student / np.sqrt(chi2_value)

# полная погрешность систематической составляющей (арифметическая сумма)
delta_sist_full = delta_sist + delta_et + delta_izm

print(f"эталонное значение: H_et = {H_et:.2e} Зв/ч")
print(f"измеренное значение чистого сигнала: H_ist = {H_ist:.2e} Зв/ч")
print(f"систематическая погрешность (по модулю): |delta_сист| = {delta_sist:.2e} Зв/ч")
print(f"погрешность измерения (с максимизацией): delta_izm = {delta_izm:.2e} Зв/ч")
print(f"полная погрешность: Delta_сист = {delta_sist_full:.2e} Зв/ч")
print()

# 5. Итоговый вывод
print("5. Итоговый вывод")

print(
    f"Результат измерения чистого сигнала: H_ist = ({H_ist:.2e} +- {delta_H_ist:.2e}) Зв/ч, p = 0.95"
)
print(f"Эталонное значение: H_et = ({H_et:.2e} +- {delta_et:.2e}) Зв/ч, p = 0.95")
print(f"Систематическая погрешность: |Delta_сист| = {delta_sist:.2e} Зв/ч")
print(
    f"Полная погрешность систематической составляющей: Delta_сист,полн = {delta_sist_full:.2e} Зв/ч"
)
