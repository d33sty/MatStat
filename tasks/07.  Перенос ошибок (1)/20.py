"""
Задача 24.5 — Перенос ошибок: плотность потока нейтронов от источника 252Cf
"""

import numpy as np

# ── Исходные данные ───────────────────────────────────────────────────────────
S0 = 7.63e9  # скорость эмиссии нейтронов, с⁻¹
sigma_S = 0.015 * S0  # абсолютная погрешность S (1,5%)

r = 50.0  # расстояние, см
sigma_r = 0.3  # погрешность расстояния, см

T_half = 2.645  # период полураспада, лет
sigma_T = 0.212  # погрешность T₁/₂, лет

t_yr = 30 / 365.25  # время наблюдения, лет


# ── Вспомогательная функция округления ───────────────────────────────────────
def round_result(value, uncertainty):
    """Округляет до 2 значащих цифр погрешности, значение — до того же разряда."""
    mag = int(np.floor(np.log10(abs(uncertainty))))
    dec = -(mag - 1)
    return round(value, dec), round(uncertainty, dec)


# ── Часть а) ──────────────────────────────────────────────────────────────────
Phi_a = S0 / (4 * np.pi * r**2)

var_S = (sigma_S / S0) ** 2
var_r = (2 * sigma_r / r) ** 2
total = var_S + var_r

sigma_Phi_a = Phi_a * np.sqrt(total)

print("ЧАСТЬ А)")
print(f"  Φ             = {Phi_a:.4e} см⁻²с⁻¹")
print(f"  σ_Φ           = {sigma_Phi_a:.3e} см⁻²с⁻¹  ({np.sqrt(total)*100:.3f}%)")
print(f"  Вклад от S    = {var_S/total*100:.1f}% дисперсии")
print(f"  Вклад от r    = {var_r/total*100:.1f}% дисперсии")
v, u = round_result(Phi_a, sigma_Phi_a)
print(f"\n  ► Φ = ({v:.2e} ± {u:.1e}) см⁻²с⁻¹")

# ── Часть б) ──────────────────────────────────────────────────────────────────
lam = np.log(2) / T_half
decay = np.exp(-lam * t_yr)
Phi_b = Phi_a * decay

var_T = (np.log(2) * t_yr / T_half**2 * sigma_T) ** 2
total_b = var_S + var_r + var_T

sigma_Phi_b = Phi_b * np.sqrt(total_b)

print("\nЧАСТЬ Б)")
print(f"  λ             = {lam:.5f} лет⁻¹")
print(f"  exp(−λt)      = {decay:.6f}  (снижение {(1-decay)*100:.3f}%)")
print(f"  Φ(t)          = {Phi_b:.4e} см⁻²с⁻¹")
print(f"  σ_Φ           = {sigma_Phi_b:.3e} см⁻²с⁻¹  ({np.sqrt(total_b)*100:.3f}%)")
print(f"  Вклад от S    = {var_S/total_b*100:.1f}% дисперсии")
print(f"  Вклад от r    = {var_r/total_b*100:.1f}% дисперсии")
print(f"  Вклад от T₁/₂ = {var_T/total_b*100:.2f}% дисперсии")
v, u = round_result(Phi_b, sigma_Phi_b)
print(f"\n  ► Φ(t) = ({v:.2e} ± {u:.1e}) см⁻²с⁻¹")
