"""Точні результати: Онзагер (нескінченна ґратка) і повний перебір 2^N станів (мала ґратка). J = k_B = 1."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ellipk

from .presets import T_C


def onsager_abs_m(T):
    """⟨|m|⟩ = (1 - sinh^{-4}(2/T))^{1/8} для T < T_c, 0 для T >= T_c. Приймає скаляр або масив."""
    T = np.asarray(T, dtype=float)
    below = T < T_C
    T_safe = np.where(below, T, 1.0)  # щоб не рахувати поза областю
    value = (1.0 - np.sinh(2.0 / T_safe) ** -4) ** 0.125
    result = np.where(below, value, 0.0)
    return float(result) if result.ndim == 0 else result


def onsager_energy(T):
    """e(T) = -coth(2β)[1 + (2/π)(2 tanh²(2β) - 1) K(κ)], κ = 2 sinh(2β)/cosh²(2β).
    scipy.special.ellipk приймає κ², а не κ! Приймає скаляр або масив."""
    T = np.asarray(T, dtype=float)
    x = 2.0 / T  # x = 2β
    kappa = 2.0 * np.sinh(x) / np.cosh(x) ** 2
    m = kappa**2  # параметр для ellipk
    at_tc = m >= 1.0 - 1e-14  # K розбігається, а множник -> 0: беремо границю
    m_safe = np.where(at_tc, 0.0, m)
    bracket = 1.0 + (2.0 / math.pi) * (2.0 * np.tanh(x) ** 2 - 1.0) * ellipk(m_safe)
    value = -bracket / np.tanh(x)
    result = np.where(at_tc, -math.sqrt(2.0), value)
    return float(result) if result.ndim == 0 else result


def onsager_cv(T):
    """c_V = de/dT (чисельна похідна від onsager_energy; обережно біля T_c, де є логарифмічна розбіжність)."""
    T = np.asarray(T, dtype=float)
    h = 1e-4 * T  # крок масштабується з T
    result = (onsager_energy(T + h) - onsager_energy(T - h)) / (2.0 * h)
    return float(result) if result.ndim == 0 else result


def enumerate_states(L: int, T_values) -> dict[str, np.ndarray]:
    """Повний перебір усіх 2^(L*L) конфігурацій (L <= 4) з pbc.
    Повертає масиви довжини len(T_values): ключі "Z", "e" (=⟨E⟩/N), "abs_m", "m2", "cv" (на вузол)."""
    N = L * L
    T_values = np.atleast_1d(np.asarray(T_values, dtype=float))

    # матриця станів 2^N x N зі спінами ±1 (біти числа стану)
    index = np.arange(2**N, dtype=np.int64)
    bits = (index[:, None] >> np.arange(N)) & 1
    spins = (2 * bits - 1).astype(np.int64)

    # сусіди праворуч і вниз з pbc: E = -sum s_ij (s_{i+1,j} + s_{i,j+1})
    site = np.arange(N).reshape(L, L)
    down = np.roll(site, -1, axis=0).ravel()
    right = np.roll(site, -1, axis=1).ravel()
    energy = -(spins * (spins[:, down] + spins[:, right])).sum(axis=1)
    mag = spins.sum(axis=1) / N

    Z = np.empty(len(T_values))
    e = np.empty(len(T_values))
    abs_m = np.empty(len(T_values))
    m2 = np.empty(len(T_values))
    cv = np.empty(len(T_values))
    e_min = energy.min()
    for k, T in enumerate(T_values):
        w = np.exp(-(energy - e_min) / T)  # віднімаємо E_min для стійкості
        norm = w.sum()
        mean_E = (w * energy).sum() / norm
        mean_E2 = (w * energy.astype(float) ** 2).sum() / norm
        Z[k] = norm * math.exp(-e_min / T) if -e_min / T < 700 else np.inf
        e[k] = mean_E / N
        abs_m[k] = (w * np.abs(mag)).sum() / norm
        m2[k] = (w * mag**2).sum() / norm
        cv[k] = (mean_E2 - mean_E**2) / (N * T**2)
    return {"Z": Z, "e": e, "abs_m": abs_m, "m2": m2, "cv": cv}
