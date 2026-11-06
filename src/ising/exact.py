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


