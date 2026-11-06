"""Спостережувані з часових рядів E(t), M(t) (int64, значення після кожного проходу).

Усі функції приймають уже відкинуту термалізацію (рядки лише для вимірювань), L та T.
N = L*L. Означення (CLAUDE.md §1, E4, E5):
  e = ⟨E⟩/N;  ⟨|m|⟩;  c_V = (⟨E²⟩ - ⟨E⟩²)/(N T²);
  де m = M/N.
"""
from __future__ import annotations

import numpy as np


def energy_per_site(E: np.ndarray, L: int) -> float:
    # float64 до усереднення, щоб не було переповнення
    return float(np.mean(np.asarray(E, dtype=np.float64)) / (L * L))


def abs_magnetization(M: np.ndarray, L: int) -> float:
    m = np.asarray(M, dtype=np.float64) / (L * L)
    return float(np.mean(np.abs(m)))


def specific_heat(E: np.ndarray, L: int, T: float) -> float:
    # c_V = (<E^2> - <E>^2) / (N T^2); np.var рахує <E^2> - <E>^2 стабільно
    E = np.asarray(E, dtype=np.float64)
    return float(np.var(E) / (L * L * T * T))


def all_observables(E: np.ndarray, M: np.ndarray, L: int, T: float) -> dict[str, float]:
    """Ключі: "e", "abs_m", "cv"."""
    return {"e": energy_per_site(E, L), "abs_m": abs_magnetization(M, L), "cv": specific_heat(E, L, T)}
