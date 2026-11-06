"""Швидке ядро Метрополіса (Numba). Той самий алгоритм, що у слайдах, але скомпільований.

Контракт (див. CLAUDE.md §1, §3):
  * E = -Σ s_ij (s_{i+1,j} + s_{i,j+1}) з періодичними граничними умовами (J = 1);
  * один sweep = N = L*L спроб перевертання випадкового вузла;
  * exp(-4/T) та exp(-8/T) рахуються раз на температуру (boltzmann_table), не в циклі;
  * усі функції окрім boltzmann_table мають бути @njit(cache=True).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def seed_numba(seed: int) -> None:
    """Засіває генератор Numba. Має викликати np.random.seed(seed) ВСЕРЕДИНІ @njit-функції."""
    np.random.seed(seed)


@njit(cache=True)
def energy(spins: np.ndarray) -> int:
    """Повна енергія E (int), кожну пару сусідів порахувано один раз."""
    L = spins.shape[0]
    E = 0
    for i in range(L):
        for j in range(L):
            # лише сусіди «вниз» і «вправо» (з pbc), щоб кожна пара була один раз
            E -= spins[i, j] * (spins[(i + 1) % L, j] + spins[i, (j + 1) % L])
    return E


@njit(cache=True)
def magnetization(spins: np.ndarray) -> int:
    """Сума спінів M = Σ s_i (int; m = M / N)."""
    L = spins.shape[0]
    M = 0
    for i in range(L):
        for j in range(L):
            M += spins[i, j]
    return M


@njit(cache=True)
def delta_energy(spins: np.ndarray, i: int, j: int) -> int:
    """ΔE = 2 s_ij (сума 4 сусідів) для перевертання вузла (i, j), pbc."""
    L = spins.shape[0]
    neighbours = (
        spins[(i + 1) % L, j]
        + spins[(i - 1) % L, j]
        + spins[i, (j + 1) % L]
        + spins[i, (j - 1) % L]
    )
    return 2 * spins[i, j] * neighbours


@njit(cache=True)
def _sweep_with_changes(spins: np.ndarray, table: np.ndarray) -> tuple[int, int]:
    """Один прохід; повертає сумарні зміни (ΔE_total, ΔM_total) для інкрементального обліку."""
    L = spins.shape[0]
    dE_total = 0
    dM_total = 0
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = delta_energy(spins, i, j)
        if dE <= 0:
            accept = True
        elif dE == 4:
            accept = np.random.random() < table[0]
        else:  # dE == 8
            accept = np.random.random() < table[1]
        if accept:
            dM_total -= 2 * spins[i, j]
            spins[i, j] = -spins[i, j]
            dE_total += dE
    return dE_total, dM_total


@njit(cache=True)
def run_chain(spins: np.ndarray, T: float, n_sweeps: int) -> tuple[np.ndarray, np.ndarray]:
    """n_sweeps проходів на місці. Повертає (E, M): два int64-масиви довжини n_sweeps,
    значення ПІСЛЯ кожного проходу. Термалізацію користувач відкидає зрізом."""
    table = np.array([np.exp(-4.0 / T), np.exp(-8.0 / T)])  # раз на температуру
    E = energy(spins)  # початкові E, M — один раз, далі інкрементально
    M = magnetization(spins)
    E_series = np.empty(n_sweeps, dtype=np.int64)
    M_series = np.empty(n_sweeps, dtype=np.int64)
    for t in range(n_sweeps):
        dE, dM = _sweep_with_changes(spins, table)
        E += dE
        M += dM
        E_series[t] = E
        M_series[t] = M
    return E_series, M_series
