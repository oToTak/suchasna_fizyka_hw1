"""Початкові конфігурації ґратки. Спіни зберігаються як int8 (+1 / -1), форма (L, L)."""
from __future__ import annotations

import numpy as np


def hot_lattice(L: int, rng: np.random.Generator) -> np.ndarray:
    """Випадкова ґратка: кожен спін ±1 з імовірністю 1/2. dtype int8."""
    return (2 * rng.integers(0, 2, size=(L, L)) - 1).astype(np.int8)


def cold_lattice(L: int, sign: int = 1) -> np.ndarray:
    """Впорядкована ґратка: усі спіни = sign (±1). dtype int8."""
    if sign not in (-1, 1):
        raise ValueError("sign має бути +1 або -1")
    return np.full((L, L), sign, dtype=np.int8)
