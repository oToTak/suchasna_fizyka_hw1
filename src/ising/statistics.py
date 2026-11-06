"""Статистика корельованих рядів: автокореляція, blocking, block-bootstrap."""
from __future__ import annotations

from typing import Callable

import numpy as np

MIN_BLOCKS = 16  # менше блоків — оцінка похибки надто шумна


def tau_int(x: np.ndarray, c: float = 6.0) -> tuple[float, int]:
    """Інтегральний час автокореляції з автоматичним вікном Сокала (вікно W — перше, де W >= c*tau(W)).
    Повертає (tau_int, W). Для некорельованого ряду tau_int ≈ 0.5.
    Сталий ряд (дисперсія 0) -> (0.5, 0)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    if n < 2 or not np.any(x):
        return 0.5, 0
    # автоковаріація через FFT з доповненням нулями до 2n (без циклічних ефектів)
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n]
    rho = acov / acov[0]
    # tau(W) = 1/2 + sum_{t=1}^W rho(t)
    tau_w = 0.5 + np.cumsum(rho[1:])
    windows = np.arange(1, n)
    ok = np.nonzero(windows >= c * tau_w)[0]
    # якщо вікно не знайдено — беремо найбільше доступне
    i = ok[0] if len(ok) else n - 2
    return float(tau_w[i]), int(windows[i])


def blocking_error(x: np.ndarray) -> float:
    """Похибка середнього корельованого ряду методом blocking (Флювбьєрг–Петерсен):
    послідовно усереднюємо пари сусідніх значень, беремо плато оцінки похибки.

    На рівні k: s_k = sqrt(var(блоки) / n_блоків), її власна невизначеність ≈ s_k / sqrt(2 (n_блоків - 1)).
    Критерій плато: перший рівень k, для якого наступний рівень уже не вищий за s_k на величину
    власної невизначеності s_{k+1}, тобто s_{k+1} <= s_k + s_{k+1}/sqrt(2 (n_{k+1} - 1)).
    Повертаємо s_k цього рівня. Рівні розглядаємо, поки лишається >= 16 блоків; якщо плато не знайдено —
    повертаємо оцінку на останньому такому рівні. Сталий ряд -> 0."""
    x = np.asarray(x, dtype=np.float64)
    estimates = []  # (s_k, s_k_error)
    while len(x) >= MIN_BLOCKS:
        nb = len(x)
        s = np.sqrt(np.var(x, ddof=1) / nb)
        estimates.append((s, s / np.sqrt(2 * (nb - 1))))
        x = x[: nb // 2 * 2].reshape(-1, 2).mean(axis=1)  # об'єднуємо пари
    if not estimates:
        raise ValueError(f"ряд закороткий для blocking (потрібно >= {MIN_BLOCKS} значень)")
    for k in range(len(estimates) - 1):
        s_k = estimates[k][0]
        s_next, err_next = estimates[k + 1]
        if s_next <= s_k + err_next:
            return float(s_k)
    return float(estimates[-1][0])


def block_bootstrap(
    series: tuple[np.ndarray, ...],
    func: Callable[..., float],
    block_len: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Block-bootstrap для довільної функції рядів (c_V, χ, U4 ...). Рядки series одного
    ресемплу беруться за ОДНИМИ й тими ж індексами блоків. func(*series_resampled) -> float.
    Повертає (func на оригіналі, std по бутстреп-вибірках).

    Блоки неперекриті, довжини block_len; хвіст, що не вміщає цілий блок, відкидається у ресемплах."""
    series = tuple(np.asarray(s) for s in series)
    n = len(series[0])
    if any(len(s) != n for s in series):
        raise ValueError("усі ряди мають бути однакової довжини")
    n_blocks = n // block_len
    if n_blocks < 1:
        raise ValueError("block_len більший за довжину ряду")
    # початки блоків для всіх ресемплів одразу: (n_boot, n_blocks)
    starts = rng.integers(0, n_blocks, size=(n_boot, n_blocks)) * block_len
    offsets = np.arange(block_len)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = (starts[b][:, None] + offsets).ravel()  # ті самі індекси для всіх рядів
        values[b] = func(*(s[idx] for s in series))
    return float(func(*series)), float(np.std(values, ddof=1))
