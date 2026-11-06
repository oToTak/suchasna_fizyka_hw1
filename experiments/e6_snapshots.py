"""E6. Знімки конфігурацій: нижче / біля / вище T_c та застряглий метастабільний стан (T = 2.0).

Застряглий стан шукаємо серед гарячих стартів; віддаємо перевагу справді смугастому (критерій stripe_stats).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # щоб працював запуск як скрипта

from ising import lattice, metropolis, plotting, presets
from ising.exact import onsager_abs_m
from ising.run import experiment_cli

NAME = "e6_snapshots"
EXP_ID = 6
T_STUCK = 2.0
STRIPE_MAJOR_MIN = 0.5   # смуги: профіль середніх по рядках/стовпцях сильно змінюється вздовж однієї осі,
STRIPE_RATIO_MAX = 0.25  # а вздовж іншої майже сталий (відношення розмахів менше за цей поріг)

PRESETS = {
    "quick": {"L": 40, "temps": (1.5, presets.T_C, 3.5), "n_therm": 500,
              "seed_limit": 60, "seed_limit_max": 200, "stuck_factor": 0.9},
    "full": {"L": 100, "temps": (1.5, presets.T_C, 3.5), "n_therm": 2000,
             "seed_limit": 60, "seed_limit_max": 200, "stuck_factor": 0.9},
}


def thermalize(L: int, T: float, n_sweeps: int, seed: int, cold: bool) -> np.ndarray:
    """Старт (холодний/гарячий) + n_sweeps проходів. Повертає кінцеву конфігурацію."""
    rng = np.random.default_rng(seed)
    metropolis.seed_numba(seed)
    spins = lattice.cold_lattice(L) if cold else lattice.hot_lattice(L, rng)
    metropolis.run_chain(spins, T, n_sweeps)
    return spins


def abs_m(spins: np.ndarray) -> float:
    return abs(float(metropolis.magnetization(spins))) / spins.size


def stripe_stats(spins: np.ndarray) -> tuple[float, float]:
    """(S_major, S_minor): стандартні відхилення середніх по стовпцях і по рядках, більше значення першим.

    Смуга (доменні стінки навколо ґратки): вздовж однієї осі середнє змінюється ±0.9, вздовж іншої майже стале.
    Круглий домен чи рівновага дають однакові S по обох осях.
    """
    f = spins.astype(np.float64)
    s_cols, s_rows = f.mean(axis=0).std(), f.mean(axis=1).std()
    return max(s_cols, s_rows), min(s_cols, s_rows)


def is_stripe(spins: np.ndarray) -> bool:
    major, minor = stripe_stats(spins)
    return major > STRIPE_MAJOR_MIN and minor / major < STRIPE_RATIO_MAX


def run(preset: str) -> Path:
    p = PRESETS[preset]
    L, n_therm = p["L"], p["n_therm"]
    seeds = presets.make_seeds(EXP_ID, p["seed_limit_max"] + 4)  # перші 4 — для рівноважних знімків

    configs, temps, mags, used_seeds, labels = [], [], [], [], []

    # 1-3: рівноважні знімки (холодний старт нижче T_c, гарячий інакше)
    for k, T in enumerate(p["temps"]):
        n = 2 * n_therm if abs(T - presets.T_C) < 0.05 else n_therm  # біля T_c довше
        s = thermalize(L, T, n, seeds[k], cold=T < presets.T_C)
        configs.append(s); temps.append(T); mags.append(abs_m(s)); used_seeds.append(seeds[k])
        labels.append("equilibrium")

    # 4: рівноважний знімок при T = 2.0 (холодний старт)
    s = thermalize(L, T_STUCK, n_therm, seeds[3], cold=True)
    eq_m = abs_m(s)
    configs.append(s); temps.append(T_STUCK); mags.append(eq_m); used_seeds.append(seeds[3])
    labels.append("equilibrium_T2")

    # 5: пошук застряглого стану (гарячий старт, перебір seed-ів): беремо перший смугастий;
    # якщо за seed_limit смугастого немає — перший застряглий (домен), інакше найменша |m|
    threshold = p["stuck_factor"] * float(onsager_abs_m(T_STUCK))
    candidates = seeds[4:]
    first_stuck = None   # (|m|, spins, seed) — перший застряглий не-смугастий
    best = None          # (|m|, spins, seed) — найменша |m|, якщо нічого не застрягло
    stripe = None
    n_tried = 0
    while n_tried < p["seed_limit"] and stripe is None:
        s = thermalize(L, T_STUCK, n_therm, candidates[n_tried], cold=False)
        m = abs_m(s)
        if best is None or m < best[0]:
            best = (m, s, candidates[n_tried])
        if m < threshold:
            if is_stripe(s):
                stripe = (m, s, candidates[n_tried])
            elif first_stuck is None:
                first_stuck = (m, s, candidates[n_tried])
        n_tried += 1
    chosen = stripe or first_stuck or best
    found = stripe is not None or first_stuck is not None
    kind = "stripe" if stripe else ("domain" if first_stuck else "none")
    major, minor = stripe_stats(chosen[1])
    configs.append(chosen[1]); temps.append(T_STUCK); mags.append(chosen[0]); used_seeds.append(chosen[2])
    labels.append("stuck")
    print(f"застряглий стан: тип={kind}, |m|={chosen[0]:.3f} (поріг {threshold:.3f}), S_major={major:.3f}, "
          f"S_minor={minor:.3f}, перебрано seed-ів: {n_tried}; рівновага |m|={eq_m:.3f}")

    out = presets.result_path(preset, NAME)
    np.savez(out, configs=np.array(configs, dtype=np.int8), T=np.array(temps), abs_m=np.array(mags),
             seeds=np.array(used_seeds), labels=np.array(labels), n_seeds_tried=n_tried,
             stuck_found=found, stuck_kind=kind, stuck_S=np.array([major, minor]),
             stuck_threshold=threshold, L=L, n_therm=n_therm,
             onsager_m_T2=float(onsager_abs_m(T_STUCK)))
    return out


def plot(preset: str) -> list[Path]:
    plotting.apply_style()
    d = np.load(presets.result_path(preset, NAME))
    L, n_therm, n_tried = int(d["L"]), int(d["n_therm"]), int(d["n_seeds_tried"])
    kind = str(d["stuck_kind"])
    stuck_title = {"stripe": "метастабільні смуги", "domain": "метастабільний домен"}.get(kind, "найменша $|m|$ (не застрягла)")
    titles = ["нижче $T_c$", "біля $T_c$", "вище $T_c$", "рівновага", stuck_title]
    order = [0, 1, 2, 3, 4]  # позиції в сітці 2x3: 0,1,2 / 4(застряглий),3(рівновага)
    cells = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0)]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    for ax in axes.ravel():
        ax.axis("off")
    for k, (r, c) in zip(order, cells):
        ax = axes[r, c]
        ax.imshow(d["configs"][k], cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{titles[k]}\n$T={d['T'][k]:.3f}$, $|m|={d['abs_m'][k]:.3f}$")
        ax.add_patch(plt.Rectangle((-0.5, -0.5), L, L, fill=False, edgecolor="gray", linewidth=1.0))
    S_major, S_minor = d["stuck_S"]
    note = (f"{stuck_title}\nпроти рівноваги\nпри $T=2.0$ (гарячий старт,\nперебрано seed-ів: {n_tried})\n"
            f"розмахи профілів $S$: {S_major:.2f} і {S_minor:.2f}")
    if kind == "domain":
        note += "\nсмугастого стану серед\nперебраних seed-ів не було"
    if not bool(d["stuck_found"]):
        note += "\nпоріг застрягання не досягнуто"
    axes[1, 2].text(0.5, 0.5, note, ha="center", va="center", transform=axes[1, 2].transAxes)
    fig.suptitle(f"Знімки конфігурацій, $L={L}$, термалізація {n_therm} проходів (біля $T_c$ — удвічі довше)")
    return plotting.save_figure(fig, preset, NAME)


if __name__ == "__main__":
    experiment_cli(run, plot)
