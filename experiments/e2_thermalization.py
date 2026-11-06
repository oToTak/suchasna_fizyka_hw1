"""E2. Траєкторії m(t), e(t) для гарячого та холодного старту при T = 1.5, 2.0, T_c, 3.0.

Гарячий (випадковий) старт при низьких T може застрягти в метастабільних смугастих доменах —
це фізика (довге життя доменних стін), а не помилка коду. Холодний старт (усі +1) такого не дає.
"""
from __future__ import annotations

import numpy as np

from ising import metropolis, plotting, presets
from ising.exact import onsager_abs_m, onsager_energy
from ising.lattice import cold_lattice, hot_lattice
from ising.run import experiment_cli

NAME = "e2_thermalization"
EXP_ID = 2

PRESETS = {
    "quick": {"L_traj": 32, "n_sweeps_traj": 600, "n_hot_traj": 3,
              "T_traj": [1.5, 2.0, presets.T_C, 3.0]},
    "full": {"L_traj": 100, "n_sweeps_traj": 3000, "n_hot_traj": 3,
             "T_traj": [1.5, 2.0, presets.T_C, 3.0]},
}


def start_chain(L: int, seed: int, cold: bool) -> np.ndarray:
    """Початкова ґратка + seed обох генераторів (NumPy для ґратки, Numba для ядра)."""
    rng = np.random.default_rng(seed)
    metropolis.seed_numba(seed)
    return cold_lattice(L) if cold else hot_lattice(L, rng)


def run(preset: str):
    p = PRESETS[preset]
    T_traj = np.array(p["T_traj"])
    n_hot = p["n_hot_traj"]
    L, n_sw = p["L_traj"], p["n_sweeps_traj"]
    seeds = presets.make_seeds(EXP_ID, len(T_traj) * (1 + n_hot))
    seeds_A = np.array(seeds).reshape(len(T_traj), 1 + n_hot)

    # [T, 0 = холодний | 1.. = гарячі, t]
    m_traj = np.empty((len(T_traj), 1 + n_hot, n_sw))
    e_traj = np.empty_like(m_traj)
    for a, T in enumerate(T_traj):
        for b in range(1 + n_hot):
            spins = start_chain(L, int(seeds_A[a, b]), cold=(b == 0))
            E, M = metropolis.run_chain(spins, float(T), n_sw)
            m_traj[a, b] = M / L**2
            e_traj[a, b] = E / L**2
        print(f"T = {T:.3f} готово", flush=True)

    path = presets.result_path(preset, NAME)
    np.savez(path, preset=preset, L_traj=L, n_sweeps_traj=n_sw, T_traj=T_traj,
             seeds_A=seeds_A, m_traj=m_traj, e_traj=e_traj)
    return path


def plot(preset: str) -> list:
    import matplotlib.pyplot as plt

    plotting.apply_style()
    d = np.load(presets.result_path(preset, NAME))
    T_traj, L = d["T_traj"], int(d["L_traj"])
    n_sw = int(d["n_sweeps_traj"])
    t = np.arange(1, n_sw + 1)
    n_hot = d["m_traj"].shape[1] - 1
    fig, axes = plt.subplots(len(T_traj), 2, figsize=(10, 2.6 * len(T_traj)), sharex=True)
    hot_colors = ["#D55E00", "#E69F00", "#CC79A7", "#009E73"]
    for a, T in enumerate(T_traj):
        for col, (series, theory, ylabel) in enumerate([
            (np.abs(d["m_traj"]), float(onsager_abs_m(T)), r"$|m|$"),
            (d["e_traj"], float(onsager_energy(T)), r"$e$"),
        ]):
            ax = axes[a, col]
            for b in range(n_hot):
                ax.plot(t, series[a, 1 + b], color=hot_colors[b % len(hot_colors)], lw=0.9, alpha=0.9,
                        label="гарячий старт" if b == 0 else None)
            ax.plot(t, series[a, 0], color="#0072B2", lw=1.0, label="холодний старт")
            ax.axhline(theory, **plotting.theory_style, label="Онзагер")
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            if col == 0:
                ax.set_title(rf"$T = {T:.3f}$", loc="left", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("проходи Метрополіса")
    fig.suptitle(f"E2: $m(t)$, $e(t)$; $L = {L}$, {n_sw} проходів, {n_hot} гарячих запусків і 1 холодний на панель",
                 fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)  # легенда поза даними
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return plotting.save_figure(fig, preset, "e2_trajectories")


if __name__ == "__main__":
    experiment_cli(run, plot)
