"""E1. Термодинаміка: e(T), <|m|>(T), c_V(T) для кількох L з теорією Онзагера (слайд лекції 2)."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from ising import exact, metropolis, plotting, presets
from ising.lattice import cold_lattice, hot_lattice
from ising.observables import all_observables, specific_heat
from ising.run import experiment_cli
from ising.statistics import block_bootstrap, blocking_error, tau_int

NAME = "e1_thermodynamics"
N_BOOT = 100

PRESETS = {
    "quick": {"L": (16, 32), "T": np.linspace(1.5, 3.5, 41), "n_therm": 500, "n_meas": 2000},
    "full": {"L": (20, 50, 100), "T": np.arange(1.5, 3.5 + 1e-9, 0.02), "n_therm": 2000, "n_meas": 5000},
}


def measure_point(spins, L, T, seed, n_therm, n_meas):
    """Одна точка (L, T): ланцюг на місці, відкидаємо термалізацію, рахуємо величини та похибки."""
    metropolis.seed_numba(seed)
    E, M = metropolis.run_chain(spins, T, n_therm + n_meas)
    E, M = E[n_therm:], M[n_therm:]
    obs = all_observables(E, M, L, T)
    m = np.abs(M) / (L * L)
    tau_e, _ = tau_int(E)
    block_len = max(1, math.ceil(2 * tau_e))
    rng = np.random.default_rng(seed)
    _, cv_err = block_bootstrap((E,), lambda x: specific_heat(x, L, T), block_len, N_BOOT, rng)
    return {
        "e": obs["e"], "e_err": blocking_error(E / (L * L)),
        "abs_m": obs["abs_m"], "abs_m_err": blocking_error(m),
        "cv": obs["cv"], "cv_err": cv_err,
        "tau_e": tau_e,
    }


def scan(L, temps, seeds, p, protocol):
    """Скан по сітці T для одного L (від високих T до низьких). Повертає словник масивів довжини len(temps)."""
    keys = ["e", "e_err", "abs_m", "abs_m_err", "cv", "cv_err", "tau_e"]
    out = {k: np.zeros(len(temps)) for k in keys}
    order = np.argsort(temps)[::-1]  # від найвищої T до найнижчої
    spins = None
    for rank, k in enumerate(order):
        T = float(temps[k])
        rng = np.random.default_rng(seeds[k])
        if (protocol == "anneal" and rank == 0):
            spins = hot_lattice(L, rng)
        elif protocol == "cold_below_hot_above":
            spins = cold_lattice(L) if T < presets.T_C else hot_lattice(L, rng)
        # anneal: spins лишається з попередньої температури
        res = measure_point(spins, L, T, seeds[k], p["n_therm"], p["n_meas"])
        for key in keys:
            out[key][k] = res[key]
    return out


def run(preset: str):
    p = PRESETS[preset]
    protocol = presets.EQUILIBRIUM_PROTOCOL
    Ls, temps = p["L"], np.asarray(p["T"], dtype=float)
    nL, nT = len(Ls), len(temps)
    seeds = np.array(presets.make_seeds(1, nL * nT)).reshape(-1)
    main_seeds = seeds.reshape(nL, nT)

    data = {k: np.zeros((nL, nT)) for k in
            ["e", "e_err", "abs_m", "abs_m_err", "cv", "cv_err", "tau_e"]}
    for a, L in enumerate(Ls):
        print(f"L = {L} ({protocol})", flush=True)
        res = scan(L, temps, main_seeds[a], p, protocol)
        for key in data:
            data[key][a] = res[key]

    path = presets.result_path(preset, NAME)
    np.savez(
        path, L=np.array(Ls), T=temps, seeds=main_seeds,
        n_therm=p["n_therm"], n_meas=p["n_meas"], protocol=protocol, n_boot=N_BOOT,
        **data,
    )
    return path


def plot(preset: str):
    plotting.apply_style()
    d = np.load(presets.result_path(preset, NAME))
    Ls, T = d["L"], d["T"]
    info = f"$n_{{therm}}$={int(d['n_therm'])}, $n_{{meas}}$={int(d['n_meas'])}, протокол: {d['protocol']}"
    Tt = np.linspace(T.min(), T.max(), 800)
    Tt = Tt[np.abs(Tt - presets.T_C) > 0.004]  # c_V Онзагера розбігається в T_c

    # --- рисунок 1: три панелі
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("e", "e_err", "$e$ ($J$ на вузол)", exact.onsager_energy),
              ("abs_m", "abs_m_err", r"$\langle |m| \rangle$", exact.onsager_abs_m),
              ("cv", "cv_err", r"$c_V$ ($k_B$ на вузол)", exact.onsager_cv)]
    for ax, (key, err, label, theory) in zip(axes, panels):
        ax.plot(Tt, theory(Tt), label=r"Онзагер ($L\to\infty$)", **plotting.theory_style)
        for a, L in enumerate(Ls):
            L = int(L)
            ax.errorbar(T, d[key][a], yerr=d[err][a], fmt=plotting.marker_for_L(L), ls="none",
                        color=plotting.color_for_L(L), label=f"$L={L}$")
        plotting.add_tc_line(ax)
        ax.set_xlabel(plotting.T_LABEL)
        ax.set_ylabel(label)
    axes[0].legend()
    fig.suptitle(f"E1: термодинаміка 2D моделі Ізінга. {info}")
    paths = plotting.save_figure(fig, preset, NAME)

    return paths


if __name__ == "__main__":
    experiment_cli(run, plot)
