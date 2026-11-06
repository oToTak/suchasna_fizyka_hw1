"""Єдиний стиль графіків і збереження PNG + PDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ising import presets

# Палітра Окабе-Іто (придатна для дальтоніків); жовтий замінено на темніший, щоб було видно на білому.
# Порядок за зростанням L: від світлих до темних, плюс різні маркери — читається і в чорно-білому друці.
L_COLORS: dict[int, str] = {
    8: "#E69F00",    # оранжевий
    16: "#56B4E9",   # блакитний
    20: "#009E73",   # зелений
    32: "#CC79A7",   # рожевий
    50: "#0072B2",   # синій
    64: "#D55E00",   # червоно-помаранчевий
    100: "#7F4A00",  # коричневий
    128: "#000000",  # чорний
}

# Маркер на розмір ґратки (ті самі L, що й у L_COLORS)
L_MARKERS: dict[int, str] = {
    8: "o", 16: "s", 20: "^", 32: "D", 50: "v", 64: "P", 100: "X", 128: "*",
}

# Запасні кольори для інших L
_FALLBACK_COLORS = ["#999999", "#44AA99", "#AA4499", "#117733"]

T_LABEL = r"$T$ ($J/k_B$)"

# Лінія теорії завжди виглядає однаково
theory_style = {"color": "black", "linestyle": "-", "linewidth": 1.2}


def apply_style() -> None:
    """Виставити rcParams: шрифти, сітка, розміри, dpi. Викликати на початку кожного plot()."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.figsize": (7.5, 4.8),
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.markersize": 4,
        "errorbar.capsize": 2,
        "axes.prop_cycle": plt.cycler(color=list(L_COLORS.values())),
    })


def color_for_L(L: int) -> str:
    """Стабільний колір для розміру L (для невідомих L — детермінований запасний колір)."""
    if L in L_COLORS:
        return L_COLORS[L]
    return _FALLBACK_COLORS[L % len(_FALLBACK_COLORS)]


def marker_for_L(L: int) -> str:
    """Маркер для розміру L (для невідомих L — «o»)."""
    return L_MARKERS.get(L, "o")


def add_tc_line(ax: plt.Axes) -> None:
    """Вертикальна пунктирна лінія T = T_c з підписом у легенді."""
    ax.axvline(presets.T_C, color="gray", linestyle="--", linewidth=1.0,
               label=rf"$T_c = {presets.T_C:.3f}$")


def save_figure(fig: plt.Figure, preset: str, name: str) -> list[Path]:
    """Зберегти fig у figures/<preset>/<name>.png (dpi 200) і .pdf. Повертає обидва шляхи."""
    out_dir = presets.figures_dir(preset)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths
