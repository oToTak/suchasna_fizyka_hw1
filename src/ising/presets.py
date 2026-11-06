"""Спільні константи, шляхи та seed-и. Параметри конкретних експериментів живуть у самих експериментах."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Критична температура Онзагера (J = k_B = 1)
T_C = 2.0 / math.log(1.0 + math.sqrt(2.0))

ROOT = Path(__file__).resolve().parents[2]
PRESET_NAMES = ("quick", "full")

BASE_SEED = 20261104

# "anneal" | "cold_below_hot_above" (докладніше в CLAUDE.md §3; підтверджується в E2)
EQUILIBRIUM_PROTOCOL = "anneal"


def results_dir(preset: str) -> Path:
    path = ROOT / "results" / preset
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(preset: str) -> Path:
    path = ROOT / "figures" / preset
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(preset: str, name: str) -> Path:
    """results/<preset>/<name>.npz, напр. result_path("quick", "e1_thermodynamics")."""
    return results_dir(preset) / f"{name}.npz"


def make_seeds(exp_id: int, n: int) -> list[int]:
    """n незалежних int-seed-ів для експерименту exp_id (0..7). Детерміновано від BASE_SEED."""
    children = np.random.SeedSequence(BASE_SEED, spawn_key=(exp_id,)).spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]
