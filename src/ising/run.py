"""Єдина точка входу: `python -m ising.run --preset quick|full [--only e1 e3] [--plot-only]`.

Також `experiment_cli(run, plot)` — спільний CLI для скриптів у experiments/.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from ising.presets import PRESET_NAMES, ROOT

EXPERIMENTS = {
    "e1": "experiments.e1_thermodynamics",
    "e2": "experiments.e2_thermalization",
    "e6": "experiments.e6_snapshots",
}


def _utf8_stdout() -> None:
    """Щоб українські повідомлення не ламались у консолі Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def experiment_cli(run, plot) -> None:
    """CLI для одного експерименту: --preset {quick,full} [--plot-only]."""
    _utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESET_NAMES, default="quick")
    parser.add_argument("--plot-only", action="store_true", help="лише перемалювати з наявного .npz")
    args = parser.parse_args()
    if not args.plot_only:
        print("дані:", run(args.preset))
    for path in plot(args.preset):
        print("рисунок:", path)


def main() -> None:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESET_NAMES, default="quick")
    parser.add_argument("--only", nargs="+", choices=sorted(EXPERIMENTS), help="лише ці експерименти")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))  # щоб працював `import experiments.*`
    for key in args.only or sorted(EXPERIMENTS):
        try:
            module = importlib.import_module(EXPERIMENTS[key])
        except ModuleNotFoundError as err:
            if err.name != EXPERIMENTS[key]:
                raise
            print(f"[{key}] пропущено: {EXPERIMENTS[key]} ще не написано")
            continue
        print(f"=== {key}: {EXPERIMENTS[key]} ({args.preset}) ===")
        if not args.plot_only:
            print("дані:", module.run(args.preset))
        for path in module.plot(args.preset):
            print("рисунок:", path)


if __name__ == "__main__":
    main()
