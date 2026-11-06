# 2D модель Ізінга, алгоритм Метрополіса (ДЗ1)

Гамільтоніан `H = -J Σ_<i,j> s_i s_j`, `s_i = ±1`, ґратка `L×L`, періодичні граничні умови, `J = k_B = 1`.
Ядро Метрополіса — Python + Numba (`src/ising/metropolis.py`).

| Рисунок | Скрипт | Що показує |
|---|---|---|
| `figures/full/e1_thermodynamics.png` | `experiments/e1_thermodynamics.py` | `e(T)`, `⟨|m|⟩(T)`, `c_V(T)` для `L = 20, 50, 100` + Онзагер |
| `figures/full/e2_trajectories.png` | `experiments/e2_thermalization.py` | `m(t)`, `e(t)` для гарячого й холодного старту |
| `figures/full/e6_snapshots.png` | `experiments/e6_snapshots.py` | знімки конфігурацій, у т.ч. застряглий смугастий стан |

## Запуск

```bash
pip install -e .
python -m ising.run --preset quick               # хвилини
python -m ising.run --preset full                # повний розрахунок (довго)
python -m ising.run --preset full --plot-only    # лише перемалювати з results/full/*.npz
```

Один експеримент: `python -m experiments.e2_thermalization --preset quick`.
Дані (`results/<preset>/*.npz`) окремо від графіків (`figures/<preset>/*.png|pdf`).

## Обраховані графіки

- [](figures/full/e1_thermodynamics.png)
- [](figures/full/e2_trajectories.png)
- [](figures/full/e6_snapshots.png)
