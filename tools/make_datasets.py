"""Generate the two demo datasets under data/.

Both files are plain numeric CSVs with columns [x1, x2, target], target in
{-1, +1}. The seed is fixed, so re-running this reproduces the files exactly.

    python tools/make_datasets.py

Requires numpy and scipy (scipy only for the separability check).
"""

import pathlib

import numpy as np
from scipy.optimize import linprog

SEED = 20250127
OUT = pathlib.Path(__file__).resolve().parent.parent / "data"


def make(rng, n_per_class, centre_a, centre_b, spread, flipped=0):
    """Two Gaussian clusters, optionally with a few labels flipped."""
    a = rng.normal(centre_a, spread, size=(n_per_class, 2))
    b = rng.normal(centre_b, spread, size=(n_per_class, 2))
    points = np.vstack([a, b])
    targets = np.hstack([-np.ones(n_per_class), np.ones(n_per_class)])
    if flipped:
        targets[rng.choice(len(targets), flipped, replace=False)] *= -1
    order = rng.permutation(len(targets))
    return np.column_stack([points[order], targets[order]])


def is_separable(dataset):
    """True iff some (w, b) satisfies target_i * (w . x_i + b) >= 1 for all i."""
    points, targets = dataset[:, :2], dataset[:, 2]
    a_ub = -(targets[:, None] * np.column_stack([points, np.ones(len(points))]))
    result = linprog(
        c=np.zeros(3),
        A_ub=a_ub,
        b_ub=-np.ones(len(points)),
        bounds=[(None, None)] * 3,
        method="highs",
    )
    return result.status == 0


def main():
    rng = np.random.default_rng(SEED)
    sets = {
        "separable2D.csv": make(rng, 30, (2, 2), (6, 6), 0.7),
        "nonseparable2D.csv": make(rng, 40, (3.0, 3.0), (5.0, 4.6), 1.25, flipped=4),
    }
    OUT.mkdir(exist_ok=True)
    for name, dataset in sets.items():
        np.savetxt(OUT / name, dataset, delimiter=",", fmt="%.4f")
        print(f"{name}: N={len(dataset)}, linearly separable: {is_separable(dataset)}")


if __name__ == "__main__":
    main()
