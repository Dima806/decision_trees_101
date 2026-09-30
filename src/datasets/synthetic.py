"""Synthetic 2-D datasets, generated on the fly so the structure is known.

Every problem here has a boundary we can draw and a claim we can measure: the
staircase on moons/spirals, the greedy failure on XOR, the win on the
checkerboard, the memorization on noisy blobs, and the piecewise-constant fit on
the regression curve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs, make_moons

Dataset = tuple[NDArray[np.float64], NDArray[np.int_]]


def two_blobs(n_samples: int = 200, seed: int = 42) -> Dataset:
    """Two well-separated Gaussian blobs — the one-split starter problem."""
    X, y = make_blobs(
        n_samples=n_samples, centers=2, n_features=2, cluster_std=1.2, random_state=seed
    )
    return X.astype(np.float64), y.astype(np.int_)


def moons(n_samples: int = 400, noise: float = 0.2, seed: int = 42) -> Dataset:
    """Two interleaving half-moons — a curved boundary the staircase approximates."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X.astype(np.float64), y.astype(np.int_)


def spirals(n_samples: int = 400, noise: float = 0.5, seed: int = 42) -> Dataset:
    """Two intertwined spirals — a hard curved boundary."""
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    theta = np.sqrt(rng.uniform(0.0, 1.0, n)) * 3.0 * np.pi
    arm = np.c_[theta * np.cos(theta), theta * np.sin(theta)]
    x0 = arm + rng.normal(0.0, noise, (n, 2))
    x1 = -arm + rng.normal(0.0, noise, (n, 2))
    X = np.vstack([x0, x1])
    X = X / np.max(np.abs(X))
    y = np.r_[np.zeros(n), np.ones(n)].astype(np.int_)
    return X.astype(np.float64), y


def checkerboard(n_samples: int = 400, grid: int = 2, seed: int = 42) -> Dataset:
    """A checkerboard — axis-aligned greedy splits are exactly right here."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n_samples, 2))
    y = (np.floor(X[:, 0] * grid) + np.floor(X[:, 1] * grid)).astype(np.int_) % 2
    return X.astype(np.float64), y


def xor(n_samples: int = 400, noise: float = 0.1, seed: int = 42) -> Dataset:
    """The XOR showpiece: label is the interaction of two features, neither informative alone."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n_samples, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int_)
    X = X + rng.normal(0.0, noise, X.shape)
    return X.astype(np.float64), y


def noisy_blobs(n_samples: int = 400, noise: float = 0.25, seed: int = 42) -> Dataset:
    """Two blobs with a fraction of labels flipped — the overfitting showpiece."""
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=2.0, random_state=seed)
    rng = np.random.default_rng(seed)
    flip = rng.uniform(0.0, 1.0, n_samples) < noise
    y = y.copy()
    y[flip] = 1 - y[flip]
    return X.astype(np.float64), y.astype(np.int_)


def regression_curve(
    n_samples: int = 200, noise: float = 0.15, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A nonlinear 1-D curve the regression tree fits in piecewise-constant steps."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, n_samples))
    y = np.sin(x) + 0.3 * x + rng.normal(0.0, noise, n_samples)
    return x.reshape(-1, 1).astype(np.float64), y.astype(np.float64)
