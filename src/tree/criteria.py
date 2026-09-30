"""Impurity criteria and information gain, from scratch.

Gini and entropy for classification, MSE for regression — each a short function
measuring how mixed a group of labels is. The information gain of a split
(parent impurity minus weighted child impurity), computed with *entropy*, is
exactly the mutual information between the split and the target; that identity
is asserted in the tests and is the concrete link to information theory.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _class_probabilities(y: NDArray) -> Array:
    """Empirical class probabilities of a label array."""
    _, counts = np.unique(y, return_counts=True)
    return counts.astype(np.float64) / len(y)


def gini(y: NDArray) -> float:
    """Gini impurity ``1 - sum(p_k^2)``. Zero when pure, 0.5 for a 50/50 binary split."""
    if len(y) == 0:
        return 0.0
    p = _class_probabilities(y)
    return float(1.0 - np.sum(p * p))


def entropy(y: NDArray) -> float:
    """Shannon entropy in **bits** (``log2``). Zero when pure, 1.0 for a 50/50 binary split."""
    if len(y) == 0:
        return 0.0
    p = _class_probabilities(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mse(y: NDArray) -> float:
    """Mean squared error about the mean — the regression impurity (a variance)."""
    if len(y) == 0:
        return 0.0
    yf = np.asarray(y, dtype=np.float64)
    return float(np.mean((yf - yf.mean()) ** 2))


_CRITERIA = {"gini": gini, "entropy": entropy, "mse": mse}


def impurity(y: NDArray, criterion: str = "gini") -> float:
    """Dispatch to the named impurity function."""
    return _CRITERIA[criterion](y)


def information_gain(y: NDArray, left_mask: NDArray, criterion: str = "gini") -> float:
    """Impurity of the parent minus the sample-weighted impurity of the two children."""
    n = len(y)
    if n == 0:
        return 0.0
    left, right = y[left_mask], y[~left_mask]
    n_l, n_r = len(left), len(right)
    if n_l == 0 or n_r == 0:
        return 0.0
    weighted = (n_l / n) * impurity(left, criterion) + (n_r / n) * impurity(right, criterion)
    return float(impurity(y, criterion) - weighted)


def mutual_information(split: NDArray, y: NDArray) -> float:
    """Mutual information ``I(split; y)`` in bits, from the joint distribution.

    For a boolean ``split`` this equals ``information_gain(y, split, "entropy")`` —
    "splits to maximize information gain" and "splits to reduce uncertainty about
    the label" are the same statement.
    """
    n = len(y)
    if n == 0:
        return 0.0
    mi = 0.0
    for s in np.unique(split):
        p_s = float(np.mean(split == s))
        for c in np.unique(y):
            p_c = float(np.mean(y == c))
            p_joint = float(np.mean((split == s) & (y == c)))
            if p_joint > 0.0:
                mi += p_joint * np.log2(p_joint / (p_s * p_c))
    return float(mi)
