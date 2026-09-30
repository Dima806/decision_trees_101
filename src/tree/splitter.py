"""The greedy heart: find the single best (feature, threshold) split.

For every feature and every candidate threshold, score the split by its
information gain and return the best one. Candidate thresholds are the midpoints
between adjacent sorted feature values, and impurity is accumulated with a single
sorted scan (cumulative class counts, or cumulative sums for regression) so the
search is instant on the small 2-D datasets used throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Split:
    """A chosen split. ``left_mask`` selects rows with ``X[:, feature] <= threshold``."""

    feature: int
    threshold: float
    gain: float
    left_mask: NDArray[np.bool_]
    right_mask: NDArray[np.bool_]


def _impurity_from_counts(counts: NDArray, n: NDArray, criterion: str) -> NDArray:
    """Vectorized impurity for many candidate children given their class counts.

    ``counts`` has shape ``(m, K)`` and ``n`` shape ``(m,)``.
    """
    p = counts / n[:, None]
    if criterion == "gini":
        return 1.0 - np.sum(p * p, axis=1)
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    return -np.sum(p * logp, axis=1)


def _best_threshold_classification(
    x: NDArray, y_codes: NDArray, n_classes: int, criterion: str, min_samples_leaf: int
) -> tuple[float, float] | None:
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y_codes[order]
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), ys] = 1.0
    cum = np.cumsum(onehot, axis=0)
    total = cum[-1]

    n_left = np.arange(1, n)
    n_right = n - n_left
    left_counts = cum[:-1]
    right_counts = total - left_counts

    parent = _impurity_from_counts(total[None, :], np.array([n]), criterion)[0]
    imp_left = _impurity_from_counts(left_counts, n_left, criterion)
    imp_right = _impurity_from_counts(right_counts, n_right, criterion)
    gain = parent - (n_left * imp_left + n_right * imp_right) / n

    valid = (xs[:-1] != xs[1:]) & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    if not valid.any():
        return None
    gain = np.where(valid, gain, -np.inf)
    i = int(np.argmax(gain))
    return float(gain[i]), float((xs[i] + xs[i + 1]) / 2.0)


def _best_threshold_regression(
    x: NDArray, y: NDArray, min_samples_leaf: int
) -> tuple[float, float] | None:
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], np.asarray(y, dtype=np.float64)[order]
    csum = np.cumsum(ys)
    csumsq = np.cumsum(ys * ys)

    n_left = np.arange(1, n)
    n_right = n - n_left
    sum_l, sumsq_l = csum[:-1], csumsq[:-1]
    sum_r, sumsq_r = csum[-1] - sum_l, csumsq[-1] - sumsq_l

    mse_l = sumsq_l / n_left - (sum_l / n_left) ** 2
    mse_r = sumsq_r / n_right - (sum_r / n_right) ** 2
    parent = csumsq[-1] / n - (csum[-1] / n) ** 2
    gain = parent - (n_left * mse_l + n_right * mse_r) / n

    valid = (xs[:-1] != xs[1:]) & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    if not valid.any():
        return None
    gain = np.where(valid, gain, -np.inf)
    i = int(np.argmax(gain))
    return float(gain[i]), float((xs[i] + xs[i + 1]) / 2.0)


def best_split(
    X: NDArray, y: NDArray, *, criterion: str = "gini", min_samples_leaf: int = 1
) -> Split | None:
    """Return the highest-gain split over all features, or ``None`` if none is valid."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, n_features = X.shape
    if n < 2 * min_samples_leaf:
        return None

    y_codes = np.zeros(0, dtype=np.intp)
    n_classes = 0
    if criterion != "mse":
        classes, y_codes = np.unique(y, return_inverse=True)
        n_classes = len(classes)

    best_feature, best_gain, best_threshold = -1, -np.inf, 0.0
    for feature in range(n_features):
        x = X[:, feature]
        if criterion == "mse":
            found = _best_threshold_regression(x, y, min_samples_leaf)
        else:
            found = _best_threshold_classification(
                x, y_codes, n_classes, criterion, min_samples_leaf
            )
        if found is None:
            continue
        gain, threshold = found
        if gain > best_gain:
            best_feature, best_gain, best_threshold = feature, gain, threshold

    if best_feature < 0:
        return None
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
        return None
    return Split(best_feature, best_threshold, float(best_gain), left_mask, right_mask)
