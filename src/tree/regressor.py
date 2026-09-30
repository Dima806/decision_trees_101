"""The regression tree: the same recursive algorithm, MSE impurity, leaf means.

Demonstrates that a decision tree is one idea serving both tasks — swap the
impurity (Gini/entropy -> MSE) and the leaf summary (majority vote -> mean) and
nothing else changes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .classifier import BaseDecisionTree


class DecisionTreeRegressor(BaseDecisionTree):
    """A regression tree. Each leaf predicts the mean target of its samples."""

    criterion = "mse"

    def _leaf_value(self, y: NDArray) -> NDArray:
        return np.array([float(np.mean(y))])

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._route(x).value[0] for x in X])
