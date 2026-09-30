"""The recursive decision tree, from scratch.

``BaseDecisionTree`` is the whole algorithm: grow a node, and unless a stopping
rule fires, find the best split and recurse on each side. The classifier below
predicts by the majority class in each leaf; the regressor (``regressor.py``)
reuses this exact base with MSE impurity and leaf means. One idea, both tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .criteria import impurity
from .splitter import best_split


@dataclass
class Node:
    """A tree node. Leaves have ``feature is None``; ``value`` is always set."""

    n_samples: int
    depth: int
    impurity: float
    value: NDArray
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class BaseDecisionTree:
    """Shared growing/prediction machinery for the classifier and regressor."""

    criterion: str = "gini"
    root_: Node
    n_features_in_: int

    def __init__(
        self,
        *,
        criterion: str | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        if criterion is not None:
            self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    # --- hooks the subclasses fill in ---
    def _setup(self, y: NDArray) -> None:
        """Record any target metadata (e.g. classes) before growing."""

    def _leaf_value(self, y: NDArray) -> NDArray:
        raise NotImplementedError

    def _node_impurity(self, y: NDArray) -> float:
        return impurity(y, self.criterion)

    # --- the algorithm ---
    def fit(self, X: NDArray, y: NDArray) -> BaseDecisionTree:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self._setup(y)
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: NDArray, y: NDArray, depth: int) -> Node:
        node = Node(
            n_samples=len(y),
            depth=depth,
            impurity=self._node_impurity(y),
            value=self._leaf_value(y),
        )
        if (
            node.impurity <= 1e-12
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
        ):
            return node
        split = best_split(X, y, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf)
        if split is None or split.gain < self.min_impurity_decrease:
            return node
        node.feature = split.feature
        node.threshold = split.threshold
        node.left = self._grow(X[split.left_mask], y[split.left_mask], depth + 1)
        node.right = self._grow(X[split.right_mask], y[split.right_mask], depth + 1)
        return node

    def _route(self, x: NDArray) -> Node:
        node = self.root_
        while node.feature is not None:
            assert node.threshold is not None and node.left is not None and node.right is not None
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X: NDArray) -> NDArray:
        raise NotImplementedError


class DecisionTreeClassifier(BaseDecisionTree):
    """A classification tree. Leaves store class probabilities; predict by majority."""

    criterion = "gini"
    classes_: NDArray
    n_classes_: int

    def _setup(self, y: NDArray) -> None:
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

    def _leaf_value(self, y: NDArray) -> NDArray:
        counts = np.zeros(self.n_classes_, dtype=np.float64)
        present, present_counts = np.unique(y, return_counts=True)
        counts[np.searchsorted(self.classes_, present)] = present_counts
        return counts / counts.sum()

    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._route(x).value for x in X])

    def predict(self, X: NDArray) -> NDArray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
