"""The from-scratch best split picks the same feature and threshold as sklearn."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SkTree

from src.datasets.synthetic import two_blobs
from src.tree.splitter import best_split


def test_best_split_matches_sklearn():
    X, y = two_blobs(n_samples=200, seed=0)
    split = best_split(X, y, criterion="gini")
    sk = SkTree(max_depth=1, criterion="gini", random_state=0).fit(X, y)

    assert split is not None
    assert split.feature == int(sk.tree_.feature[0])
    assert abs(split.threshold - float(sk.tree_.threshold[0])) < 1e-6


def test_best_split_respects_min_samples_leaf():
    X, y = two_blobs(n_samples=100, seed=1)
    split = best_split(X, y, criterion="gini", min_samples_leaf=10)
    assert split is not None
    assert split.left_mask.sum() >= 10
    assert split.right_mask.sum() >= 10


def test_no_split_when_features_constant():
    # No distinct threshold exists on any feature -> no valid split.
    X = np.ones((20, 2))
    y = np.array([0, 1] * 10)
    assert best_split(X, y, criterion="gini") is None
