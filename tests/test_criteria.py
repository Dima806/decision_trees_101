"""Impurity and information gain match hand calculations, and information gain
(with entropy) equals the mutual information between the split and the target.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mutual_info_score

from src.tree.criteria import (
    entropy,
    gini,
    information_gain,
    mse,
    mutual_information,
)


def test_gini_pure_is_zero():
    assert gini(np.array([1, 1, 1, 1])) == 0.0


def test_gini_balanced_is_half():
    assert abs(gini(np.array([0, 0, 1, 1])) - 0.5) < 1e-12


def test_entropy_balanced_is_one_bit():
    assert abs(entropy(np.array([0, 0, 1, 1])) - 1.0) < 1e-12


def test_entropy_pure_is_zero():
    assert entropy(np.array([2, 2, 2])) == 0.0


def test_mse_constant_is_zero():
    assert mse(np.array([3.0, 3.0, 3.0])) == 0.0


def test_mse_known_value():
    # values {0, 2}: mean 1, squared deviations {1, 1}, mean 1.0
    assert abs(mse(np.array([0.0, 2.0])) - 1.0) < 1e-12


def test_information_gain_perfect_split():
    y = np.array([0, 0, 1, 1])
    left = np.array([True, True, False, False])
    assert abs(information_gain(y, left, "entropy") - 1.0) < 1e-12
    assert abs(information_gain(y, left, "gini") - 0.5) < 1e-12


def test_information_gain_equals_mutual_information():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    # y depends on x, so the split carries real information
    y = (rng.uniform(size=400) < np.where(x > 0.0, 0.75, 0.2)).astype(int)
    left = x <= 0.0

    ig = information_gain(y, left, "entropy")
    mi_bits = mutual_information(left, y)
    assert abs(ig - mi_bits) < 1e-9

    # cross-check against sklearn (which reports nats -> convert to bits)
    mi_sklearn = mutual_info_score(left.astype(int), y) / np.log(2)
    assert abs(ig - mi_sklearn) < 1e-9
