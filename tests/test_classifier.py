"""The from-scratch tree matches sklearn's accuracy on the same data and seed,
and the regression tree fits a nonlinear curve about as well as sklearn's.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkTree
from sklearn.tree import DecisionTreeRegressor as SkReg

from src.datasets.synthetic import moons, regression_curve, two_blobs
from src.evaluation.comparison import accuracy
from src.tree.classifier import DecisionTreeClassifier
from src.tree.regressor import DecisionTreeRegressor


def test_classifier_matches_sklearn_accuracy():
    X, y = moons(n_samples=400, noise=0.2, seed=0)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    ours = DecisionTreeClassifier(max_depth=5).fit(x_tr, y_tr)
    sk = SkTree(max_depth=5, criterion="gini", random_state=0).fit(x_tr, y_tr)

    acc_ours = accuracy(y_te, ours.predict(x_te))
    acc_sk = accuracy(y_te, sk.predict(x_te))
    assert acc_ours >= 0.8
    assert abs(acc_ours - acc_sk) <= 0.05


def test_classifier_separable_and_proba():
    X, y = two_blobs(n_samples=200, seed=1)
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert accuracy(y, clf.predict(X)) >= 0.98

    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_entropy_criterion_also_works():
    X, y = moons(n_samples=300, noise=0.2, seed=2)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5).fit(X, y)
    assert accuracy(y, clf.predict(X)) >= 0.85


def test_regressor_fits_curve():
    X, y = regression_curve(n_samples=200, noise=0.1, seed=0)
    ours = DecisionTreeRegressor(max_depth=5).fit(X, y)
    sk = SkReg(max_depth=5, random_state=0).fit(X, y)

    mse_ours = mean_squared_error(y, ours.predict(X))
    mse_sk = mean_squared_error(y, sk.predict(X))
    assert mse_ours <= mse_sk * 1.2 + 1e-9
