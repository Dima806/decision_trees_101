"""From-scratch vs scikit-learn, and single tree vs forest.

These helpers back the claims: our tree matches sklearn's accuracy, and a forest
recovers what one greedy tree cannot (XOR, and not memorizing noise).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    """Fraction of labels predicted correctly."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def compare_to_sklearn(
    X: NDArray,
    y: NDArray,
    *,
    max_depth: int | None = 5,
    criterion: str = "gini",
    random_state: int = 0,
) -> dict[str, float]:
    """Train accuracy of our tree vs sklearn's on the same data and hyperparameters."""
    from sklearn.tree import DecisionTreeClassifier as SkTree

    from ..tree.classifier import DecisionTreeClassifier

    ours = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth).fit(X, y)
    sk = SkTree(criterion=criterion, max_depth=max_depth, random_state=random_state).fit(X, y)
    return {"ours": accuracy(y, ours.predict(X)), "sklearn": accuracy(y, sk.predict(X))}


def tree_vs_forest(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    *,
    max_depth: int | None = None,
    n_estimators: int = 200,
    random_state: int = 0,
) -> dict[str, float]:
    """Test accuracy of a single tree vs a random forest — the bridge to ensembles."""
    from sklearn.ensemble import RandomForestClassifier

    from ..tree.classifier import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
    forest: Any = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    ).fit(X_train, y_train)
    return {
        "tree": accuracy(y_test, tree.predict(X_test)),
        "forest": accuracy(y_test, forest.predict(X_test)),
    }
