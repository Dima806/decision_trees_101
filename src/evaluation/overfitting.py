"""Train/test accuracy versus depth — the overfitting curve.

As depth grows, training accuracy climbs toward 1.0 while test accuracy peaks and
then falls: the train-test gap is the picture of a tree memorizing.
"""

from __future__ import annotations

from collections.abc import Iterable

from numpy.typing import NDArray

from ..tree.classifier import DecisionTreeClassifier
from .comparison import accuracy


def train_test_accuracy_vs_depth(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    depths: Iterable[int],
    *,
    criterion: str = "gini",
) -> list[dict[str, float]]:
    """For each depth, fit a tree and record train and test accuracy."""
    rows: list[dict[str, float]] = []
    for depth in depths:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth).fit(X_train, y_train)
        train_acc = accuracy(y_train, clf.predict(X_train))
        test_acc = accuracy(y_test, clf.predict(X_test))
        rows.append(
            {
                "depth": float(depth),
                "train": train_acc,
                "test": test_acc,
                "gap": train_acc - test_acc,
            }
        )
    return rows
