"""Evaluation helpers: boundaries, overfitting curves, and comparisons."""

from .boundaries import boundary_grid, plot_decision_boundary, plot_regression_fit
from .comparison import accuracy, compare_to_sklearn, tree_vs_forest
from .overfitting import train_test_accuracy_vs_depth

__all__ = [
    "accuracy",
    "boundary_grid",
    "compare_to_sklearn",
    "plot_decision_boundary",
    "plot_regression_fit",
    "train_test_accuracy_vs_depth",
    "tree_vs_forest",
]
