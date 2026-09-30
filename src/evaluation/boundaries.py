"""Decision-boundary and regression-fit rendering with matplotlib.

Drawing the staircase boundary is how the greedy algorithm's axis-aligned nature
becomes visible: every edge is perpendicular to an axis.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


class _Predictor(Protocol):
    def predict(self, X: NDArray) -> NDArray: ...


def plot_decision_boundary(
    model: _Predictor,
    X: NDArray,
    y: NDArray,
    ax: Axes | None = None,
    *,
    resolution: int = 300,
    title: str | None = None,
) -> Axes:
    """Fill the plane with the model's predictions and overlay the training points."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    return ax


def plot_regression_fit(
    model: _Predictor,
    X: NDArray,
    y: NDArray,
    ax: Axes | None = None,
    *,
    title: str | None = None,
) -> Axes:
    """Scatter the 1-D data and draw the tree's piecewise-constant prediction."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(X[:, 0])
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 500).reshape(-1, 1)
    ax.scatter(X[:, 0], y, s=15, alpha=0.6, label="data")
    ax.plot(grid[:, 0], model.predict(grid), color="crimson", lw=2, label="tree")
    _ = order
    ax.legend()
    if title is not None:
        ax.set_title(title)
    return ax


def boundary_grid(model: _Predictor, X: NDArray, resolution: int = 300) -> dict[str, Any]:
    """Return the mesh and predictions (for the Streamlit app / custom plots)."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return {"xx": xx, "yy": yy, "zz": zz}
