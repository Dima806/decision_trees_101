"""Rendering a fitted tree — see the greedy split-scoring result directly.

``export_text`` prints the learned splits the way scikit-learn's does, so a
reader can read off exactly which feature and threshold each node chose;
``plot_tree`` draws the same structure as a node-link diagram that grows with
depth alongside the decision boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .tree.classifier import BaseDecisionTree, Node
from .tree.prune import tree_depth


def export_text(
    tree: BaseDecisionTree, feature_names: list[str] | None = None, precision: int = 3
) -> str:
    """Return an indented text description of the fitted tree's splits and leaves."""
    lines: list[str] = []

    def name(feature: int) -> str:
        return feature_names[feature] if feature_names is not None else f"x{feature}"

    def recurse(node: Node, depth: int) -> None:
        indent = "|   " * depth
        if node.is_leaf:
            value = np.round(node.value, precision).tolist()
            lines.append(f"{indent}|--- value: {value}")
            return
        assert node.feature is not None and node.threshold is not None
        assert node.left is not None and node.right is not None
        lines.append(f"{indent}|--- {name(node.feature)} <= {node.threshold:.{precision}f}")
        recurse(node.left, depth + 1)
        lines.append(f"{indent}|--- {name(node.feature)} >  {node.threshold:.{precision}f}")
        recurse(node.right, depth + 1)

    recurse(tree.root_, 0)
    return "\n".join(lines)


def plot_tree(
    tree: BaseDecisionTree,
    ax: Any = None,
    *,
    feature_names: list[str] | None = None,
    precision: int = 2,
) -> Any:
    """Draw the fitted tree as a node-link diagram (internal nodes blue, leaves orange)."""
    import matplotlib.pyplot as plt

    positions: dict[int, tuple[float, float]] = {}
    counter = [0]

    def assign(node: Node, depth: int) -> None:
        if node.is_leaf:
            positions[id(node)] = (float(counter[0]), float(-depth))
            counter[0] += 1
            return
        assert node.left is not None and node.right is not None
        assign(node.left, depth + 1)
        assign(node.right, depth + 1)
        mid = (positions[id(node.left)][0] + positions[id(node.right)][0]) / 2.0
        positions[id(node)] = (mid, float(-depth))

    assign(tree.root_, 0)

    if ax is None:
        width = 1.7 * max(counter[0], 1)
        height = 1.3 * (tree_depth(tree.root_) + 1)
        _, ax = plt.subplots(figsize=(width, height))

    def label(node: Node) -> str:
        if node.is_leaf:
            return f"n={node.n_samples}\n{np.round(node.value, precision).tolist()}"
        assert node.feature is not None and node.threshold is not None
        name = feature_names[node.feature] if feature_names is not None else f"x{node.feature}"
        return f"{name} ≤ {node.threshold:.{precision}f}"

    def draw(node: Node) -> None:
        x, y = positions[id(node)]
        if not node.is_leaf:
            assert node.left is not None and node.right is not None
            for child in (node.left, node.right):
                cx, cy = positions[id(child)]
                ax.plot([x, cx], [y, cy], color="0.6", lw=1.0, zorder=1)
                draw(child)
        face = "#fde9e0" if node.is_leaf else "#e7f0fb"
        ax.text(
            x,
            y,
            label(node),
            ha="center",
            va="center",
            fontsize=8,
            zorder=2,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": face, "edgecolor": "0.5"},
        )

    draw(tree.root_)
    ax.axis("off")
    ax.margins(0.1)
    return ax
