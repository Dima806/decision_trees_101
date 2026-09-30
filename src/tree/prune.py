"""Stopping rules and pruning — the mechanisms that stop a tree from memorizing.

Maximum depth, minimum samples per leaf, and minimum impurity decrease are
enforced during growth (see ``BaseDecisionTree``). This module adds
cost-complexity pruning: a fitted tree is post-pruned by repeatedly collapsing
the "weakest link" — the internal node whose subtree buys the least accuracy per
extra leaf — until every remaining link is worth more than ``ccp_alpha``. Each
of these is the regularization idea applied to trees, and each is an ensemble
hyperparameter the reader now understands from the inside.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, TypeVar

from .classifier import Node

if TYPE_CHECKING:
    from .classifier import BaseDecisionTree

TreeT = TypeVar("TreeT", bound="BaseDecisionTree")


def n_leaves(node: Node) -> int:
    """Number of leaves in the subtree rooted at ``node``."""
    if node.is_leaf:
        return 1
    assert node.left is not None and node.right is not None
    return n_leaves(node.left) + n_leaves(node.right)


def n_nodes(node: Node) -> int:
    """Total number of nodes (internal + leaves) in the subtree."""
    if node.is_leaf:
        return 1
    assert node.left is not None and node.right is not None
    return 1 + n_nodes(node.left) + n_nodes(node.right)


def tree_depth(node: Node) -> int:
    """Depth of the subtree (a single leaf has depth 0)."""
    if node.is_leaf:
        return 0
    assert node.left is not None and node.right is not None
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def _leaf_cost(node: Node, n_total: int) -> float:
    """Cost R(t) of turning ``node`` into a leaf: its impurity weighted by coverage."""
    return node.impurity * node.n_samples / n_total


def _subtree_cost(node: Node, n_total: int) -> float:
    """Cost R(T_t) of the subtree: sum of leaf costs it currently produces."""
    if node.is_leaf:
        return _leaf_cost(node, n_total)
    assert node.left is not None and node.right is not None
    return _subtree_cost(node.left, n_total) + _subtree_cost(node.right, n_total)


def _alpha_eff(node: Node, n_total: int) -> float:
    """Effective alpha: extra cost per extra leaf that keeping this subtree buys."""
    return (_leaf_cost(node, n_total) - _subtree_cost(node, n_total)) / (n_leaves(node) - 1)


def _internal_nodes(node: Node, out: list[Node]) -> None:
    if not node.is_leaf:
        assert node.left is not None and node.right is not None
        out.append(node)
        _internal_nodes(node.left, out)
        _internal_nodes(node.right, out)


def _collapse(node: Node) -> None:
    """Turn an internal node into a leaf (its ``value`` was set during growth)."""
    node.feature = None
    node.threshold = None
    node.left = None
    node.right = None


def cost_complexity_prune(tree: TreeT, ccp_alpha: float) -> TreeT:
    """Return a copy of ``tree`` weakest-link pruned at complexity ``ccp_alpha``.

    ``ccp_alpha = 0`` returns the tree unchanged; larger values prune more, and a
    large enough value collapses the tree to its root.
    """
    pruned = copy.copy(tree)
    pruned.root_ = copy.deepcopy(tree.root_)
    n_total = pruned.root_.n_samples
    while True:
        internal: list[Node] = []
        _internal_nodes(pruned.root_, internal)
        if not internal:
            break
        weakest = min(internal, key=lambda node: _alpha_eff(node, n_total))
        if _alpha_eff(weakest, n_total) > ccp_alpha:
            break
        _collapse(weakest)
    return pruned
