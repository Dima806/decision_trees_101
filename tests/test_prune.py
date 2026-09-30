"""An unlimited tree memorizes noisy data; a depth limit reduces the train-test
gap; and cost-complexity pruning shrinks the tree.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.datasets.synthetic import noisy_blobs
from src.evaluation.comparison import accuracy
from src.tree.classifier import DecisionTreeClassifier
from src.tree.prune import cost_complexity_prune, n_nodes


def test_unlimited_tree_memorizes_and_depth_limit_reduces_gap():
    X, y = noisy_blobs(n_samples=500, noise=0.25, seed=0)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    full = DecisionTreeClassifier(max_depth=None).fit(x_tr, y_tr)
    shallow = DecisionTreeClassifier(max_depth=3).fit(x_tr, y_tr)

    # Unlimited tree drives training accuracy to (essentially) 1.0 — memorization.
    assert accuracy(y_tr, full.predict(x_tr)) >= 0.99

    gap_full = accuracy(y_tr, full.predict(x_tr)) - accuracy(y_te, full.predict(x_te))
    gap_shallow = accuracy(y_tr, shallow.predict(x_tr)) - accuracy(y_te, shallow.predict(x_te))
    assert gap_shallow < gap_full


def test_cost_complexity_pruning_shrinks_tree():
    X, y = noisy_blobs(n_samples=500, noise=0.25, seed=1)
    full = DecisionTreeClassifier(max_depth=None).fit(X, y)

    unpruned_nodes = n_nodes(full.root_)
    assert unpruned_nodes > 1

    # alpha = 0 is a no-op; a large alpha collapses to the root.
    assert n_nodes(cost_complexity_prune(full, ccp_alpha=0.0).root_) == unpruned_nodes
    assert n_nodes(cost_complexity_prune(full, ccp_alpha=10.0).root_) == 1

    # a moderate alpha prunes strictly between those extremes
    moderate = n_nodes(cost_complexity_prune(full, ccp_alpha=0.01).root_)
    assert 1 <= moderate <= unpruned_nodes


def test_pruning_does_not_mutate_original():
    X, y = noisy_blobs(n_samples=300, noise=0.2, seed=2)
    full = DecisionTreeClassifier(max_depth=None).fit(X, y)
    before = n_nodes(full.root_)
    cost_complexity_prune(full, ccp_alpha=10.0)
    assert n_nodes(full.root_) == before
