"""Decision Trees 101 — a small interactive companion to the notebooks.

Panels: split finder, tree grower, greedy-failure demo (checkerboard vs XOR), and
the overfitting dial. Run with ``make run``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src.datasets.synthetic import checkerboard, moons, noisy_blobs, two_blobs, xor
from src.evaluation.boundaries import plot_decision_boundary
from src.tree.classifier import DecisionTreeClassifier
from src.tree.splitter import best_split
from src.visualisation import export_text

st.set_page_config(page_title="Decision Trees 101", layout="wide")
st.title("🌳 Decision Trees 101")
st.caption("The greedy little algorithm inside every model you trust.")

tab_split, tab_grow, tab_greedy, tab_overfit = st.tabs(
    ["Split finder", "Tree grower", "Greedy failure", "Overfitting dial"]
)

with tab_split:
    st.subheader("One split, scored")
    n = st.slider("Points", 50, 400, 200, key="split_n")
    X, y = two_blobs(n_samples=n, seed=0)
    split = best_split(X, y, criterion="gini")
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_decision_boundary(stump, X, y, ax=ax)
    st.pyplot(fig)
    if split is not None:
        st.metric("Best split", f"x{split.feature} ≤ {split.threshold:.2f}")
        st.metric("Information gain", f"{split.gain:.3f}")

with tab_grow:
    st.subheader("Grow the tree, watch the staircase refine")
    depth = st.slider("max_depth", 1, 10, 3, key="grow_depth")
    X, y = moons(n_samples=400, noise=0.2, seed=0)
    clf = DecisionTreeClassifier(max_depth=depth).fit(X, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_decision_boundary(clf, X, y, ax=ax, title=f"depth = {depth}")
    st.pyplot(fig)
    st.code(export_text(clf), language="text")

with tab_greedy:
    st.subheader("Where greedy wins and where it fails")
    problem = st.radio("Problem", ["checkerboard (easy)", "XOR (hard)"], horizontal=True)
    X, y = checkerboard(seed=0) if problem.startswith("checkerboard") else xor(seed=0)
    root = best_split(X, y, criterion="gini")
    clf = DecisionTreeClassifier(max_depth=8).fit(X, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_decision_boundary(clf, X, y, ax=ax)
    st.pyplot(fig)
    if root is not None:
        st.metric("Information gain of the FIRST split", f"{root.gain:.4f}")
        st.write("On XOR the first split gains almost nothing — neither feature alone helps.")

with tab_overfit:
    st.subheader("Memorize, then regularize")
    noise = st.slider("Label noise", 0.0, 0.4, 0.25, key="of_noise")
    limit = st.checkbox("Limit the tree (max_depth=3, min_samples_leaf=10)", value=False)
    X, y = noisy_blobs(n_samples=400, noise=noise, seed=0)
    clf = (
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=10)
        if limit
        else DecisionTreeClassifier(max_depth=None)
    ).fit(X, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_decision_boundary(clf, X, y, ax=ax)
    st.pyplot(fig)
    st.write("Unlimited: the boundary wraps around each noisy point. Limited: it smooths out.")
