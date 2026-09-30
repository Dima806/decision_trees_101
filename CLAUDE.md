# CLAUDE.md — decision_trees_101

Operating guide for working in this repo. Read this before writing code, notebooks, or app changes. Source of truth for scope and intent is [.llm/PRD_decision_trees_101.md](.llm/PRD_decision_trees_101.md); this file is the working distillation.

> Placement note: this lives at the repo root so Claude Code auto-loads it as project memory. The PRD's file tree also references `.claude/CLAUDE.md` — treat this root file as canonical.

---

## 1. Identity & thesis

An educational **"101" portfolio project** that builds a decision tree from scratch — one `if`-statement up to a full recursive tree in ~80 lines of Python — to demystify the base learner inside random forests, gradient boosting, XGBoost, LightGBM, and CatBoost.

**The one claim everything serves:** a greedy algorithm that only ever asks *"is one feature above or below one threshold?"*, never looks more than one split ahead, nevertheless carves arbitrarily complex boundaries (spirals, checkerboards) — **and** the same greediness makes it fail XOR at the first split and memorize noisy data completely. Every ensemble hyperparameter (`max_depth`, `min_samples_leaf`, `criterion`, pruning) is a decision about *this* algorithm.

**Audience:** data scientists who use tree ensembles daily but have never built the tree underneath. Tone is explanatory and honest about the algorithm's weaknesses. Every claim is measured on known synthetic data, not asserted.

**Deliverables:** a `src/` library, 6 notebooks, a Streamlit app, and a test suite — all runnable on a 2-CPU / 8 GB GitHub Codespace with no GPU.

**Current state:** built and green. The `src/` library, all 6 notebooks (executed, with outputs embedded), the Streamlit app, and 18 tests exist; `make ci` passes (lint + ty + pytest). Notebook artifacts are generated under `outputs/`. Still to do (not started): `.devcontainer/devcontainer.json`, `.github/workflows/ci.yml`, `.claude/skills/caveman.md`. Do not re-scaffold — extend what's here.

---

## 2. Environment & stack

- **Python 3.11+** (project targets `py311`; do not use syntax newer than 3.11 even if a newer interpreter is present).
- **Package manager: `uv`. Never use `pip` directly.** Add deps via `uv add` / edit `pyproject.toml` then `uv sync --all-extras`. Run everything through `uv run ...`.
- **No GPU. 2 CPU cores, 8 GB RAM.** Keep datasets to hundreds–few thousand 2-D points so from-scratch trees build instantly and notebooks stay **under 3 minutes each**.
- **License: Apache-2.0.**

**Runtime deps:** `scikit-learn>=1.5`, `numpy>=1.26`, `scipy>=1.14`, `pandas>=2.2`, `matplotlib>=3.9`, `plotly>=5.22`, `streamlit>=1.38`, `pydantic>=2.9`, `pydantic-settings>=2.5`, `pyyaml>=6.0`.

**Dev deps:** `pytest>=8.3`, `ruff>=0.8`, `ty>=0.0.1a7`, `jupyter>=1.1`, `ipykernel>=6.29`, `nbconvert>=7.16`, `nbclient>=0.10`.

---

## 3. Repository structure

```
src/
  config.py                  # pydantic-settings, loads config/settings.yaml
  tree/
    criteria.py              # Gini, entropy, MSE impurity + information gain (from scratch)
    splitter.py              # Best-split search over every feature × threshold
    classifier.py            # Full recursive decision tree (~80 lines, from scratch)
    regressor.py             # Regression tree: MSE impurity, leaf means
    prune.py                 # max_depth / min_samples / min_impurity_decrease / cost-complexity
  datasets/
    synthetic.py             # spirals, moons, checkerboard, XOR, noisy blobs, regression curve
  evaluation/
    boundaries.py            # plot_decision_boundary, plot_regression_fit, boundary_grid
    overfitting.py           # train/test accuracy vs depth
    comparison.py            # accuracy; from-scratch vs sklearn; single tree vs forest
  visualisation.py           # export_text + plot_tree (node-link diagram)
  outputs.py                 # save_figure -> outputs/figures, save_json -> outputs/data
config/settings.yaml         # seed, data (n_samples/noise), tree (depth/min-samples/criterion), forest
notebooks/                   # 01_one_split → 06_from_tree_to_forest (executed, outputs embedded)
app/streamlit_app.py
tests/                       # test_criteria, test_splitter, test_classifier, test_prune (18 tests)
outputs/figures/             # 20 PNGs (01_*.png … 06_*.png)
outputs/data/                # 15 JSONs (01_*.json … 06_*.json)
```

Build order and layering: `criteria` → `splitter` (uses criteria) → `classifier`/`regressor` (use splitter) → `prune` → `evaluation` (uses trees + datasets). Notebooks and the app import from `src/`; they must not reimplement tree logic inline. `src` is installed as an editable package (hatchling, `packages = ["src"]`), so `import src.tree...` works everywhere.

**Notebook output convention (required):** every notebook saves visuals as PNG via `save_figure(fig, "0N_name")` → `outputs/figures/`, and every textual/numerical result as JSON via `save_json(obj, "0N_name")` → `outputs/data/` (both from `src.outputs`). Prefix each artifact with the notebook number.

---

## 4. Key concepts (get these right — they are the whole point)

- **Greedy build.** At each node: try every feature × every candidate threshold, score each split by impurity reduction (information gain), pick the single best, recurse. No lookahead, no backtracking. This loop *is* the tree.
- **Information gain = mutual information.** A split's information gain (impurity before − weighted impurity after, using entropy) equals the mutual information between the split indicator and the target. This identity is **asserted in tests** and is the concrete link to the information-theory project. "Splits to maximize information gain" and "splits to reduce uncertainty about the label" are the same statement.
- **Axis-aligned ⇒ staircase boundaries.** Every split is perpendicular to one axis, so boundaries are piecewise-constant staircases that *approximate* curves (moons, spirals) more finely as depth grows.
- **XOR defeats the greedy first split.** Two features whose combination is the label but where *neither feature alone carries any information* → information gain at the root is ~0 → the greedy search finds no good first split and stalls or takes a convoluted deep path. This is the showpiece and the reason forests exist. Demonstrate it with the exact info-gain-≈0-at-root computation, then show the convoluted deep solution. Frame it as **greediness, not a bug.**
- **Unpruned trees memorize.** Grown without limits on noisy data, a tree drives every leaf to purity, wrapping the boundary individually around each mislabeled point → 100% train accuracy, poor test accuracy. The most literal picture of overfitting in the series.
- **Pruning = regularization = ensemble hyperparameters.** `max_depth`, `min_samples_leaf`, `min_impurity_decrease`, and cost-complexity pruning are the fixes; they are the regularization idea applied to trees and the exact knobs people tune in boosted models. A reader who has built this understands them from the inside.
- **Trees' own feature importances are biased** — flag this honestly (link to the feature-selection project) rather than presenting tree importances as ground truth.

---

## 5. Commands (Makefile)

Run these, don't reinvent them:

- `make setup` — first-time: install uv if missing, `uv sync --all-extras`, register the `decision-trees-101` ipykernel.
- `make sync` — `uv sync --all-extras`.
- `make lint` — runs `format` + `check` + `typecheck`.
  - `make format` → `ruff format src/ tests/ app/`
  - `make check` → `ruff check --fix src/ tests/ app/`
  - `make typecheck` → `ty check src/`
- `make test` — `pytest` (18 tests). `make test-cov` for coverage (`pytest-cov`).
- `make notebooks` — execute all `notebooks/0*.ipynb` via nbconvert (180 s timeout each) under the `decision-trees-101` kernel. Notebooks must pass this clean. To re-execute with outputs embedded, use `--inplace`.
- `make run` — Streamlit app on port 8501. `make lab` — JupyterLab on 8888.
- `make ci` — `sync lint test` (what CI runs). `make dev` — fast `lint test` loop.

Before declaring any task done: `make lint` and `make test` must pass with zero errors.

---

## 6. Coding conventions

- **From-scratch means from scratch.** In `src/tree/`, implement impurity, splitting, recursion, and prediction yourself with `numpy`. Do **not** wrap `sklearn.tree` there. `scikit-learn` is used only for (a) comparison/validation in `evaluation/` and tests, and (b) the forest preview in notebook 06 and cost-complexity reference in `prune.py`.
- **Keep the classifier ~80 lines.** It's a headline claim ("about eighty lines"). Favor a small, readable recursive implementation over cleverness. Push impurity into `criteria.py` and split search into `splitter.py` so `classifier.py` stays lean.
- **Vectorize the splitter** with sorted-threshold scanning so it's instant on small 2-D data; correctness first, then keep it fast enough that no notebook exceeds 3 minutes.
- **Reproducibility is mandatory.** All datasets are synthetic and **seeded**; seeds live in `config/settings.yaml` and load via `pydantic-settings` in `src/config.py`. Never hardcode a magic seed inline in a notebook — pull it from config so results are stable across runs.
- **Style:** ruff with `line-length = 99`, `target-version = "py311"`, lint rules `E,F,W,I,UP,N,B,A,SIM,PTH`; ignores are `E501` (formatter handles wrapping) and `N803`/`N806` (allow the sklearn-idiomatic `X`). Type-annotate public functions in `src/`; `ty check src/` must be clean (ty is strict about numpy overloads — prefer `len(np.unique(...))` over `arr.max()` where it complains).
- **Artifacts** go through `src.outputs`: figures → `outputs/figures/*.png`, numbers/text → `outputs/data/*.json`. Don't call `plt.savefig`/`json.dump` directly in notebooks.

---

## 7. Testing & success criteria (assert, don't hand-wave)

Tests encode the PRD's claims — mirror these when adding code:

- `test_criteria.py` — impurity and information gain **match hand calculations exactly**; **information gain equals the split's mutual information within tolerance**.
- `test_splitter.py` — from-scratch best split picks the **same feature and threshold as sklearn**.
- `test_classifier.py` — from-scratch tree matches **sklearn accuracy within tolerance** on the same data and seed.
- `test_prune.py` — a depth limit / pruning **reduces the train-test gap** (asserted).

Demonstrated-but-measured behaviors the notebooks must show: staircase boundary on moons/spirals improving with depth; XOR struggle with the info-gain-≈0-at-root computation; unlimited tree → 100% train / poor test on noisy data; forest recovers XOR and stops memorizing; regression tree fits the nonlinear curve in piecewise-constant steps.

---

## 8. Notebooks (order is pedagogy — keep it)

1. **01_one_split** — one `if`: find the single best threshold on two blobs by hand, draw the one-split boundary.
2. **02_growing_a_tree** — make the split recursive; show depth-1 / depth-3 / deep on moons, staircase refining, tree diagram growing alongside.
3. **03_impurity_and_information_gain** — Gini vs entropy (barely differ), the info-gain = mutual-information reveal (link to information-theory project).
4. **04_where_greedy_fails** — root gain of XOR (~0.02) and the 2×2 checkerboard (~0.01) vs an *easy* baseline, moons (~0.21). ⚠️ A 2×2 checkerboard **is** XOR, so both stall at the root — the honest contrast is against an informative-first-split problem (moons), not "checkerboard has high root gain."
5. **05_overfitting_and_pruning** — unpruned memorization on noisy data (train 1.0 / test ~0.65), then depth/min-samples/cost-complexity fixes; train-test-gap-vs-depth curve; ccp path; pruning = regularization.
6. **06_from_tree_to_forest** — greedy stump (0.54) vs forest (1.0) on XOR, and single full tree (0.75) vs forest (0.81) on noisy moons. ⚠️ Pass full depth to the *forest* (don't cap it at the stump depth); a single deep tree also solves clean XOR, so the forest's honest, measurable win is variance reduction on noisy data. Practical `max_depth`/`min_samples_leaf`/`criterion` guide; biased-importance caveat (impurity vs permutation).

Each notebook: runs top-to-bottom clean via `make notebooks`, **under 3 minutes**, imports tree logic from `src/` (no inline reimplementation), and saves its PNG/JSON artifacts via `src.outputs`.

---

## 9. Datasets & app

- **Datasets** (`src/datasets/synthetic.py`, all synthetic + seeded so structure is known and boundaries are 2-D drawable): spirals, moons, checkerboard, XOR, noisy blobs, and a nonlinear regression curve.
- **Streamlit app** (`app/streamlit_app.py`) — four panels: (1) split finder (place points, watch threshold search + information gain), (2) tree grower (depth slider grows staircase + diagram together), (3) greedy-failure demo (toggle checkerboard ↔ XOR), (4) overfitting dial (add label noise, grow unlimited, then apply limits and watch it un-memorize).

---

## 10. Sibling projects (cross-link, don't duplicate)

Part of the [Dima806](https://github.com/Dima806) "101/arena" series; keep the handoffs explicit. Directly cross-linked:
- [ensembles_101](https://github.com/Dima806/ensembles_101) — the sequel; forests/boosting built from these trees. Notebook 06 is the explicit handoff.
- [information_theory_101](https://github.com/Dima806/information_theory_101) — entropy, cross-entropy, mutual information; the info-gain = MI identity (notebook 03).
- [regularization_101](https://github.com/Dima806/regularization_101) — the idea that makes models generalize instead of memorize; pruning/depth are it (notebook 05).
- [feature_selection_arena](https://github.com/Dima806/feature_selection_arena) — why a tree's own impurity importances mislead (notebook 06 caveat).

Future-extension neighbors:
- [encoding_arena](https://github.com/Dima806/encoding_arena) — high-cardinality categorical strategies (how CatBoost/LightGBM split on categories).
- [missing_data_imputation_arena](https://github.com/Dima806/missing_data_imputation_arena) — missing values (surrogate splits / native NaN handling).
- [explainability_arena](https://github.com/Dima806/explainability_arena) — SHAP/LIME/permutation vs a tree's own decision paths.
- [bootstrap_101](https://github.com/Dima806/bootstrap_101) — resampling, the foundation of the forest's bagging in notebook 06.

---

## 11. Token & workflow rules

- **Caveman mode: ON.** Code-first, minimal prose in responses; let the code and figures carry the explanation.
- `/compact` at notebook boundaries.
- Commit or push only when asked; branch off `main` first if so.
