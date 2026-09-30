"""Persist notebook artifacts: figures as PNG, textual/numerical results as JSON.

Every notebook saves its visuals to ``outputs/figures/*.png`` and its numbers and
text to ``outputs/data/*.json`` through these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
DATA_DIR = ROOT / "outputs" / "data"


def _to_serializable(obj: Any) -> Any:
    """Fallback encoder: turn numpy scalars/arrays into plain Python for ``json``."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, name: str) -> Path:
    """Write ``data`` as pretty JSON to ``outputs/data/<name>.json`` and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / (name if name.endswith(".json") else f"{name}.json")
    path.write_text(json.dumps(data, indent=2, default=_to_serializable))
    return path


def save_figure(fig: Any, name: str, dpi: int = 130) -> Path:
    """Save a matplotlib figure to ``outputs/figures/<name>.png`` and return the path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
