"""Typed settings loaded from ``config/settings.yaml`` via pydantic-settings.

Notebooks and the app import :func:`load_settings` so every run is seeded and
reproducible; nothing hardcodes a magic seed inline.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Synthetic-dataset defaults."""

    n_samples: int = 400
    noise: float = 0.2


class TreeConfig(BaseModel):
    """Decision-tree hyperparameters — the knobs this whole project is about."""

    criterion: str = "gini"
    max_depth: int | None = 5
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_impurity_decrease: float = 0.0


class ForestConfig(BaseModel):
    """Forest-preview settings for notebook 06."""

    n_estimators: int = 200
    max_samples: float = 0.8


class Settings(BaseSettings):
    """Top-level project settings."""

    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    tree: TreeConfig = Field(default_factory=TreeConfig)
    forest: ForestConfig = Field(default_factory=ForestConfig)


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from ``config/settings.yaml``, falling back to defaults."""
    path = path or DEFAULT_PATH
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        return Settings(**data)
    return Settings()
