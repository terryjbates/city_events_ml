from __future__ import annotations

from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class CVSpec:
    n_splits: int = 5


def make_time_series_cv(spec: CVSpec) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=spec.n_splits)


def hgbr_param_distributions() -> dict[str, list]:
    # small, fast ranges
    return {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "model__max_depth": [None, 3, 5, 7, 9],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 50, 100],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
    }
