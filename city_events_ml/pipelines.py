from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor


@dataclass(frozen=True)
class FeatureSpec:
    categorical: tuple[str, ...] = ("neighborhood",)
    numeric: tuple[str, ...] = (
        "year",
        "month",
        "day",
        "dow_sin",
        "dow_cos",
        "hour_sin",
        "hour_cos",
    )


def make_preprocess(
    spec: FeatureSpec,
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    num_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(spec.categorical),
            ),
            ("num", num_transform, list(spec.numeric)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_ridge_pipeline(
    spec: FeatureSpec,
    *,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    preprocess = make_preprocess(spec, scale_numeric=True)
    model = Ridge(alpha=alpha, random_state=random_state)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def make_hgbr_pipeline(
    spec: FeatureSpec,
    *,
    random_state: int = 42,
    **model_kwargs,
) -> Pipeline:
    preprocess = make_preprocess(spec, scale_numeric=False)
    model = HistGradientBoostingRegressor(random_state=random_state, **model_kwargs)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def make_poisson_pipeline(
    spec: FeatureSpec,
    *,
    alpha: float = 1.0,
    max_iter: int = 2000,
) -> Pipeline:
    preprocess = make_preprocess(spec, scale_numeric=True)
    model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def make_dummy_pipeline(strategy: str = "mean") -> Pipeline:
    """
    Baseline model: predicts a constant (mean/median/quantile) regardless of features.

    Note: DummyRegressor ignores X, but we wrap it in a Pipeline for a consistent interface.
    """
    model = DummyRegressor(strategy=strategy)
    return Pipeline([("model", model)])


# Dummy pipeline usage
# from city_events_ml.pipelines import make_dummy_pipeline
# dummy_pipe = make_dummy_pipeline("mean")
