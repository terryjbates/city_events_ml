# city_events_ml/workflows.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .io import TargetSpec, build_target_table
from .features import add_time_features
from .pipelines import FeatureSpec


@dataclass(frozen=True)
class DatasetBundle:
    dense: pd.DataFrame
    feat: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def make_dataset_bundle(
    sf_dataframe: pd.DataFrame,
    *,
    category: str,
    neighborhoods: tuple[str, ...],
    start: str,
    end: str,
    freq: str = "6H",
    test_size: float = 0.15,
    feature_spec: FeatureSpec | None = None,
) -> DatasetBundle:
    spec = TargetSpec(
        category=category, neighborhoods=neighborhoods, freq=freq, start=start, end=end
    )
    dense = build_target_table(sf_dataframe, spec=spec)
    feat = (
        add_time_features(dense, time_col="interval_start")
        .sort_values("interval_start")
        .reset_index(drop=True)
    )

    split_idx = int(len(feat) * (1 - test_size))
    train_df = feat.iloc[:split_idx].copy()
    test_df = feat.iloc[split_idx:].copy()

    if feature_spec is None:
        feature_spec = FeatureSpec()

    feature_cols = list(feature_spec.categorical) + list(feature_spec.numeric)

    X_train = train_df[feature_cols]
    y_train = train_df["count"]
    X_test = test_df[feature_cols]
    y_test = test_df["count"]

    return DatasetBundle(
        dense, feat, train_df, test_df, X_train, y_train, X_test, y_test
    )
