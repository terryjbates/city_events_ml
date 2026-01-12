from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": mean_squared_error(y_true, y_pred, multioutput='raw_values'),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MaxError": max_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


@dataclass
class MetricsStore:
    rows: list[dict] = field(default_factory=list)

    def log(
        self,
        *,
        model: str,
        split: str,  # "train" or "test" or "cv"
        metrics: dict[str, float],
        extra: dict | None = None,
    ) -> None:
        row = {"model": model, "split": split, **metrics}
        if extra:
            row.update(extra)
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        # nice pivot similar to what you’ve been printing
        if {"model", "split"}.issubset(df.columns):
            return df.set_index(["model", "split"]).sort_index()
        return df


def evaluate_model(
    model_name: str,
    pipeline,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    store: MetricsStore | None = None,
) -> pd.DataFrame:
    pipeline.fit(X_train, y_train)

    yhat_tr = pipeline.predict(X_train)
    yhat_te = pipeline.predict(X_test)

    tr = compute_metrics(y_train, yhat_tr)
    te = compute_metrics(y_test, yhat_te)

    if store is not None:
        store.log(model=model_name, split="train", metrics=tr)
        store.log(model=model_name, split="test", metrics=te)

    return pd.DataFrame([{"model": model_name, "split": "train", **tr},
                         {"model": model_name, "split": "test", **te}]).set_index(["model","split"])
