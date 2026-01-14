from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score
from sklearn.inspection import permutation_importance


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
        split: str,
        metrics: dict[str, float],
        category: str | None = None,
        extra: dict | None = None,
    ) -> None:
        row = {"model": model, "split": split, **metrics}
        if category is not None:
            row["category"] = category
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
    category: str | None = None, 
) -> pd.DataFrame:
    pipeline.fit(X_train, y_train)

    yhat_tr = pipeline.predict(X_train)
    yhat_te = pipeline.predict(X_test)

    tr = compute_metrics(y_train, yhat_tr)
    te = compute_metrics(y_test, yhat_te)

    if store is not None:
        store.log(model=model_name, split="train", metrics=tr, category=category)
        store.log(model=model_name, split="test", metrics=te, category=category)

    return pd.DataFrame([{"model": model_name, "split": "train", **tr},
                         {"model": model_name, "split": "test", **te}]).set_index(["model","split"])


def get_feature_names(pipeline) -> list[str]:
    pre = pipeline.named_steps["preprocess"]
    feature_names = []

    for name, transformer, cols in pre.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)

    return feature_names


def linear_feature_importance(pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    coefs = model.coef_.ravel()
    importance = np.abs(coefs)

    return (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "importance": importance,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def hgbr_permutation_importance(
    pipeline,
    X_test,
    y_test,
    *,
    scoring: str = "neg_root_mean_squared_error",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance computed on the *raw input features* (columns of X_test),
    which is the correct interpretation when passing a full sklearn Pipeline.

    Returns a dataframe sorted by importance_mean descending.
    """
    feature_names = list(X_test.columns)

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    if len(feature_names) != len(perm.importances_mean):
        raise ValueError(
            f"Feature name length mismatch: X_test has {len(feature_names)} cols "
            f"but permutation_importance returned {len(perm.importances_mean)}."
        )

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )



def plot_feature_importance(df, value_col, title, top_n=15):
    df = df.head(top_n)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df[value_col])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
