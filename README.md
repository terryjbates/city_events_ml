# city-event-ml

Train and evaluate machine learning pipelines that predict the **frequency (count)** of San Francisco safety events per **6-hour interval** and **neighborhood**, with one model per event category.

This repository contains:

- A reusable Python package: `city_events_ml/`
- Notebooks under `notebooks/` for model training and evaluation
- Lightweight tests under `tests/`

---
## Project goal

Given SF 311-style call/event data, build a model that predicts:

> the number of events in a given neighborhood during a given 6-hour time window.

The workflow is designed to satisfy an assignment requiring:

- Aggregation into 6-hour bins for top neighborhoods
- Time-aware train/test split (no shuffle; test occurs later in time)
- Feature engineering from timestamps (year/month/day/dow/hour + cyclical transforms)
- Neighborhood encoding (OneHotEncoder via ColumnTransformer)
- Baseline (DummyRegressor), linear model (Ridge), and a stronger model (HistGradientBoostingRegressor)
- Evaluation via RMSE / MAE / MaxError on train and test
- Optional model persistence using `skops`

---
## How to use this package (functional guide)

This package is designed to support an end-to-end workflow:

1. Convert raw event records (one row per event) into a **dense target table** of counts per neighborhood per time bin.
2. Add **time-based features** (including cyclical encodings) suitable for modeling.
3. Build **scikit-learn pipelines** that include preprocessing + model in one object.
4. Train/evaluate models using time-aware splits and log results into a **MetricsStore**.
5. Save the best model pipelines to disk using **skops** for reuse.

The public API is organized to match that workflow.

---

### 1) `city_events_ml.io`

#### `TargetSpec`
A small configuration object describing how to build the modeling dataset:

- `category`: which event category to model (e.g., `"Potentially Life-Threatening"`)
- `neighborhoods`: which neighborhoods to include
- `start`, `end`: time window (end is exclusive)
- `freq`: bin size (default `"6H"`)

Why it exists:
- Keeps dataset parameters explicit and repeatable.
- Makes it easy to run the same pipeline for multiple categories.

#### `build_target_table(sf_dataframe, spec)`
Takes raw events and returns a **dense** target table with one row per:

- `(neighborhood, interval_start)` where `interval_start` is a 6-hour bin boundary
- `count` = number of events in that bin

Key behaviors:
- Filters by `spec.category` and `spec.neighborhoods`
- Bins timestamps into 6-hour intervals
- **Reindexes** each neighborhood to fill missing intervals with `count = 0`

Output columns:
- `neighborhood` (string)
- `interval_start` (datetime)
- `count` (int)

Why density matters:
- Missing intervals are real “zero-event” observations. Filling them improves training and evaluation.

---

### 2) `city_events_ml.features`

#### `add_time_features(df, time_col="interval_start")`
Adds time-derived feature columns based on the bin timestamp:

- `year`, `month`, `day`
- `dow` (day of week, 0–6)
- `hour` (0, 6, 12, 18)
- cyclical encodings:
  - `dow_sin`, `dow_cos`
  - `hour_sin`, `hour_cos`

Why sine/cosine:
- Day-of-week and hour are cyclical (Sunday wraps to Monday; 18 wraps to 0).
- Sine/cosine preserves the idea that “nearby” times should have “nearby” representations.

This function intentionally does **not** create lag features by default to avoid leakage mistakes.
Lag features can be added later once the baseline is stable.

---

### 3) `city_events_ml.pipelines`

#### `FeatureSpec`
Defines which feature columns are treated as categorical vs numeric.

Typical setup for this project:
- categorical: `("neighborhood",)`
- numeric: time-derived features (year/month/day + cyclical sin/cos)

Why it exists:
- Keeps the preprocessing pipeline consistent across models.
- Makes it easy to reuse across categories.

#### `make_dummy_pipeline(strategy="mean")`
Returns a baseline pipeline using `DummyRegressor`.
This model ignores features and predicts a constant (mean by default).

Why it exists:
- Establishes a floor. If your “real” model can’t beat Dummy, you’re not learning signal.

#### `make_ridge_pipeline(feature_spec, alpha=1.0)`
Pipeline:
- ColumnTransformer:
  - OneHotEncode categorical features (neighborhood)
  - Scale numeric features
- Ridge regression

Why it exists:
- Fast baseline linear model for comparison.

#### `make_hgbr_pipeline(feature_spec, **kwargs)`
Pipeline:
- ColumnTransformer (OneHotEncoder for neighborhood, passthrough numeric)
- HistGradientBoostingRegressor

Why it exists:
- Strong non-linear model that typically performs best on this task.

#### `make_poisson_pipeline(feature_spec, alpha=1.0)`
Pipeline:
- ColumnTransformer
- PoissonRegressor (log-link for counts)

Why it exists:
- Count-aware model that guarantees non-negative predictions.
- Included as a principled comparison even if it does not outperform HGBR.

---

### 4) `city_events_ml.evaluation`

#### `compute_metrics(y_true, y_pred)`
Returns a metrics dict including:
- RMSE
- MAE
- MaxError
- R²

#### `MetricsStore`
A simple logger that stores model metrics across:
- category
- model
- split (train/test)

Use it to build one combined results table across many model runs.

#### `evaluate_model(model_name, pipeline, ..., store=store, category=cat)`
Fits the pipeline, predicts on train/test, computes metrics, and appends results into the store.

Why it exists:
- Ensures all models are evaluated identically.
- Prevents “manual list append” chaos in notebooks.

---

### 5) `city_events_ml.serialization`

#### `save_model_skops(model, path)` / `load_model_skops(path, trusted=True)`
Save and load fitted pipelines using `skops`.

Why it exists:
- Pipelines are the deliverable: preprocess + model together.
- Allows reuse without rerunning training.

---

## End-to-end example (single category + multiple categories)

Below is a complete script-style example you can run in a notebook.

```
import os
import pandas as pd

from city_events_ml.io import TargetSpec, build_target_table
from city_events_ml.features import add_time_features
from city_events_ml.pipelines import (
    FeatureSpec,
    make_dummy_pipeline,
    make_ridge_pipeline,
    make_hgbr_pipeline,
    make_poisson_pipeline,
)
from city_events_ml.evaluation import MetricsStore, evaluate_model
from city_events_ml.serialization import save_model_skops

# --- Configuration ---
NEIGHBORHOODS = ("Mission", "SoMa", "Tenderloin", "Nob Hill", "Ingleside")
START = "2016-01-01"
END = "2020-01-01"     # exclusive end bound for 2016–2019 inclusive
FREQ = "6H"
TEST_SIZE = 0.15
ARTIFACT_DIR = "artifacts"

feature_spec = FeatureSpec()
feature_cols = list(feature_spec.categorical) + list(feature_spec.numeric)

def make_dataset(sf_dataframe: pd.DataFrame, *, category: str):
    spec = TargetSpec(
        category=category,
        neighborhoods=NEIGHBORHOODS,
        start=START,
        end=END,
        freq=FREQ,
    )
    dense = build_target_table(sf_dataframe, spec=spec)
    feat = add_time_features(dense, time_col="interval_start").sort_values("interval_start").reset_index(drop=True)

    split_idx = int(len(feat) * (1 - TEST_SIZE))
    train_df = feat.iloc[:split_idx].copy()
    test_df  = feat.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["count"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["count"]

    return train_df, test_df, X_train, y_train, X_test, y_test

# --- Single category run ---
store = MetricsStore()
category = "Potentially Life-Threatening"

train_df, test_df, X_train, y_train, X_test, y_test = make_dataset(sf_dataframe, category=category)

models = {
    "DummyRegressor(mean)": make_dummy_pipeline("mean"),
    "Ridge": make_ridge_pipeline(feature_spec, alpha=1.0),
    "HistGradientBoostingRegressor": make_hgbr_pipeline(feature_spec),
    "PoissonRegressor": make_poisson_pipeline(feature_spec, alpha=1.0),
}

for model_name, pipe in models.items():
    evaluate_model(
        model_name,
        pipe,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        store=store,
        category=category,
    )

results_single = store.to_frame().reset_index()
report_single = results_single.pivot_table(
    index=["category", "model"],
    columns="split",
    values=["RMSE", "MAE", "MaxError", "R2"],
    aggfunc="first",
)
print(report_single)

# Save the best model for this category (by lowest test RMSE)
flat = report_single.copy()
flat.columns = [f"{metric}_{split}" for metric, split in flat.columns]
flat = flat.reset_index()
best = flat.sort_values("RMSE_test").iloc[0]

best_model_name = best["model"]
best_pipe = models[best_model_name].fit(X_train, y_train)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
out_path = f"{ARTIFACT_DIR}/{category.lower().replace(' ', '_')}__{best_model_name.lower().replace(' ', '_')}.skops"
save_model_skops(best_pipe, out_path)
print("Saved:", out_path)

# --- Multiple categories run ---
CATEGORIES = [
    "Encampments",
    "Graffiti",
    "Non Life-threatening",
    "Potentially Life-Threatening",
    "Street and Sidewalk Cleaning",
]

store_all = MetricsStore()

for cat in CATEGORIES:
    train_df, test_df, X_train, y_train, X_test, y_test = make_dataset(sf_dataframe, category=cat)

    # (recreate fresh pipes each time)
    models = {
        "DummyRegressor(mean)": make_dummy_pipeline("mean"),
        "Ridge": make_ridge_pipeline(feature_spec, alpha=1.0),
        "HistGradientBoostingRegressor": make_hgbr_pipeline(feature_spec),
        "PoissonRegressor": make_poisson_pipeline(feature_spec, alpha=1.0),
    }

    for model_name, pipe in models.items():
        evaluate_model(
            model_name,
            pipe,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            store=store_all,
            category=cat,
        )

results_all = store_all.to_frame().reset_index()
report_all = results_all.pivot_table(
    index=["category", "model"],
    columns="split",
    values=["RMSE", "MAE", "MaxError", "R2"],
    aggfunc="first",
)
print(report_all)
```


---
## Repository structure
```
├── LICENSE
├── README.md
├── city_events_ml
│   ├── __init__.py
│   ├── base.py
│   ├── evaluation.py
│   ├── features.py
│   ├── io.py
│   ├── models.py
│   ├── pipelines.py
│   ├── serialization.py
│   └── workflows.py
├── city_events_ml.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
├── dist
│   ├── city_events_ml-0.1.0-py2.py3-none-any.whl
│   └── city_events_ml-0.1.0.tar.gz
├── notebooks
│   └── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── scripts
│   └── README.md
├── setup.cfg
├── setup.py
└── tests
    └── test_smoke.py

```
## Installation

1. (optional) create a virtual environement
   ```bash
   conda env create -n example-env python=3.8
   conda activate example-env
   ```

2. Install the package
   ```bash
   # 
   pip install city_events_ml-0.1.0-py2.py3-none-any.whl
   ```


### Running tests

Tests can be run with [pytest](https://docs.pytest.org/en/latest/)
```
pytest
```

### Code style

Use [black](https://pypi.org/project/black/) Python extension and
[nb-black](https://github.com/dnanhkhoa/nb_black) to auto-format jupyter
notebooks.
 



