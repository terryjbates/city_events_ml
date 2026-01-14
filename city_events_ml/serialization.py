# city_events_ml/serialization.py
from __future__ import annotations

from pathlib import Path
from typing import Any


def save_model_skops(model: Any, path: str | Path) -> None:
    """
    Save a fitted scikit-learn estimator/pipeline using skops.
    """
    from skops.io import dump  # local import so package doesn't hard-require skops

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)


def load_model_skops(path: str | Path, *, trusted: bool = True) -> Any:
    """
    Load a model saved with skops.

    'trusted=True' assumes you're loading your own artifact. If loading unknown artifacts,
    keep trusted=False and review the allowed types.
    """
    from skops.io import load

    return load(path, trusted=trusted)


# Usage
# from city_events_ml.serialization import save_model_skops, load_model_skops

# best_pipe = hgbr_pipe.fit(X_train, y_train)
# save_model_skops(best_pipe, "artifacts/hgbr_potentially_life_threatening.skops")

# reloaded = load_model_skops("artifacts/hgbr_potentially_life_threatening.skops")
