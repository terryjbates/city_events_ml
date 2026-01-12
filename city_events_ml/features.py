from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    *,
    time_col: str = "interval_start",
) -> pd.DataFrame:
    """
    Adds:
      year, month, day,
      dow (0-6), hour (0,6,12,18),
      and cyclical encodings for dow/hour: sin/cos
    """
    out = df.copy()
    dt = out[time_col]

    out["year"] = dt.dt.year.astype("int16")
    out["month"] = dt.dt.month.astype("int8")
    out["day"] = dt.dt.day.astype("int8")

    dow = dt.dt.dayofweek.astype("int8")
    hour = dt.dt.hour.astype("int8")

    out["dow"] = dow
    out["hour"] = hour

    # cyclical encodings
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # your hour values are 0,6,12,18; still treat as 24-hour cycle
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return out
