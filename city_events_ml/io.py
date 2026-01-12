from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    category: str
    neighborhoods: tuple[str, ...]
    freq: str = "6H"
    start: str | None = None
    end: str | None = None


def build_target_table(
    city_df: pd.DataFrame,
    *,
    spec: TargetSpec,
    datetime_col: str = "dateTime",
    category_col: str = "category",
    neighborhood_col: str = "neighborhood",
) -> pd.DataFrame:
    """
    Returns a fully dense table with columns:
      neighborhood, interval_start, count
    with missing intervals filled with 0 for each neighborhood.
    """
    df = city_df.copy()

    # Filter by category and neighborhoods
    df = df[df[category_col] == spec.category]
    df = df[df[neighborhood_col].isin(spec.neighborhoods)]

    # Optional time window
    if spec.start is not None:
        df = df[df[datetime_col] >= pd.Timestamp(spec.start)]
    if spec.end is not None:
        df = df[df[datetime_col] < pd.Timestamp(spec.end)]

    # 6-hour bins
    interval_start = df[datetime_col].dt.floor(spec.freq)

    event_counts = (
        df.assign(interval_start=interval_start)
          .groupby([neighborhood_col, "interval_start"])
          .size()
          .rename("count")
          .reset_index()
          .rename(columns={neighborhood_col: "neighborhood"})
    )

    # Build complete grid (dense time index per neighborhood)
    min_t = event_counts["interval_start"].min()
    max_t = event_counts["interval_start"].max()
    full_range = pd.date_range(min_t, max_t, freq=spec.freq)

    neighborhoods = sorted(set(event_counts["neighborhood"]))
    full_index = pd.MultiIndex.from_product(
        [neighborhoods, full_range],
        names=["neighborhood", "interval_start"],
    )

    dense = (
        event_counts.set_index(["neighborhood", "interval_start"])
                   .reindex(full_index, fill_value=0)
                   .reset_index()
    )

    dense["count"] = dense["count"].astype("int64")
    return dense
