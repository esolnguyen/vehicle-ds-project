"""Feature engineering for EV adoption forecasting.

Turns the cleaned NZ MVR DataFrame (one row per vehicle) into a per-year
time series of EV share, then augments it with lag and rolling features
suitable for supervised ML forecasting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_analyzer import _EV_MOTIVE_LABELS

logger = logging.getLogger(__name__)

_YEAR_COL = "VEHICLE_YEAR"
_POWER_COL = "MOTIVE_POWER"


def build_ev_yearly_series(
    df: pd.DataFrame,
    min_year: int = 1990,
    max_year: int | None = None,
) -> pd.DataFrame:
    """Aggregate a cleaned MVR DataFrame into a per-year EV time series.

    Returns a DataFrame indexed by ``vehicle_year`` with columns:
        total        : total vehicles registered for that manufacture year
        ev_count     : number classified as EV / hybrid
        ev_share     : ev_count / total  (float in [0, 1])
    """
    if _YEAR_COL not in df.columns or _POWER_COL not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{_YEAR_COL}' and '{_POWER_COL}' columns."
        )

    year = pd.to_numeric(df[_YEAR_COL], errors="coerce").astype("Int64")
    power = df[_POWER_COL].astype("string").str.upper()

    mask = year.notna()
    year = year[mask]
    power = power[mask]

    ev_flag = power.isin(_EV_MOTIVE_LABELS).astype(int)

    grouped = pd.DataFrame({"vehicle_year": year.astype(int), "is_ev": ev_flag})
    agg = (
        grouped.groupby("vehicle_year")
        .agg(total=("is_ev", "size"), ev_count=("is_ev", "sum"))
        .sort_index()
    )

    agg = agg[agg.index >= min_year]
    if max_year is not None:
        agg = agg[agg.index <= max_year]

    # Fill any missing years with zeros so lag features stay contiguous.
    full_index = pd.RangeIndex(
        start=int(agg.index.min()),
        stop=int(agg.index.max()) + 1,
        name="vehicle_year",
    )
    agg = agg.reindex(full_index, fill_value=0)

    agg["ev_share"] = np.where(
        agg["total"] > 0, agg["ev_count"] / agg["total"], 0.0
    )
    logger.info(
        "Built EV yearly series: %d years (%d-%d), %d EVs of %d total",
        len(agg),
        agg.index.min(),
        agg.index.max(),
        int(agg["ev_count"].sum()),
        int(agg["total"].sum()),
    )
    return agg


@dataclass
class EVFeatureBuilder:
    """Build supervised (X, y) features from a per-year EV share series.

    The target is ``ev_share`` in the current year. Features are:
      - ``lag_k``       : ev_share k years ago (k in lags)
      - ``roll_mean_w`` : rolling mean of ev_share over past w years
      - ``roll_std_w``  : rolling std of ev_share over past w years
      - ``log_total``   : log(total registrations that year) — captures
                          whether a year has a meaningful denominator
    """

    lags: tuple[int, ...] = (1, 2, 3, 5)
    rolling_windows: tuple[int, ...] = (3, 5)
    target_col: str = "ev_share"

    def transform(self, series_df: pd.DataFrame) -> pd.DataFrame:
        """Return a feature DataFrame aligned on the source year index.

        Rows with any NaN (from incomplete lag history) are dropped.
        """
        if self.target_col not in series_df.columns:
            raise ValueError(
                f"Input series must have '{self.target_col}' column"
            )

        feats = pd.DataFrame(index=series_df.index)
        feats["y"] = series_df[self.target_col].astype(float)

        for k in self.lags:
            feats[f"lag_{k}"] = feats["y"].shift(k)

        for w in self.rolling_windows:
            shifted = feats["y"].shift(1)
            feats[f"roll_mean_{w}"] = shifted.rolling(window=w).mean()
            feats[f"roll_std_{w}"] = shifted.rolling(window=w).std()

        if "total" in series_df.columns:
            feats["log_total"] = np.log1p(
                series_df["total"].astype(float)
            )

        feats = feats.dropna()
        logger.info(
            "Feature matrix: %d rows x %d features (after dropna)",
            len(feats),
            feats.shape[1] - 1,
        )
        return feats

    def feature_columns(self, feats: pd.DataFrame) -> list[str]:
        return [c for c in feats.columns if c != "y"]

    def split_xy(self, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        cols = self.feature_columns(feats)
        return feats[cols], feats["y"]

    def build_single_row(
        self,
        history: pd.Series,
        total_for_year: float | None = None,
    ) -> pd.DataFrame:
        """Build a one-row feature frame for the *next* year after ``history``.

        ``history`` is the full ev_share series up to (and including) the
        most recent observed year. This is used for recursive multi-step
        forecasting: the predicted value is appended to ``history`` and the
        method is called again.
        """
        row: dict[str, float] = {}
        for k in self.lags:
            if len(history) < k:
                raise ValueError(
                    f"history too short for lag {k}: need {k}, have {len(history)}"
                )
            row[f"lag_{k}"] = float(history.iloc[-k])

        for w in self.rolling_windows:
            window_vals = history.iloc[-w:]
            row[f"roll_mean_{w}"] = float(window_vals.mean())
            row[f"roll_std_{w}"] = float(window_vals.std(ddof=1)) if w > 1 else 0.0

        if total_for_year is not None:
            row["log_total"] = float(np.log1p(total_for_year))

        return pd.DataFrame([row])


def iter_feature_columns(
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    include_log_total: bool = True,
) -> list[str]:
    """Deterministic column order used when building single-row forecasts."""
    cols: list[str] = []
    for k in lags:
        cols.append(f"lag_{k}")
    for w in rolling_windows:
        cols.append(f"roll_mean_{w}")
        cols.append(f"roll_std_{w}")
    if include_log_total:
        cols.append("log_total")
    return cols
