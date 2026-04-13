"""Walk-forward backtesting and metrics for EV adoption forecasters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.ml.features import EVFeatureBuilder, build_ev_yearly_series
from src.ml.forecaster import Forecaster

logger = logging.getLogger(__name__)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean absolute percentage error, guarded against zero targets."""
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


@dataclass
class BacktestResult:
    model_name: str
    horizon: int
    mae: float
    rmse: float
    mape: float
    predictions: pd.DataFrame  # columns: vehicle_year, y_true, y_pred, fold

    def as_summary_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model_name,
            "horizon": self.horizon,
            "mae": round(self.mae, 5),
            "rmse": round(self.rmse, 5),
            "mape_pct": round(self.mape, 3),
            "n_preds": len(self.predictions),
        }


def backtest(
    series_df: pd.DataFrame,
    model_factory: Callable[[], Forecaster],
    builder: EVFeatureBuilder,
    horizon: int = 1,
    min_train_years: int = 10,
) -> BacktestResult:
    """Walk-forward backtest.

    At each step, train on years ``[t0 .. t)``, forecast the next ``horizon``
    years and record (y_true, y_pred) pairs. Expands the training window by
    one year and repeats.

    Args:
        series_df: per-year series with columns ``total``, ``ev_count``,
                   ``ev_share`` indexed by ``vehicle_year``.
        model_factory: zero-arg callable returning a fresh forecaster.
        builder: feature builder (shared so lags/windows match).
        horizon: number of years ahead to evaluate per fold.
        min_train_years: minimum rows in the feature matrix before the first
                         fold is scored.
    """
    if series_df.empty:
        raise ValueError("Empty series — nothing to backtest")

    feats = builder.transform(series_df)
    years = feats.index.to_list()

    if len(years) < min_train_years + horizon:
        raise ValueError(
            f"Not enough history for backtest: need "
            f"{min_train_years + horizon} feature-years, have {len(years)}"
        )

    records: list[dict[str, float | int]] = []
    model_name = model_factory().name

    for split_idx in range(min_train_years, len(years) - horizon + 1):
        train_years = years[:split_idx]
        train_feats = feats.loc[train_years]
        model = model_factory().fit(train_feats)

        last_train_year = train_years[-1]
        history = series_df.loc[: last_train_year, "ev_share"]
        totals = series_df.loc[
            last_train_year + 1 : last_train_year + horizon, "total"
        ].tolist()
        totals = [float(t) for t in totals] if len(totals) == horizon else None

        forecast = model.forecast(history, n_steps=horizon, totals=totals)

        for step, year in enumerate(forecast.index, start=1):
            if year not in series_df.index:
                continue
            records.append(
                {
                    "vehicle_year": int(year),
                    "step_ahead": step,
                    "y_true": float(series_df.loc[year, "ev_share"]),
                    "y_pred": float(forecast.loc[year]),
                    "fold": split_idx - min_train_years,
                }
            )

    if not records:
        raise RuntimeError("Backtest produced no predictions")

    preds_df = pd.DataFrame.from_records(records)
    y_true = preds_df["y_true"].to_numpy()
    y_pred = preds_df["y_pred"].to_numpy()

    result = BacktestResult(
        model_name=model_name,
        horizon=horizon,
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mape=mape(y_true, y_pred),
        predictions=preds_df,
    )
    logger.info(
        "Backtest %-18s horizon=%d  MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  n=%d",
        model_name,
        horizon,
        result.mae,
        result.rmse,
        result.mape,
        len(preds_df),
    )
    return result


def series_from_cleaned(
    cleaned_df: pd.DataFrame,
    min_year: int = 1990,
    max_year: int | None = None,
) -> pd.DataFrame:
    """Convenience re-export so callers can import from one place."""
    return build_ev_yearly_series(cleaned_df, min_year=min_year, max_year=max_year)
