"""Forecasting models for EV adoption share.

Two models are provided:

    NaiveForecaster      : last-observed value carried forward. Baseline.
    RidgeLagForecaster   : Ridge regression on lag + rolling features with
                           recursive multi-step forecasting.

Both expose the same minimal interface:

    fit(feats_df)          -> self
    predict(X)             -> np.ndarray            (one-step, vectorised)
    forecast(history, n)   -> pd.Series             (n-step, recursive)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.features import EVFeatureBuilder

logger = logging.getLogger(__name__)


class Forecaster(Protocol):
    name: str

    def fit(self, feats_df: pd.DataFrame) -> "Forecaster": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def forecast(
        self,
        history: pd.Series,
        n_steps: int,
        totals: list[float] | None = None,
    ) -> pd.Series: ...


@dataclass
class NaiveForecaster:
    """Persistence baseline: ``y_hat(t+k) = y(t)`` for all k."""

    name: str = "naive_last_value"
    _last_value: float = field(default=np.nan, init=False)

    def fit(self, feats_df: pd.DataFrame) -> "NaiveForecaster":
        if feats_df.empty:
            raise ValueError("Cannot fit NaiveForecaster on empty frame")
        self._last_value = float(feats_df["y"].iloc[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Prefer lag_1 if present, otherwise fall back to the fitted value.
        if "lag_1" in X.columns:
            return X["lag_1"].to_numpy(dtype=float)
        return np.full(len(X), self._last_value, dtype=float)

    def forecast(
        self,
        history: pd.Series,
        n_steps: int,
        totals: list[float] | None = None,
    ) -> pd.Series:
        if history.empty:
            raise ValueError("Cannot forecast with empty history")
        last = float(history.iloc[-1])
        start_year = int(history.index[-1]) + 1
        idx = pd.RangeIndex(start_year, start_year + n_steps, name="vehicle_year")
        return pd.Series(np.full(n_steps, last), index=idx, name="ev_share_forecast")


@dataclass
class RidgeLagForecaster:
    """Ridge regression on lag + rolling features, clipped to [0, 1]."""

    builder: EVFeatureBuilder = field(default_factory=EVFeatureBuilder)
    alpha: float = 1.0
    name: str = "ridge_lag"
    _model: Pipeline | None = field(default=None, init=False)
    _feature_cols: list[str] = field(default_factory=list, init=False)

    def fit(self, feats_df: pd.DataFrame) -> "RidgeLagForecaster":
        if feats_df.empty:
            raise ValueError("Cannot fit RidgeLagForecaster on empty frame")
        X, y = self.builder.split_xy(feats_df)
        self._feature_cols = list(X.columns)
        self._model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self._model.fit(X.to_numpy(), y.to_numpy())
        logger.info(
            "Fitted %s on %d samples, %d features (alpha=%.3f)",
            self.name,
            len(X),
            X.shape[1],
            self.alpha,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        aligned = X.reindex(columns=self._feature_cols)
        y_hat = self._model.predict(aligned.to_numpy())
        return np.clip(y_hat, 0.0, 1.0)

    def forecast(
        self,
        history: pd.Series,
        n_steps: int,
        totals: list[float] | None = None,
    ) -> pd.Series:
        """Recursive multi-step forecast.

        Args:
            history: observed ev_share indexed by vehicle_year.
            n_steps: number of future years to forecast.
            totals:  optional list of assumed total-registrations per future
                     year (same length as ``n_steps``). If ``None``, the mean
                     of the last 3 historical totals is carried forward — the
                     caller should pass realistic values if available.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        if history.empty:
            raise ValueError("Cannot forecast with empty history")

        if totals is not None and len(totals) != n_steps:
            raise ValueError(
                f"totals length ({len(totals)}) must match n_steps ({n_steps})"
            )

        working = history.astype(float).copy()
        last_year = int(working.index[-1])
        preds: list[float] = []

        for step in range(n_steps):
            total_for_year = totals[step] if totals is not None else None
            row = self.builder.build_single_row(
                history=working, total_for_year=total_for_year
            )
            row = row.reindex(columns=self._feature_cols)
            y_hat = float(self._model.predict(row.to_numpy())[0])
            y_hat = float(np.clip(y_hat, 0.0, 1.0))
            preds.append(y_hat)

            next_year = last_year + step + 1
            working.loc[next_year] = y_hat

        idx = pd.RangeIndex(
            last_year + 1, last_year + 1 + n_steps, name="vehicle_year"
        )
        return pd.Series(preds, index=idx, name="ev_share_forecast")
