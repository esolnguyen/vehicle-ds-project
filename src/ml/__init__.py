"""EV adoption forecasting pipeline."""

from src.ml.evaluator import BacktestResult, backtest
from src.ml.features import EVFeatureBuilder, build_ev_yearly_series
from src.ml.forecaster import NaiveForecaster, RidgeLagForecaster

__all__ = [
    "EVFeatureBuilder",
    "build_ev_yearly_series",
    "NaiveForecaster",
    "RidgeLagForecaster",
    "BacktestResult",
    "backtest",
]
