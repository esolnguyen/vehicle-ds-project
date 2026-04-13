"""End-to-end ML pipeline: cleaned MVR data → EV adoption forecast.

Steps:
    1. Load a cleaned DataFrame (from parquet/csv produced by the data
       pipeline) OR run the data pipeline inline.
    2. Aggregate into a per-year EV share time series.
    3. Build lag/rolling feature matrix.
    4. Walk-forward backtest Naive baseline and RidgeLagForecaster.
    5. Fit the best model on all available history.
    6. Produce an N-year forward forecast.
    7. Write all artefacts (series, features, metrics, forecast) to
       ``reports/ml/``.

Usage:
    python -m src.ml_pipeline --cleaned data/cleaned.parquet
    python -m src.ml_pipeline --cleaned data/cleaned.parquet --horizon 5
    python -m src.ml_pipeline --cleaned data/cleaned.parquet --forecast-years 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ml.evaluator import BacktestResult, backtest
from src.ml.features import EVFeatureBuilder, build_ev_yearly_series
from src.ml.forecaster import NaiveForecaster, RidgeLagForecaster

_DEFAULT_OUTPUT = _PROJECT_ROOT / "reports" / "ml"


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def _load_cleaned(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {path}. Run the data pipeline first "
            f"(e.g. `python -m src.pipeline --save-cleaned {path}`)."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _choose_best(results: list[BacktestResult]) -> BacktestResult:
    return min(results, key=lambda r: r.mae)


def run(
    cleaned_path: Path,
    output_dir: Path = _DEFAULT_OUTPUT,
    min_year: int = 1990,
    max_year: int | None = None,
    horizon: int = 1,
    forecast_years: int = 5,
    ridge_alpha: float = 1.0,
) -> None:
    log = logging.getLogger("ml_pipeline")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load cleaned MVR ───────────────────────────────────────────────
    log.info("Loading cleaned data from %s", cleaned_path)
    cleaned_df = _load_cleaned(cleaned_path)
    log.info("Loaded %d rows, %d columns", len(cleaned_df), cleaned_df.shape[1])

    # ── 2. Build yearly EV share time series ──────────────────────────────
    series_df = build_ev_yearly_series(
        cleaned_df, min_year=min_year, max_year=max_year
    )
    series_path = output_dir / "ev_yearly_series.csv"
    series_df.to_csv(series_path)
    log.info("EV yearly series → %s", series_path)

    # ── 3. Feature engineering ────────────────────────────────────────────
    builder = EVFeatureBuilder()
    feats = builder.transform(series_df)
    feats_path = output_dir / "features.csv"
    feats.to_csv(feats_path)
    log.info("Feature matrix → %s  (%d rows)", feats_path, len(feats))

    # ── 4. Walk-forward backtest ──────────────────────────────────────────
    log.info("Backtesting models (horizon=%d)…", horizon)
    model_factories = [
        ("naive", lambda: NaiveForecaster()),
        (
            "ridge",
            lambda: RidgeLagForecaster(builder=builder, alpha=ridge_alpha),
        ),
    ]
    results: list[BacktestResult] = []
    for _, factory in model_factories:
        try:
            results.append(
                backtest(
                    series_df=series_df,
                    model_factory=factory,
                    builder=builder,
                    horizon=horizon,
                )
            )
        except ValueError as exc:
            log.warning("Skipping model %s: %s", factory().name, exc)

    if not results:
        raise RuntimeError("No backtest results — cannot continue")

    summary_df = pd.DataFrame([r.as_summary_row() for r in results])
    summary_path = output_dir / "backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Backtest summary → %s\n%s", summary_path, summary_df.to_string(index=False))

    for r in results:
        preds_path = output_dir / f"backtest_preds_{r.model_name}.csv"
        r.predictions.to_csv(preds_path, index=False)

    # ── 5. Fit best model on all data ─────────────────────────────────────
    best = _choose_best(results)
    log.info("Best model by MAE: %s (MAE=%.4f)", best.model_name, best.mae)

    if best.model_name == "ridge_lag":
        final_model = RidgeLagForecaster(builder=builder, alpha=ridge_alpha)
    else:
        final_model = NaiveForecaster()

    final_model.fit(feats)

    # ── 6. Forward forecast ───────────────────────────────────────────────
    history = series_df["ev_share"]
    recent_totals_mean = float(series_df["total"].tail(5).mean())
    totals_forward = [recent_totals_mean] * forecast_years

    forecast = final_model.forecast(
        history=history, n_steps=forecast_years, totals=totals_forward
    )
    forecast_df = forecast.to_frame()
    forecast_df["model"] = final_model.name
    forecast_path = output_dir / "forecast.csv"
    forecast_df.to_csv(forecast_path)
    log.info("Forecast → %s\n%s", forecast_path, forecast_df.to_string())

    # ── 7. Overview ───────────────────────────────────────────────────────
    overview = [
        "EV Adoption Forecast — Pipeline Summary",
        "=" * 50,
        f"  Cleaned source      : {cleaned_path}",
        f"  Series years        : {int(series_df.index.min())}–{int(series_df.index.max())}",
        f"  Feature rows        : {len(feats)}",
        f"  Backtest horizon    : {horizon}",
        f"  Best model          : {best.model_name}",
        f"  Best MAE            : {best.mae:.5f}",
        f"  Best RMSE           : {best.rmse:.5f}",
        f"  Best MAPE           : {best.mape:.2f}%",
        f"  Forecast years      : {forecast_years}",
        "",
        "Forecast:",
        forecast_df.to_string(),
        "",
    ]
    overview_path = output_dir / "overview.txt"
    overview_path.write_text("\n".join(overview))
    log.info("Overview → %s", overview_path)
    log.info("ML pipeline complete.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EV adoption forecasting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cleaned",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a cleaned MVR DataFrame (.parquet or .csv). "
        "Produce it via `python -m src.pipeline --save-cleaned PATH`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="DIR",
        help="Where to write ML artefacts.",
    )
    parser.add_argument("--min-year", type=int, default=1990)
    parser.add_argument("--max-year", type=int, default=None)
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon (in years) used during walk-forward backtest.",
    )
    parser.add_argument(
        "--forecast-years",
        type=int,
        default=5,
        help="Number of future years to forecast after fitting the best model.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _setup_logging(args.log_level)
    run(
        cleaned_path=args.cleaned,
        output_dir=args.output_dir,
        min_year=args.min_year,
        max_year=args.max_year,
        horizon=args.horizon,
        forecast_years=args.forecast_years,
        ridge_alpha=args.ridge_alpha,
    )
