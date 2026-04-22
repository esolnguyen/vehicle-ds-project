# NZ Vehicle Data Science Project

A data science pipeline for the **New Zealand NZTA Motor Vehicle Register (MVR)** open dataset. It downloads the full NZ fleet, cleans it against the official metadata schema, runs exploratory analyses, and forecasts electric-vehicle (EV) adoption.

## What it does

1. **Crawl** downloads the NZTA "all vehicle years" fleet data and splits it into per-year CSVs.
2. **Clean** parses the `MVROpenData-Dictionary.csv` schema, coerces types, validates categorical codes, and deduplicates.
3. **Analyse** fleet composition, body type, make share, geography, motive power, EV share, fleet age, fuel consumption.
4. **Forecast** per-year EV adoption forecast using lag + rolling features and a Ridge regressor (with a naive baseline), evaluated via walk-forward backtesting.

## Project layout

```
.
├── src/
│   ├── new_zealand_crawler.py    Download & split NZTA fleet ZIP → per-year CSVs
│   ├── data_cleaner.py           Schema-driven type coercion + validation
│   ├── data_analyzer.py          EDA reports (16 analyses)
│   ├── pipeline.py               Data pipeline CLI  (clean → analyse → report)
│   ├── ml_pipeline.py            ML pipeline CLI    (features → backtest → forecast)
│   └── ml/
│       ├── features.py           EV yearly series + lag/rolling feature builder
│       ├── forecaster.py         NaiveForecaster, RidgeLagForecaster
│       └── evaluator.py          Walk-forward backtest, MAE / RMSE / MAPE
├── notebooks/
│   └── nz_vehicle_analysis.ipynb
├── data/                         (created at runtime gitignored)
├── reports/                      (created at runtime gitignored)
├── run_pipeline.sh
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download the NZTA dataset

```python
from src.new_zealand_crawler import NewZealandDataCrawler
NewZealandDataCrawler().download_all_years()
```

This writes per-year CSVs to `data/nz/storage/VehicleYear-<YEAR>.csv` and `VehicleYear-Pre1990.csv`.

You also need the metadata dictionary at `data/MVROpenData-Dictionary.csv` (obtainable from NZTA).

### 2. Run the data pipeline

```bash
# All years, write a cleaned parquet for downstream ML
python -m src.pipeline --save-cleaned data/cleaned.parquet

# Only specific years
python -m src.pipeline --years 2018 2019 2020 2021 2022 2023 2024

# Custom report directory
python -m src.pipeline --output-dir reports/run_2026
```

Outputs go to `reports/`:
- `quality_report.csv` per-column null rates, invalid categorical counts
- `overview.txt` dataset-level summary
- 16× `*_distribution.csv` / analysis CSVs

### 3. Run the EV adoption forecasting pipeline

```bash
python -m src.ml_pipeline --cleaned data/cleaned.parquet --forecast-years 5
```

CLI options:

| Flag | Default | Description |
|---|---|---|
| `--cleaned` | *required* | Path to cleaned DataFrame (`.parquet` or `.csv`) |
| `--output-dir` | `reports/ml` | Where to write ML artefacts |
| `--min-year` | `1990` | Earliest vehicle year to include |
| `--max-year` | *none* | Latest vehicle year to include |
| `--horizon` | `1` | Forecast horizon (years) used in walk-forward backtest |
| `--forecast-years` | `5` | Number of future years to forecast after fitting best model |
| `--ridge-alpha` | `1.0` | L2 regularisation strength for the Ridge model |

Outputs in `reports/ml/`:
- `ev_yearly_series.csv` per-year totals, EV counts, EV share
- `features.csv` lag + rolling feature matrix
- `backtest_summary.csv` MAE / RMSE / MAPE per model
- `backtest_preds_<model>.csv` per-fold predictions
- `forecast.csv` N-year forward forecast from best model
- `overview.txt` run summary

## ML approach

**Target:** `ev_share = ev_count / total` per vehicle manufacture year, where EV includes BEV + hybrids + PHEV + fuel cell (see `_EV_MOTIVE_LABELS` in `src/data_analyzer.py`).

**Features:** lagged EV share (1, 2, 3, 5 years), rolling mean & std over 3 and 5 years, `log(total)` as a denominator-size signal.

**Models:**
- `NaiveForecaster` persistence baseline (`y_hat(t+k) = y(t)`)
- `RidgeLagForecaster` Ridge regression with `StandardScaler`, recursive multi-step forecast, output clipped to `[0, 1]`

**Evaluation:** expanding-window walk-forward backtest with configurable horizon; models are ranked by MAE and the winner is refit on the full series for the forward forecast.

## Requirements

Python ≥ 3.10. Key dependencies: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `requests`, `matplotlib`, `seaborn`, `jupyter`. Full list in `requirements.txt`.

## Data source

[NZTA Motor Vehicle Register open data](https://nzta.govt.nz/resources/open-data/) NZ fleet composition, released quarterly.
