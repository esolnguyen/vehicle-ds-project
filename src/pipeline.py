"""End-to-end data pipeline: clean → analyse → report.

Usage (from project root):

    # Process all available year files
    python src/pipeline.py
    # OR: python -m src.pipeline

    # Process specific years only
    python src/pipeline.py --years 2020 2021 2022 2023 2024

    # Skip pre-1990 vehicles, save cleaned data as Parquet
    python src/pipeline.py --no-pre1990 --save-cleaned data/cleaned.parquet

    # Custom output directory
    python src/pipeline.py --output-dir reports/run_2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' imports work
# whether run as `python src/pipeline.py` or `python -m src.pipeline`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_analyzer import DataAnalyzer
from src.data_cleaner import DataCleaner, load_schema
_DICT_CSV = _PROJECT_ROOT / "data" / "MVROpenData-Dictionary.csv"
_STORAGE_DIR = _PROJECT_ROOT / "data" / "nz" / "storage"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "reports"


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def run(
    years: list[int] | None = None,
    include_pre1990: bool = True,
    output_dir: Path = _DEFAULT_OUTPUT,
    save_cleaned: Path | None = None,
) -> None:
    """Run the full clean → analyse pipeline.

    Steps:
        1. Parse the metadata schema from MVROpenData-Dictionary.csv.
        2. Load raw per-year CSVs from the storage directory.
        3. Clean: type-cast, validate categoricals, strip whitespace.
        4. Deduplicate on VIN11 + VEHICLE_YEAR.
        5. Optionally persist the cleaned DataFrame.
        6. Run all analyses and write CSV + overview reports.

    Args:
        years: Vehicle manufacture years to include. ``None`` = all years.
        include_pre1990: Whether to include VehicleYear-Pre1990.csv.
        output_dir: Directory where report CSVs will be written.
        save_cleaned: Path to persist the cleaned DataFrame (.parquet or .csv).
                      ``None`` skips persistence.
    """
    log = logging.getLogger("pipeline")

    # ── 1. Schema ─────────────────────────────────────────────────────────────
    log.info("Loading metadata schema from %s", _DICT_CSV)
    schema = load_schema(_DICT_CSV)
    log.info("Schema loaded: %d fields", len(schema))

    # ── 2. Raw data ───────────────────────────────────────────────────────────
    cleaner = DataCleaner(_STORAGE_DIR)
    raw_df = cleaner.load_years(years=years, include_pre1990=include_pre1990)

    # ── 3. Clean ──────────────────────────────────────────────────────────────
    log.info("Cleaning data…")
    cleaned_df, quality_df = cleaner.clean(raw_df, schema)

    # ── 4. Deduplicate ────────────────────────────────────────────────────────
    n_before = len(cleaned_df)
    cleaned_df, n_dupes = DataCleaner.drop_duplicates(cleaned_df)
    log.info(
        "Clean dataset: %d rows, %d columns  (removed %d exact dupes, kept %.1f%%)",
        len(cleaned_df),
        len(cleaned_df.columns),
        n_dupes,
        len(cleaned_df) / n_before * 100,
    )

    # ── Quality report ────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = output_dir / "quality_report.csv"
    quality_df.to_csv(quality_path, index=False)
    log.info("Quality report → %s", quality_path)

    # ── 5. Persist cleaned data (optional) ───────────────────────────────────
    if save_cleaned is not None:
        DataCleaner.save(cleaned_df, save_cleaned)

    # ── 6. Analyse ────────────────────────────────────────────────────────────
    log.info("Running analysis…")
    analyzer = DataAnalyzer(cleaned_df)
    analyzer.generate_report(output_dir)

    log.info("Pipeline complete.  Reports written to: %s", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NZ MVR data cleaning and analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Vehicle manufacture years to process (e.g. 2020 2021 2022). "
        "Omit to process all available year files.",
    )
    parser.add_argument(
        "--no-pre1990",
        dest="include_pre1990",
        action="store_false",
        help="Exclude the VehicleYear-Pre1990.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="DIR",
        help="Directory to write report CSVs and overview.txt.",
    )
    parser.add_argument(
        "--save-cleaned",
        type=Path,
        default=None,
        metavar="PATH",
        help="Persist the cleaned DataFrame to this path (.parquet or .csv).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _setup_logging(args.log_level)
    run(
        years=args.years,
        include_pre1990=args.include_pre1990,
        output_dir=args.output_dir,
        save_cleaned=args.save_cleaned,
    )
