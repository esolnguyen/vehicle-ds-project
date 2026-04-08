from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FieldSchema:
    name: str
    dtype: str
    description: str
    units: str
    valid_codes: set[str] = field(default_factory=set)
    valid_values: set[str] = field(default_factory=set)

    @property
    def all_valid(self) -> set[str]:
        """All accepted string tokens for this categorical column."""
        return self.valid_codes | self.valid_values


def load_schema(dict_csv_path: Path) -> dict[str, FieldSchema]:
    """Parse MVROpenData-Dictionary.csv into a column-name → FieldSchema map.

    Both the 'Codes' and 'Values' pipe-separated lists are captured so that
    columns which store short codes (e.g. CLASS = 'NC') and columns that store
    labels (e.g. MOTIVE_POWER = 'DIESEL') are both validated correctly.
    """
    meta = pd.read_csv(dict_csv_path, dtype=str, encoding="latin-1").fillna("")
    schema: dict[str, FieldSchema] = {}

    for _, row in meta.iterrows():
        name = row["Name"].strip()
        codes_str = row["Codes"].strip()
        values_str = row["Values"].strip()

        valid_codes = {c.strip() for c in codes_str.split("|") if c.strip()}
        valid_values = {v.strip() for v in values_str.split("|") if v.strip()}

        schema[name] = FieldSchema(
            name=name,
            dtype=row["Type"].strip(),
            description=row["Description"].strip(),
            units=row["Units"].strip(),
            valid_codes=valid_codes,
            valid_values=valid_values,
        )

    return schema


class DataCleaner:
    """Load raw per-year CSVs, clean them against the metadata schema.

    Typical use:
        schema  = load_schema(dict_csv_path)
        cleaner = DataCleaner(storage_dir)
        raw_df  = cleaner.load_years()
        clean_df, quality_df = cleaner.clean(raw_df, schema)
        clean_df, n_dropped  = DataCleaner.drop_duplicates(clean_df)
        DataCleaner.save(clean_df, output_path)
    """

    _CATEGORICAL = "Text (Categorical)"
    _INTEGER = "Integer"
    _DECIMAL = "Decimal"

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)

    def load_years(
        self,
        years: Optional[list[int]] = None,
        include_pre1990: bool = True,
    ) -> pd.DataFrame:
        """Load and concatenate per-year CSV files.

        Args:
            years: Specific manufacture years to load. ``None`` loads all.
            include_pre1990: Whether to include VehicleYear-Pre1990.csv.

        Returns:
            A single concatenated DataFrame with all string columns (raw).
        """
        paths = self._resolve_paths(years, include_pre1990)
        if not paths:
            raise FileNotFoundError(
                f"No CSV files found in {self.storage_dir} for the requested years."
            )

        parts: list[pd.DataFrame] = []
        for p in paths:
            logger.info("Loading %s", p.name)
            df = pd.read_csv(p, low_memory=False, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            parts.append(df)

        combined = pd.concat(parts, ignore_index=True)
        logger.info(
            "Loaded %d total records from %d files", len(combined), len(paths)
        )
        return combined

    def _resolve_paths(
        self,
        years: Optional[list[int]],
        include_pre1990: bool,
    ) -> list[Path]:
        paths: list[Path] = []

        if include_pre1990:
            pre = self.storage_dir / "VehicleYear-Pre1990.csv"
            if pre.exists():
                paths.append(pre)

        year_files = sorted(self.storage_dir.glob("VehicleYear-[0-9]*.csv"))
        if years is not None:
            year_set = set(years)
            year_files = [
                p
                for p in year_files
                if int(p.stem.replace("VehicleYear-", "")) in year_set
            ]

        paths.extend(year_files)
        return paths

    def clean(
        self,
        df: pd.DataFrame,
        schema: dict[str, FieldSchema],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply type coercions and categorical value validation.

        Steps:
        1. Strip whitespace from every cell in string columns; replace
           empty strings with ``pd.NA``.
        2. Cast Integer columns to nullable ``pd.Int64Dtype()``.
        3. Cast Decimal columns to ``float64``.
        4. Cast Categorical columns to ``pandas.CategoricalDtype`` and
           count values that fall outside the declared valid set.
        5. Leave Text (Free) columns as ``object`` (stripped).

        Returns:
            cleaned_df : the cleaned DataFrame.
            quality_df : per-column quality report (nulls, invalid counts …).
        """
        df = df.copy()

        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(
            lambda s: s.str.strip()).replace("", pd.NA)

        quality_rows: list[dict] = []

        for col, fs in schema.items():
            if col not in df.columns:
                continue

            null_before = int(df[col].isna().sum())
            invalid_count = 0

            if fs.dtype == self._INTEGER:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                    pd.Int64Dtype()
                )

            elif fs.dtype == self._DECIMAL:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif fs.dtype == self._CATEGORICAL:
                all_valid = fs.all_valid
                if all_valid:
                    present_mask = df[col].notna()
                    invalid_mask = present_mask & ~df[col].isin(all_valid)
                    invalid_count = int(invalid_mask.sum())
                    if invalid_count:
                        logger.debug(
                            "Column %-35s  %d invalid categorical values",
                            col,
                            invalid_count,
                        )
                df[col] = df[col].astype("category")

            null_after = int(df[col].isna().sum())
            quality_rows.append(
                {
                    "column": col,
                    "dtype_schema": fs.dtype,
                    "dtype_pandas": str(df[col].dtype),
                    "total_rows": len(df),
                    "null_count": null_after,
                    "null_rate_pct": round(null_after / len(df) * 100, 2),
                    "coercion_nulls_added": max(0, null_after - null_before),
                    "invalid_categorical_count": invalid_count,
                }
            )

        quality_df = pd.DataFrame(quality_rows)
        logger.info("Cleaning complete. %d columns processed.",
                    len(quality_rows))
        return df, quality_df

    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Remove exact duplicate rows (all columns identical).

        Note: VIN11 contains only the first 11 characters of the VIN and is
        therefore not a unique vehicle identifier.  Full-row deduplication is
        the safest strategy for this dataset.

        Returns:
            (deduplicated_df, n_dropped)
        """
        n_before = len(df)
        df = df.drop_duplicates(keep="first")
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info("Removed %d exact duplicate rows", n_dropped)
        return df, n_dropped

    @staticmethod
    def save(df: pd.DataFrame, output_path: Path) -> None:
        """Save a cleaned DataFrame to Parquet (if suffix is .parquet) or CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        logger.info("Saved %d records → %s", len(df), output_path)
