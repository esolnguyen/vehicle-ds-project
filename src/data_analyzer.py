from __future__ import annotations
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CURRENT_YEAR: int = date.today().year

_EV_MOTIVE_LABELS: frozenset[str] = frozenset(
    {
        "ELECTRIC",
        "PETROL HYBRID",
        "DIESEL HYBRID",
        "PETROL ELECTRIC HYBRID",
        "DIESEL ELECTRIC HYBRID",
        "PLUGIN PETROL HYBRID",
        "PLUGIN DIESEL HYBRID",
        "ELECTRIC [PETROL EXTENDED]",
        "ELECTRIC [DIESEL EXTENDED]",
        "ELECTRIC FUEL CELL HYDROGEN",
        "ELECTRIC FUEL CELL OTHER",
    }
)


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fleet_composition(self) -> pd.DataFrame:
        return self._count("VEHICLE_TYPE", "vehicle_type")

    def body_type_distribution(self) -> pd.DataFrame:
        return self._count("BODY_TYPE", "body_type")

    def make_distribution(self, top_n: int = 30) -> pd.DataFrame:
        if "MAKE" not in self.df.columns:
            return pd.DataFrame(columns=["make", "count"])
        return (
            self.df["MAKE"]
            .str.upper()
            .value_counts(dropna=True)
            .head(top_n)
            .rename_axis("make")
            .reset_index(name="count")
        )

    def registrations_by_year(self) -> pd.DataFrame:
        return self._count("VEHICLE_YEAR", "vehicle_year")

    def motive_power_distribution(self) -> pd.DataFrame:
        return self._count("MOTIVE_POWER", "motive_power")

    def fuel_type_by_year(self) -> pd.DataFrame:
        if not {"VEHICLE_YEAR", "MOTIVE_POWER"}.issubset(self.df.columns):
            return pd.DataFrame()

        return (
            self.df.groupby(
                [
                    self.df["VEHICLE_YEAR"].astype("Int64"),
                    self.df["MOTIVE_POWER"].astype(str),
                ],
                observed=True,
                dropna=False,
            )
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

    def ev_adoption_by_year(self) -> pd.DataFrame:
        if not {"VEHICLE_YEAR", "MOTIVE_POWER"}.issubset(self.df.columns):
            return pd.DataFrame(
                columns=["vehicle_year", "total", "ev_count", "ev_share_pct"]
            )

        year_col = self.df["VEHICLE_YEAR"].astype("Int64")
        total = self.df.groupby(year_col, observed=True).size().rename("total")
        ev_count = (
            self.df[self.df["MOTIVE_POWER"].isin(_EV_MOTIVE_LABELS)]
            .groupby(year_col, observed=True)
            .size()
            .rename("ev_count")
        )
        result = pd.concat([total, ev_count], axis=1).fillna(0).astype(int)
        result["ev_share_pct"] = (
            result["ev_count"] / result["total"] * 100).round(2)
        return result.reset_index().rename(columns={"VEHICLE_YEAR": "vehicle_year"})

    def import_status_distribution(self) -> pd.DataFrame:
        return self._count("IMPORT_STATUS", "import_status")

    def top_origin_countries(self, top_n: int = 20) -> pd.DataFrame:
        return self._top_n("ORIGINAL_COUNTRY", "country", top_n)

    def top_previous_countries(self, top_n: int = 20) -> pd.DataFrame:
        return self._top_n("PREVIOUS_COUNTRY", "country", top_n)

    def geographic_distribution(self) -> pd.DataFrame:
        return self._count("TLA", "tla")

    def engine_stats(self) -> pd.DataFrame:
        cols = ["VEHICLE_TYPE", "CC_RATING", "POWER_RATING"]
        present = [c for c in cols if c in self.df.columns]
        if len(present) < 2:
            return pd.DataFrame()

        temp = self.df[present].copy()
        for num_col in ("CC_RATING", "POWER_RATING"):
            if num_col in temp.columns:
                temp[num_col] = pd.to_numeric(temp[num_col], errors="coerce")

        numeric_cols = [c for c in (
            "CC_RATING", "POWER_RATING") if c in present]
        return (
            temp.groupby("VEHICLE_TYPE", observed=True)[numeric_cols]
            .describe()
            .round(1)
        )

    def fuel_consumption_stats(self) -> pd.DataFrame:
        required = {"FC_COMBINED"}
        group_cols = [c for c in (
            "MOTIVE_POWER", "VEHICLE_TYPE") if c in self.df.columns]
        if not required.issubset(self.df.columns) or not group_cols:
            return pd.DataFrame()

        temp = self.df[group_cols + ["FC_COMBINED"]].copy()
        temp["FC_COMBINED"] = pd.to_numeric(
            temp["FC_COMBINED"], errors="coerce")

        return (
            temp.groupby(group_cols, observed=True)["FC_COMBINED"]
            .agg(mean_l_per_100km="mean", sample_count="count")
            .round(2)
            .reset_index()
        )

    def fleet_age_distribution(self) -> pd.DataFrame:
        if "VEHICLE_YEAR" not in self.df.columns:
            return pd.DataFrame(columns=["age_bracket", "count"])

        ages = (
            _CURRENT_YEAR
            - pd.to_numeric(self.df["VEHICLE_YEAR"], errors="coerce")
        ).dropna().astype(int).clip(lower=0)

        bins = [0, 5, 10, 15, 20, 25, float("inf")]
        labels = ["0-5 yrs", "6-10 yrs", "11-15 yrs",
                  "16-20 yrs", "21-25 yrs", "26+ yrs"]
        bucketed = pd.cut(ages, bins=bins, labels=labels,
                          right=True, include_lowest=True)

        return (
            bucketed.value_counts()
            .sort_index()
            .rename_axis("age_bracket")
            .reset_index(name="count")
        )

    def industry_class_distribution(self) -> pd.DataFrame:
        return self._count("INDUSTRY_CLASS", "industry_class")

    def vehicle_usage_distribution(self) -> pd.DataFrame:
        return self._count("VEHICLE_USAGE", "vehicle_usage")

    def generate_report(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analyses: dict[str, object] = {
            "fleet_composition": self.fleet_composition,
            "body_type_distribution": self.body_type_distribution,
            "make_distribution": self.make_distribution,
            "registrations_by_year": self.registrations_by_year,
            "motive_power_distribution": self.motive_power_distribution,
            "fuel_type_by_year": self.fuel_type_by_year,
            "ev_adoption_by_year": self.ev_adoption_by_year,
            "import_status_distribution": self.import_status_distribution,
            "top_origin_countries": self.top_origin_countries,
            "top_previous_countries": self.top_previous_countries,
            "geographic_distribution": self.geographic_distribution,
            "engine_stats": self.engine_stats,
            "fuel_consumption_stats": self.fuel_consumption_stats,
            "fleet_age_distribution": self.fleet_age_distribution,
            "industry_class_distribution": self.industry_class_distribution,
            "vehicle_usage_distribution": self.vehicle_usage_distribution,
        }

        for name, fn in analyses.items():
            try:
                result = fn()
                out_path = output_dir / f"{name}.csv"
                result.to_csv(out_path)
                logger.info("%-40s → %s", name, out_path.name)
            except Exception as exc:
                logger.warning("Analysis '%s' failed: %s", name, exc)

        self._write_overview(output_dir)

    def _write_overview(self, output_dir: Path) -> None:
        n_total = len(self.df)

        def _safe(col: str) -> pd.Series:
            return self.df.get(col, pd.Series(dtype=str))

        year_min = _safe("VEHICLE_YEAR").min()
        year_max = _safe("VEHICLE_YEAR").max()
        unique_makes = _safe("MAKE").nunique()

        ev_mask = _safe("MOTIVE_POWER").isin(_EV_MOTIVE_LABELS)
        bev_count = (_safe("MOTIVE_POWER") == "ELECTRIC").sum()

        lines = [
            "NZ Motor Vehicle Register — Dataset Overview",
            "=" * 50,
            f"  Total vehicles          : {n_total:>12,}",
            f"  Vehicle year range      : {year_min} - {year_max}",
            f"  Unique makes            : {unique_makes:>12,}",
            f"  Fully electric (BEV)    : {int(bev_count):>12,}",
            f"  EV / Hybrid total       : {int(ev_mask.sum()):>12,}",
            f"  EV / Hybrid share       : {ev_mask.mean() * 100:>11.2f}%",
            "",
        ]
        overview_path = output_dir / "overview.txt"
        overview_path.write_text("\n".join(lines))
        logger.info("Overview → %s", overview_path.name)

    def _count(self, col: str, label: str) -> pd.DataFrame:
        if col not in self.df.columns:
            return pd.DataFrame(columns=[label, "count"])
        return (
            self.df[col]
            .value_counts(dropna=False)
            .rename_axis(label)
            .reset_index(name="count")
        )

    def _top_n(self, col: str, label: str, top_n: int) -> pd.DataFrame:
        if col not in self.df.columns:
            return pd.DataFrame(columns=[label, "count"])
        return (
            self.df[col]
            .value_counts(dropna=True)
            .head(top_n)
            .rename_axis(label)
            .reset_index(name="count")
        )
