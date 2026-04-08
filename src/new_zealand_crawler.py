import logging
from pathlib import Path
from zipfile import ZipFile

import requests


class NewZealandDataCrawler:
    """Download NZTA vehicle fleet open-data files by vehicle year."""

    def __init__(self):
        self.logger = logging.getLogger("NewZealandDataCrawler")

        # Solution 1: Hardcode the Azure Blob link directly, bypassing the NZTA website firewall entirely
        self.dataset_links = {
            "all_years": "https://wksprdgisopendata.blob.core.windows.net/motorvehicleregister/Fleet-data-all-vehicle-years.zip"
        }

        self.download_dir = Path("data/nz/download")
        self.storage_dir = Path("data/nz/storage")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        # Use a standard browser User-Agent to be safe
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            }
        )

    def _download_file(self, url: str, dest_path: Path) -> None:
        """Download a file from URL to dest_path."""
        self.logger.info("Downloading %s", url)

        with self.session.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    def download_all_years(self) -> None:
        """
        Download the all-years ZIP from NZTA, extract it, and split into yearly CSV files.
        Saves as:
          - VehicleYear-Pre1990.csv (for vehicles before 1990)
          - VehicleYear-1990.csv, VehicleYear-1991.csv, etc.
        """
        import pandas as pd

        all_years_url = self.dataset_links["all_years"]
        self.logger.info(
            "Downloading all-years dataset from %s", all_years_url)

        zip_path = self.download_dir / "Fleet-data-all-vehicle-years.zip"
        extracted_csv = None

        try:
            # Download the all-years ZIP
            self._download_file(all_years_url, zip_path)

            # Extract CSV from ZIP
            with ZipFile(zip_path, "r") as zip_ref:
                csv_members = [name for name in zip_ref.namelist()
                               if name.lower().endswith(".csv")]

                if not csv_members:
                    raise FileNotFoundError(
                        "No CSV found in the all-years ZIP file")

                if len(csv_members) > 1:
                    self.logger.warning(
                        "Multiple CSVs found, using first: %s", csv_members)

                # Extract to download directory temporarily
                extracted_csv = self.download_dir / csv_members[0]
                zip_ref.extract(csv_members[0], self.download_dir)

            # Load and split by year
            self.logger.info("Loading CSV and splitting by vehicle year...")
            # Added low_memory=False to prevent DtypeWarnings on large files
            nz_df = pd.read_csv(extracted_csv, low_memory=False)

            # Drop NA values before grabbing unique years
            years = sorted(nz_df["VEHICLE_YEAR"].dropna().unique())
            self.logger.info("Processing %d unique years: %d to %d",
                             len(years), int(min(years)), int(max(years)))

            # Save pre-1990 vehicles
            pre_1990_df = nz_df[nz_df["VEHICLE_YEAR"] < 1990]
            if not pre_1990_df.empty:
                pre_1990_path = self.storage_dir / "VehicleYear-Pre1990.csv"
                pre_1990_df.to_csv(pre_1990_path, index=False)
                self.logger.info("Saved %d pre-1990 records to %s",
                                 len(pre_1990_df), pre_1990_path)

            # Save each year separately for years >= 1990
            for year in [y for y in years if y >= 1990]:
                year_df = nz_df[nz_df["VEHICLE_YEAR"] == year]
                # Cast year to int to avoid VehicleYear-2020.0.csv filenames
                year_path = self.storage_dir / f"VehicleYear-{int(year)}.csv"
                year_df.to_csv(year_path, index=False)
                self.logger.debug(
                    "Saved %d records for year %d", len(year_df), int(year))

            self.logger.info("Successfully split data into %d year files",
                             len([y for y in years if y >= 1990]) + (1 if not pre_1990_df.empty else 0))

        finally:
            # Clean up temporary files to save disk space
            if zip_path.exists():
                zip_path.unlink()
            if extracted_csv and extracted_csv.exists():
                extracted_csv.unlink()
