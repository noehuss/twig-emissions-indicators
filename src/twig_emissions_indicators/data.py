import csv
from pathlib import Path
from typing import cast

import pandas as pd
from entsoe.entsoe import EntsoePandasClient

from twig_emissions_indicators.config import EMISSION_FACTORS

class DataLoader:
    """
    Data Loader for ENTSOE data, with caching and errors handling.

    """
    def __init__(self, api_key: str, start: pd.Timestamp, end: pd.Timestamp, dir: Path, emissions_factors: dict[str, float] | None = None):
        """
        Initialize the DataLoader with the given API key and period.
        """
        self.client = EntsoePandasClient(api_key=api_key)
        self.start = start
        self.end = end
        self._generation_cache = {}  # memory cache
        self.error_bz = ['GB', 'RU']  # keep track of failed zones
        self.emissions_factors = emissions_factors or EMISSION_FACTORS
        self.co2_intensities: pd.Series = pd.Series(self.emissions_factors).rename(
            "AEF (kgCO2/MWh)"
        )
        self.dir = dir
        self._ensure_folder(dir)

    @staticmethod
    def _ensure_folder(path: Path):
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_csv_auto(file_path: Path):
        """Load an ENTSO-E csv with structure detection"""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header1 = next(reader, [])
            header2 = next(reader, [])

        # No multi index
        if not any(h2.strip() for h2 in header2):
            df = pd.read_csv(file_path, index_col=0)
            df.columns = pd.MultiIndex.from_product([["Generation"], df.columns])
        # Multi index 'Actual Aggregated'
        elif any("Actual" in h2 or "Aggregated" in h2 for h2 in header2):
            df = pd.read_csv(
                file_path, index_col=0, header=[0, 1], skipinitialspace=True
            )
            df.columns = pd.MultiIndex.from_tuples(
                [(str(a).strip(), str(b).strip()) for a, b in df.columns]
            )
        else:
            print(f"Non recognized structure {file_path.name}, simple read.")
            df = pd.read_csv(file_path, index_col=0)

        df.index = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert(
            "Europe/Brussels"
        )
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        df = cast(pd.DataFrame, df)

        df = DataLoader._standardize_columns(df)  # Standardize column names

        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names based on MultiIndex structure.
        Handles three formats:
        1. Generation/Import with countries: (Generation/Import, Technology) -> Technology or Import_Country
        2. Technology with Actual Aggregated: (Technology, Actual Aggregated) -> Technology
        3. Single level: Already handled, just wrapped in MultiIndex
        Drops Actual Consumption columns.
        """
        if isinstance(df, pd.Series):
            return df

        if not isinstance(df.columns, pd.MultiIndex):
            return df

        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(1)

        new_cols = []
        cols_to_keep = []

        for i, (l0, l1) in enumerate(zip(level0, level1)):
            l0 = str(l0).strip()
            l1 = str(l1).strip()

            # Skip Actual Consumption columns
            if l1 == "Actual Consumption":
                continue

            # Format 1: Generation/Import with countries (e.g., "Generation", "Fossil Gas")
            if l0 in ["Generation", "Import"]:
                if l0 == "Import":
                    new_cols.append(f"Import_{l1}")
                else:
                    new_cols.append(l1)
            # Format 2: Technology with Actual Aggregated (e.g., "Biomass", "Actual Aggregated")
            elif l1 in ["Actual Aggregated"]:
                new_cols.append(l0)
            # Fallback: concatenate both levels
            else:
                new_cols.append(f"{l0}_{l1}")

            cols_to_keep.append(i)

        # Filter dataframe to keep only non-Actual Consumption columns
        df = df.iloc[:, cols_to_keep]
        df.columns = new_cols

        return df

    def _check_period_coverage(self, df: pd.DataFrame) -> bool:
        """Check if dataframe covers the requested period
        Return True if coverage is sufficient, False otherwise.
        """
        lack_coverage_len = (
            pd.date_range(start=self.start, end=self.end, freq="h")
            .difference(df.index.tolist())
            .shape[0]
        )  # type:ignore
        return lack_coverage_len <= 24

    def _filter_by_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to requested period"""
        df_temp = df[(df.index >= self.start) & (df.index <= self.end)]
        if isinstance(df_temp, pd.DataFrame):
            return df_temp
        return pd.DataFrame()

    def _concatenate_and_save(
        self, df_new: pd.DataFrame, file_path: Path
    ) -> pd.DataFrame:
        """Concatenate new data with existing, remove duplicates, and save"""
        if not isinstance(df_new.index, pd.DatetimeIndex):
            df_new.index = pd.to_datetime(
                df_new.index, utc=True, errors="coerce"
            ).tz_convert("Europe/Brussels")

        df_new = self._standardize_columns(df_new)

        if file_path.exists():
            df_existing = self._load_csv_auto(file_path)
            df_concat = pd.concat([df_existing, df_new])
            df_new = cast(
                pd.DataFrame,
                df_concat[~df_concat.index.duplicated(keep="last")]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0),
            )
            if not isinstance(df_new.index, pd.DatetimeIndex):
                df_new.index = pd.to_datetime(
                    df_new.index, utc=True, errors="coerce"
                ).tz_convert("Europe/Brussels")
            df_new = df_new.sort_index()

        df_new.to_csv(file_path)
        return df_new

    def _load_or_download(
        self,
        file_path: Path,
        download_func,
        bz: str,
        needs_processing: bool = False,
        process_func=None,
    ) -> pd.DataFrame :
        """
        Generic load-or-download pattern with period checking

        Args:
            file_path: Path to cache file
            download_func: Function to call for downloading
            bz: Bidding zone code
            needs_processing: If True, apply process_func after loading
            process_func: Function to process data (e.g., resample)
        """
        if bz in self.error_bz:
            print(f"Skipping {bz} (previous error, not retrying).")
            raise ValueError(f"{bz} previously failed — skipped.")

        folder_path = file_path.parent
        self._ensure_folder(folder_path)

        missing_period = None

        # Check if file exists and covers period
        if file_path.exists():
            print(f"Using cached file: {file_path.name}")
            df = self._load_csv_auto(file_path)

            if self._check_period_coverage(df):
                print(
                    f"File contains complete period ({df.index.min()} to {df.index.max()})"
                )
                df = self._filter_by_period(df)
                if needs_processing and process_func:
                    df = process_func(df)
                return df
            else:
                # Get the missing period
                missing_period = (
                    pd.date_range(start=self.start, end=self.end, freq="h", inclusive="left")
                    .difference(df.index.tolist())
                )  # type:ignore
                print(
                    f"File incomplete (has {df.index.min()} to {df.index.max()}, need {missing_period.min()} to {missing_period.max()}). Downloading new data..."
                )

        # Download new data
        if missing_period is not None:
            print(f"Downloading data for {bz} ({missing_period.min()} to {missing_period.max()})...")
            df_new = download_func(missing_period.min(), missing_period.max())
        else:
            print(f"Downloading data for {bz} ({self.start} to {self.end})...")
            df_new = download_func(self.start, self.end)

        # Concatenate and save
        df_new = self._concatenate_and_save(df_new, file_path)

        # Filter and process
        df = self._filter_by_period(df_new)
        if needs_processing and process_func:
            df = process_func(df)

        return df

    # ---------- GENERATION ----------
    # @lru_cache(maxsize=None)
    def _get_generation_file(self, bz: str, imports: bool) -> pd.DataFrame:
        """Load CSV (cached) or download once per zone. Skip if zone in error_bz."""
        if bz in self.error_bz:
            print(f"Skipping {bz} (previous error, not retrying).")
            raise ValueError(f"{bz} previously failed — skipped.")

        folder_path = Path(self.dir / bz)
        suffix = "with_imports" if imports else "no_imports"
        file_path = folder_path / f"generation_{bz}_{suffix}.csv"

        # --- IN-MEMORY CACHE ---
        if (bz, imports) in self._generation_cache:
            return self._generation_cache[(bz, imports)]

        # --- LOAD OR DOWNLOAD ---
        stop = False
        while not stop:
            try:
                def download_func(start: pd.Timestamp, end: pd.Timestamp):
                    return (
                        self.client.query_generation_import(
                            country_code=bz, start=start, end=end

                        )
                        if imports
                        else self.client.query_generation(
                            country_code=bz, start=start, end=end
                        )
                    )

                df = self._load_or_download(file_path, download_func, bz)
                self._generation_cache[(bz, imports)] = df
                return df

            except Exception as e:
                print(f"Failed to load {bz}: {e}")
                cond = str(input("Retry? (y/n): "))
                stop = cond.lower() != 'y'
                if stop:
                    self.error_bz.append(bz)
                raise

    def get_generation_data(self, bz: str) -> pd.DataFrame:
        return self._get_generation_file(bz, imports=False)

    def get_generation_imports_data(self, bz: str) -> pd.DataFrame:
        return self._get_generation_file(bz, imports=True)

    def get_forecast_data(self, bz: str) -> pd.DataFrame:
        """Get forecast data for a specific bidding zone"""
        file_path = Path(self.dir / bz / f"forecast_{bz}.csv")

        def download_func():
            df_generation = self.client.query_generation_forecast(
                country_code=bz, start=self.start, end=self.end
            )
            df_load = self.client.query_load_forecast(
                country_code=bz, start=self.start, end=self.end
            )
            return pd.concat([df_generation, df_load], axis=1)

        return self._load_or_download(file_path, download_func, bz)

    def get_DA_prices(self, bz: str) -> pd.Series:
        """Day ahead prices"""

        file_path = Path(self.dir / bz / f"DA_prices_{bz}.csv")

        def download_func():
            return self.client.query_day_ahead_prices(
                bz, start=self.start, end=self.end
            )

        def process_func(df):
            df = df.resample("h").mean()
            return df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df

        result = self._load_or_download(
            file_path,
            download_func,
            bz,
            needs_processing=True,
            process_func=process_func,
        )
        return result if isinstance(result, pd.Series) else result.iloc[:, 0]

    def get_load(self, bz: str) -> pd.Series | None:
        """Load"""

        file_path = Path(self.dir / bz / f"load_{bz}.csv")

        def download_func(start: pd.Timestamp, end: pd.Timestamp):
            return self.client.query_load(bz, start=start, end=end)

        def process_func(df):
            df = df.resample("h").mean()
            return df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df

        result = self._load_or_download(
            file_path,
            download_func,
            bz,
            needs_processing=True,
            process_func=process_func,
        )
        return result if isinstance(result, pd.Series) else result.iloc[:, 0]

    def get_load_w_storage(self, bz: str) -> pd.Series:
        """Load, including storage consumption"""
        load = self.get_load(bz)
        prod = self.get_generation_imports_data(bz).resample("h").mean()
        # Sum negative value prod df
        prod = prod[prod < 0].sum(axis=1)
        return load + prod

    # ---------- CO₂ INTENSITY ----------
    def get_domestic_co2_intensity(self, bz: str) -> float | pd.Series:
        """Local CO2 intensity for a zone"""
        try:
            gen = self.get_generation_data(bz)
        except Exception:
            print(f"Skipping CO₂ calc for {bz} (no data).")
            return cast(float, self.co2_intensities.at[f"Import {bz}"])

        if gen is None:
            return cast(float, self.co2_intensities.at[f"Import {bz}"])

        gen = gen.resample("h").mean()

        if isinstance(gen.columns, pd.MultiIndex):
            if "Actual Aggregated" in gen.columns.get_level_values(1):
                gen = gen.droplevel(level=1, axis=1)
            if "Generation" in gen.columns.get_level_values(0):
                gen = gen.droplevel(level=0, axis=1)

        gen = gen[gen > 0]
        gen.fillna(0, inplace=True)
        ef = self.co2_intensities.reindex(gen.columns).fillna(0)
        return pd.Series((gen.to_numpy() @ ef.to_numpy()) / gen.sum(axis=1))

    def get_emissions_factor(self) -> tuple[pd.Series, pd.Series]:
        """Get CO2 factors for generation and imports"""
        import_factors = self.co2_intensities.filter(regex="^Import").fillna(0)
        # gen factor: filter out import factors
        mask = ~self.co2_intensities.index.str.startswith("Import")
        filtered = self.co2_intensities.loc[mask]
        gen_factors = filtered.fillna(0)
        return gen_factors, import_factors
