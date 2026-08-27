import csv
from typing import cast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from entsoe.mappings import Area

from twig_emissions_indicators.data import DataLoader


class CalculatorMRIO:
    def __init__(self, dataLoader: DataLoader, bidding_zone: str, cutoff: float = 0.05):
        """Multi-regional-Input-Output (MRIO) model for the entered bidding zone.

        Parameters
        ----------
        dataLoader : DataLoader
            The data loader to use for fetching generation and forecast data.
        bidding_zone : str
            The bidding zone to compute indicators for.
        cutoff : float, optional
            The cutoff threshold for BFS exploration, by default 0.05.
        """
        self.dataLoader = dataLoader
        self.bz_code = bidding_zone

        if self.bz_code not in [bz.name for bz in Area]:
            raise ValueError(f"Invalid bidding zone name: {self.bz_code}. Valid names are: {[bz.name for bz in Area]}")

        self.cutoff = cutoff

        self.gen_emissions_factors, self.import_emissions_factors = (
            self.dataLoader.get_emissions_factor()
        )
        self._calculate_all()
        self._emissions_factors_reindex()
        self._compute_aef()
        self._compute_mef()


    def _explore_bz_bfs(self):
        """Explore bidding zone through BFS algorithm.
        Compute import share for each bidding zone, and continue exploring until cutoff is reached.
        """
        root = self.bz_code
        visited = [root]
        queue = [root]
        self.bz_import_share = {root: 1}
        var = [root]
        fix = []
        self.error_bz = ["GB", "UA", "BY", "RU_KGD", "RU", "ES"]

        while queue:
            bz = queue.pop(0)
            print(bz, end=" ")

            try:
                gen_imp = self.dataLoader.get_generation_imports_data(bz)
            except Exception as e:
                print(f"Error import {bz}: {e}")
                self.error_bz.append(bz)
                continue

            if gen_imp is None:
                continue

            imports = gen_imp.filter(regex="^Import", axis=1)
            # drop sum
            imports = imports.drop("Import_sum", axis=1, errors="ignore")
            # remove Import_ prefix
            imports.columns = imports.columns.str.replace("^Import_", "", regex=True)
            total = imports.sum(axis=0).sum(axis=0)

            for neigh_bz, val in imports.sum(axis=0).items():
                if neigh_bz not in visited and neigh_bz not in self.error_bz:
                    neigh_bz = str(neigh_bz)
                    print(f"BZ {neigh_bz}, share: {val / total:.2%}")
                    share = self.bz_import_share[bz] * val / total
                    self.bz_import_share[neigh_bz] = share

                    if share >= self.cutoff:
                        queue.append(neigh_bz)
                        var.append(neigh_bz)
                        visited.append(neigh_bz)
                        if neigh_bz in fix:
                            fix.remove(neigh_bz)
                    elif neigh_bz not in var and neigh_bz not in fix:
                        fix.append(neigh_bz)

        fix += self.error_bz
        self.bz_dict = {"fix": fix, "var": var}

    def _build_T_matrix_from_bz_dict(self):
        """Build the technology matrix T from bidding zone dictionary."""
        self._explore_bz_bfs()
        keyed = {}

        for bz in self.bz_dict["var"]:
            if bz in self.error_bz:
                continue
            try:
                gen_imp = self.dataLoader.get_generation_imports_data(bz)
                if gen_imp is None:
                    continue
                # gen_imp.columns = gen_imp.columns.map('_'.join).str.strip('|')
                gen_imp.drop("Import_sum", axis=1, inplace=True)
                gen_imp = gen_imp.resample("h").mean()
                # for column without Import_, add suffix _{bz}
                gen_imp.columns = gen_imp.columns.map(
                    lambda x: f"{x}_{bz}" if "Import_" not in x else x
                )
                self._clean_outliers(gen_imp)
                keyed[bz] = gen_imp
            except Exception as e:
                print(f"Error {bz}: {e}")
                continue

        self._construct_arrays_from_keyed(keyed)

    def _clean_outliers(self, df: pd.DataFrame):
        """Remove outliers (>3 std dev) and clip negative values"""
        for col in df.columns:
            col_std = df[col].std()
            col_mean = df[col].mean()
            outliers = (df[col] - col_mean).abs() > 3 * col_std
            df.loc[outliers, col] = np.nan
            df[col] = df[col].ffill()
            df[col] = df[col].clip(lower=0)

    def _construct_arrays_from_keyed(self, keyed: dict):
        """Construct T_array and related attributes from keyed dataframes"""
        big = pd.concat(keyed, names=["BZ", "Hour"])
        big.columns.name = "Source"
        stacked = big.stack().rename("Energy")  # type:ignore
        long = stacked.reset_index()
        hours_index = pd.Index(sorted(long["Hour"].unique()))

        pivot_df = long.pivot_table(
            index="Hour",
            columns=["Source", "BZ"],
            values="Energy",
            aggfunc="sum",
            fill_value=0,
        )
        pivot_df = pivot_df.reindex(hours_index, fill_value=0)
        # Get all unique sources and sort them (Generation first, then Import)
        all_sources = pivot_df.columns.get_level_values(0).unique().to_numpy()
        gen_sources = sorted([s for s in all_sources if "Import" not in s])
        import_sources = sorted([s for s in all_sources if "Import" in s])
        sources = np.array(
            gen_sources + import_sources
        )  # Generation first, Imports at end

        bzs = sorted(pivot_df.columns.get_level_values(1).unique().to_numpy())
        n_hours, n_sources, n_bzs = len(pivot_df), len(sources), len(bzs)

        full_cols = pd.MultiIndex.from_product([sources, bzs], names=["Source", "BZ"])  # type:ignore
        pivot_df = pivot_df.reindex(columns=full_cols, fill_value=0)

        flat = pivot_df.values
        self.T_array = flat.reshape(n_hours, n_sources, n_bzs)
        self.sources = np.array(sources)
        self.hours_index = pivot_df.index
        self.bzs = np.array(bzs)
        self.pivot_df = pivot_df
        self.P_egt_index = ["Import" not in s for s in self.sources]
        self.is_P_import_fix = np.isin(
            self.sources, np.array(["Import_" + z for z in self.bz_dict["fix"]])
        )
        self.is_var = np.isin(
            self.sources, np.array(["Import_" + z for z in self.bz_dict["var"]])
        )
        self.P_import_fix_bz = [
            s.replace("Import_", "", 1) if s.startswith("Import_") else s
            for s in self.sources[self.is_P_import_fix]
        ]
        self.is_fix = ~self.is_var

    def _calculate_all(self):
        self._build_T_matrix_from_bz_dict()
        self._calc_Narray()
        self._calc_Parray()
        self._calc_Marray()

    def _calc_Narray(self):
        row_sums = self.T_array.sum(axis=1)
        diag_values = np.where(row_sums != 0, 1 / row_sums, 0)
        eye = np.eye(diag_values.shape[1])
        self.N_array = diag_values[:, :, None] * eye[None, :, :]

    def _calc_Parray(self):
        self.P = self.T_array @ self.N_array
        self.P_var = self.P[:, self.is_var, :]
        self.P_fix = self.P[:, self.is_fix, :]

    def _calc_Marray(self):
        try:
            print(self.bz_dict)
            I_minus_P_var_inv = np.linalg.inv(
                np.eye(self.P_var.shape[1])[None, :, :] - self.P_var
            )
            self.M_array = self.P_fix @ I_minus_P_var_inv
            # Assert sum equals 1
            row_sums = self.M_array.sum(axis=1)
            assert np.allclose(row_sums, 1, atol=1), (
                f"Row sums are not equal to 1: {row_sums}"
            )
        except np.linalg.LinAlgError:
            print("Singular matrix — skipping inversion.")


    def _emissions_factors_reindex(self):
        generators = [col for col in self.sources if "Import" not in col]
        # iterate over sources. If not in emissions factors, delete suffix
        gen_emissions_factors = {}
        for gen in generators:
            if gen not in self.gen_emissions_factors.index:
                gen_emissions_factors[gen] = self.gen_emissions_factors.get(
                    gen.split("_")[0], 0
                )
            else:
                gen_emissions_factors[gen] = self.gen_emissions_factors[gen]

        self.gen_emissions_factors = pd.Series(gen_emissions_factors)

        co2_array_gen = np.tile(
            self.gen_emissions_factors.to_numpy(), (self.M_array.shape[0], 1)
        )
        co2_array_boundary = self._build_boundary_co2_array()

        self.emissions_factors = np.concatenate(
            [co2_array_gen, co2_array_boundary], axis=1
        )

        # dict with sources as keys and emissions factors as values
        self.emissions_factors_dict = {
            **gen_emissions_factors,
            **self.import_emissions_factors,
        }

    def _build_boundary_co2_array(self) -> np.ndarray:
        """Build CO2 intensity array for boundary zones"""
        boundary_bz = self.P_import_fix_bz
        co2_array = np.zeros((self.M_array.shape[0], len(boundary_bz)))

        for i, bz in enumerate(boundary_bz):
            try:
                val = self.dataLoader.get_domestic_co2_intensity(bz)
                if isinstance(val, (float, int)):
                    co2_array[:, i] = val
                else:
                    print(
                        f"Domestic co2 intensity {bz}: {val.mean():.2f} kgCO2/MWh; {val.isna().sum()} missing values"
                    )
                    co2_array[:, i] = np.nan_to_num(np.array(val))
            except Exception as e:
                print(f"{bz} skipped: {e}")
                co2_array[:, i] = 0

        return co2_array

    def _compute_aef(self):
        self.aef_bz = np.nan_to_num(
            self.emissions_factors[:, :, None] * self.M_array
        ).sum(axis=1)
        self.aef_df = pd.DataFrame(
            self.aef_bz, columns=self.bzs, index=self.hours_index
        )

    def _compute_mef(self):
        M_df = pd.DataFrame(
            self.M_array[:, :, np.where(self.bzs == self.bz_code)[0][0]],
            index=self.hours_index,
            columns=self.sources[self.is_fix],
        )
        load = self.dataLoader.get_load_w_storage(bz=self.bz_code)
        consumption_sources = pd.DataFrame(index=load.index)
        for src in self.sources[self.is_fix]:
            consumption_sources[src] = load * M_df[src]
        re_columns = [
            col
            for col in self.sources[self.is_fix]
            if any(res in col for res in ["Solar", "Wind Onshore", "Wind Offshore"])
        ]
        vre = consumption_sources[re_columns].sum(axis=1)
        net_load = load - vre
        delta_net_load = net_load.diff()
        mef_df = pd.DataFrame(index=net_load.index)
        df_filtering_results = pd.DataFrame(index=net_load.index)
        df_filtering_results["Net_Load"] = net_load
        dispatchable_sources = []

        mef_cols = {}
        filter_cols = {}

        for src in self.sources[self.is_fix]:
            if src in re_columns or "Import_" in src:
                continue

            dispatchable_sources.append(src)

            diff = consumption_sources[src].diff()
            same_direction = delta_net_load * diff >= 0
            opposite_direction = ~same_direction

            mef_cols[f"D_{src}"] = consumption_sources[src].where(same_direction, 0)

            delta_d = diff.where(same_direction, 0)
            mef_cols[f"Delta_D_{src}"] = delta_d

            filter_cols[src] = diff.where(opposite_direction, 0)
            filter_cols[f"{src}_abs"] = diff.abs().where(opposite_direction, 0)
            filter_cols[f"{src}_nb"] = opposite_direction.astype(int)

            key = src.replace("Generation_", "", 1)
            emission_factor = self.emissions_factors_dict.get(key)
            mef_cols[f"Delta_D_Emissions_{src}"] = delta_d * emission_factor  # type: ignore

        # Add all new columns in one operation
        mef_df = pd.concat([mef_df, pd.DataFrame(mef_cols, index=mef_df.index)], axis=1)
        df_filtering_results = pd.concat(
            [df_filtering_results, pd.DataFrame(filter_cols, index=df_filtering_results.index)],
            axis=1,
        )

        df_filtering_results[
            [col for col in df_filtering_results.columns if "nb" in col]
        ].groupby(pd.qcut(df_filtering_results["Net_Load"], q=10)).mean().to_csv(
            "filtering_results_by_net_load.csv"
        )
        mef_df["Delta_D_Load"] = mef_df[
            [f"Delta_D_{src}" for src in dispatchable_sources]
        ].sum(axis=1)
        mef_df["Delta_D_Emissions"] = mef_df[
            [f"Delta_D_Emissions_{src}" for src in dispatchable_sources]
        ].sum(axis=1)
        mef_df["MEF"] = mef_df["Delta_D_Emissions"] / mef_df["Delta_D_Load"]
        self.mef_df = mef_df
        self.consumption_sources = consumption_sources
        self.filtering_results = df_filtering_results

    def post_process_consumption_sources(self, Load):
        MEAN_THRESHOLD = 0.01
        STD_THRESHOLD = 0.005
        DATETIME_COL = "DateTime"

        X = self.consumption_sources.drop(columns=[DATETIME_COL])

        shares = X.div(Load, axis=0)

        stats = pd.DataFrame(
            {
                "mean": shares.mean(),
                "std": shares.std(),
            }
        )

        def extract_zone(col: str) -> str:
            """
            'Wind Onshore_DK_2' → 'DK_2'
            'Import_GB'         → 'Import_GB'
            """
            if col.startswith("Import_"):
                return col
            parts = col.rsplit("_", maxsplit=2)
            if len(parts) >= 2:
                candidate = "_".join(parts[-2:])
                return (
                    candidate if parts[-1].isdigit() or parts[-1] == "LU" else parts[-1]
                )
            return parts[-1]

        stats["zone"] = stats.index.map(extract_zone)

        def classify(row):
            if row["mean"] >= MEAN_THRESHOLD or row["std"] >= STD_THRESHOLD:
                return "keep"
            return "group"

        stats["decision"] = stats.apply(classify, axis=1)

        kept = stats[stats["decision"] == "keep"].index.tolist()
        grouped = stats[stats["decision"] == "group"]

        print(f"\n{'=' * 60}")
        print(f"  Generators disaggregated : {len(kept)}")
        print(f"  GGenerators grouped in Other   : {len(grouped)}")
        print(f"{'=' * 60}\n")

        X_filtered = X[kept].copy()

        for zone, grp in grouped.groupby("zone"):
            cols_to_group = grp.index.tolist()
            other_col = f"Other_{zone}"
            X_filtered[other_col] = X[cols_to_group].sum(axis=1)
            print(f"Other_{zone:20s} ← {cols_to_group}")

        # Check Σ g_k = 1
        gen_cols = [c for c in X_filtered.columns]
        residual = (X_filtered[gen_cols].sum(axis=1) - Load).abs()
        print(f"\nMax Residual |Σ X_k - L| : {residual.max():.4f} MWh  (should be ≈ 0)")

        # Reset index and set DateTime column
        self.consumption_sources = X_filtered
        self.consumption_sources[DATETIME_COL] = self.consumption_sources.index

    def store_results(self, results_folder: str):
        """Store the main analysis datasets for the selected bidding zone.

        Args:
            results_folder (str): Path to the folder where results will be stored.

        Outputs:
            Production per technology and bidding zone.
            AEFs (emissions factors) for each source.
            Filtered consumption and emissions data and MEF ratio.
        """

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        #load_series = self.dataLoader.get_load_w_storage(self.bz_code).rename("Load")
        #self.aef_df.index = load_series.index

        self.aef_df.to_csv(self.results_folder / "aefs.csv", index=True)
        self.mef_df[['MEF']].to_csv(self.results_folder / "mefs.csv", index=True)
        self.consumption_sources.to_csv(self.results_folder / "consumption_sources.csv", index=True)


    def store_results_with_features(self, results_folder: str):
        """Build and export the main analysis datasets for the selected bidding zone."""
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Prepare the base dataset with load and carbon intensity.
        load_series = self.dataLoader.get_load_w_storage(self.bz_code).rename("Load")
        aef_series = cast(pd.Series, self.aef_df[self.bz_code]).rename("AEF")
        dataset = pd.concat([load_series, aef_series], axis=1)
        dataset["DateTime"] = dataset.index
        features = dataset.copy()

        # Add core load and emission change indicators.
        dataset["Delta_Load"] = dataset["Load"].diff()
        dataset["Emissions"] = dataset["AEF"] * dataset["Load"]
        dataset["Delta_Emissions"] = dataset["Emissions"].diff()

        # Add market and load context for the study zone and neighbouring zones.
        for bz in self.bz_dict["var"]:
            try:
                prices = self.dataLoader.get_DA_prices(bz)
                dataset[f"Price_{bz}"] = prices
                features[f"Price_{bz}"] = prices
            except Exception as exc:
                print(f"{bz} skipped: {exc}")

            try:
                zone_load = self.dataLoader.get_load(bz)
                dataset[f"Load_{bz}"] = zone_load
                features[f"Load_{bz}"] = zone_load
            except Exception as exc:
                print(f"{bz} skipped: {exc}")

        for bz in self.bz_dict["var"]:
            if bz != self.bz_code:
                features[f"Price_diff_{bz}"] = (
                    features[f"Price_{bz}"] - features[f"Price_{self.bz_code}"]
                )

        # Add generation-based indicators for each relevant zone.
        for bz in self.bz_dict["var"]:
            try:
                generation_df = self.dataLoader.get_generation_imports_data(bz)
                dataset[f"Solar_{bz}"] = generation_df[
                    [col for col in generation_df.columns if "Solar" in col]
                ].sum(axis=1)
                features[f"Wind_{bz}"] = generation_df[
                    [col for col in generation_df.columns if "Wind" in col]
                ].sum(axis=1)
                features[f"Generation_{bz}"] = generation_df[
                    [col for col in generation_df.columns if "Import" not in col]
                ].sum(axis=1)
                features[f"Load_Generation_{bz}"] = (
                    features[f"Load_{bz}"] - features[f"Generation_{bz}"]
                )
            except Exception as exc:
                print(f"{bz} skipped: {exc}")

        # Add production mix details for the study zone.
        production_df = self.dataLoader.get_generation_imports_data(bz=self.bz_code)
        production_df[production_df < 0] = 0
        production_df = production_df.resample("h").mean()

        dataset = pd.concat([dataset, production_df], axis=1)
        import_df = production_df[
            [col for col in production_df.columns if "Import" in col]
        ]
        features = pd.concat(
            [features, import_df, import_df.diff().add_prefix("Delta_")],
            axis=1,
        )

        dataset["MEF"] = self.mef_df["MEF"]
        dataset["Delta_D_Load"] = self.mef_df["Delta_D_Load"]
        dataset["Delta_D_Emissions"] = self.mef_df["Delta_D_Emissions"]

        re_columns = [
            col
            for col in self.consumption_sources.columns
            if any(res in col for res in ["Solar", "Wind Onshore", "Wind Offshore"])
        ]
        low_carbon_columns = re_columns + [
            col
            for col in self.consumption_sources.columns
            if any(
                res in col
                for res in [
                    "Hydro",
                    "Nuclear",
                    "Geothermal",
                    "renewable",
                    "Biomass",
                    "Waste",
                ]
            )
        ]
        coal_columns = [
            col for col in self.consumption_sources.columns if "coal" in col.lower()
        ]

        dataset["Generation_VRE"] = self.consumption_sources[re_columns].sum(axis=1)
        dataset["Generation_low_carbon"] = self.consumption_sources[
            low_carbon_columns
        ].sum(axis=1)
        dataset["Generation_Coal"] = self.consumption_sources[coal_columns].sum(axis=1)
        dataset["Net_Load"] = dataset["Load"] - dataset["Generation_VRE"]
        dataset["Delta_Net_Load"] = dataset["Net_Load"].diff()
        dataset["Share_Coal"] = 100 * dataset["Generation_Coal"] / dataset["Net_Load"]

        # Final cleanup and export.
        dataset.dropna(inplace=True)
        features["Net_Load"] = dataset["Net_Load"]
        features["Delta_Net_Load"] = dataset["Delta_Net_Load"]
        print(f"Final dataset shape: {dataset.shape}")

        dataset.to_csv(path_or_buf=self.results_folder / "data.csv", index=False)
        self.consumption_sources["DateTime"] = self.consumption_sources.index
        self.post_process_consumption_sources(load_series)
        self.consumption_sources.to_csv(
            path_or_buf=self.results_folder / "data_mix.csv",
            index=False,
        )

        features["DateTime"] = features.index
        features["Net_Load"] = dataset["Net_Load"]
        features.to_csv(path_or_buf=self.results_folder / "features.csv", index=False)

        self.filtering_results.describe().to_csv(self.results_folder / "filtering_results.csv")
        self.filtering_results[
            [col for col in self.filtering_results.columns if "nb" in col]
        ].mean().sort_values(axis=0).to_csv(self.results_folder / "filtering_counts.csv")

        magnitude_summary = pd.DataFrame()
        for col in self.filtering_results.columns:
            if "abs" in col:
                magnitude_summary[col] = self.filtering_results[
                    self.filtering_results[col] > 0
                ][col].describe()
        magnitude_summary.to_csv(self.results_folder / "filtering_magnitudes.csv")

        self.aef_df["DateTime"] = self.aef_df.index
        self.aef_df.to_csv(path_or_buf=self.results_folder / "aefs.csv", index=False)
