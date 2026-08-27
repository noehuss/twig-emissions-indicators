from pathlib import Path

from twig_emissions_indicators.data import DataLoader
from twig_emissions_indicators.mrio import CalculatorMRIO
import pandas as pd


ENTSOE_TOKEN = "97eefaa6-898e-4fa1-8c02-86b5c32c121c"


if __name__ == "__main__":
    print("Loading data...")
    period_start = pd.Timestamp('20240101', tz='Europe/Brussels')
    period_end = pd.Timestamp('20251231', tz='Europe/Brussels')

    data_loader = DataLoader(start=period_start, end=period_end, api_key=ENTSOE_TOKEN, dir=Path("results"))
    calc = CalculatorMRIO(data_loader, bidding_zone='DK_2')
    calc.store_results(results_folder="folder_path_for_results_2")
