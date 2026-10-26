# twig-emissions-indicators
![Python](https://img.shields.io/badge/python-3.14-blue.svg)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Python package for quantification of CO2 intensities of european electricity bidding zones. It relies on ENTSO-E data from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).

The accounting method is following a multi-regional input output approach developed in the following paper: [Evaluation Method for the Hourly Average CO2eq. Intensity of the Electricity Mix and Its Application to the Demand Response of Residential Heating](https://www.mdpi.com/1996-1073/12/7/1345) and further developed in the following documentation available in the github repository.

Through the MRIO method, average emissions factors are calculated. Then, a Linear Model Tree is trained to quantify the marginal emissions factors. For further information, please refer to the documentation available in the github repository.

The core functionalities of the `twig-emissions-indicators` package are:
- entsoe data loading
- data processing through MRIO approach
- Linear Model Tree Training for Marginal Emissions Factors quantification

---
## Funding & Development

The research leading to this work is being carried out as part of the HELIOS (Hybrid Energy Asset Live Operating System) project, funded through the ELFORSK grant under the Danish Energy Agency’s Energy Technology Development and Demonstration Program (EUDP), as part of the special fund for Smart Energy and Digitalization.

This GitHub repository was developed and is maintained by [Twig Energy](https://www.twig.energy/en) as part of the project.

---

## Installation

```bash
pip install twig-emissions-indicators
```

---

## Usage

### Data Loading and MRIO Calculation

In order to use the `twig-emissions-indicators` package and generate the needed data for the Linear Model Tree training, you need to first load the ENTSO-E data using the `DataLoader` class through the MRIO approach.

```python
from twig_emissions_indicators.data import DataLoader
from twig_emissions_indicators.mrio import CalculatorMRIO

ENTSOE_TOKEN = "Your ENTSO-E token"
period_start = pd.Timestamp('20250101', tz='Europe/Brussels')
period_end = pd.Timestamp('20251231', tz='Europe/Brussels')

data_loader = DataLoader(start=period_start, 
                         end=period_end, 
                         api_key=ENTSOE_TOKEN, 
                         dir=Path("folder_path_for_entsoe_data"))
calc = CalculatorMRIO(data_loader, bidding_zone='DK_2')
calc.store_results(results_folder="folder_path_for_results")
```

In the above example, the DataLoader is used to load the ENTSO-E data, and the CalculatorMRIO is used to process the data through the MRIO approach. The results are then stored in the specified folder.

This step is necessary to generate the data needed for the Linear Model Tree training.

### Linear Model Tree Training

Once the MRIO calculation has produced emissions data, prepare three time-aligned inputs:

- `dE`: a `pandas.Series` containing the change in emissions (the prediction target).
- `dL`: a `pandas.Series` containing the corresponding change in load.
- `X`: a `pandas.DataFrame` of explanatory features, such as generation mix or weather.

All three inputs should use the same `DatetimeIndex`. The model combines `dL` with `X`; by default, it treats `Hour`, `Month`, and `Dow` as circular time features when those columns are present. 

```python
from twig_emissions_indicators.linearmodeltree import LinearTreeModel

# Build these from your processed data. Keep the same DatetimeIndex on all inputs.
dE = data["delta_emissions"]
dL = data["delta_load"]
X = data[["Hour", "Month", "Dow", "Wind", "Solar", "Gas"]]

model = LinearTreeModel(random_state=42)
model.fit(dE=dE, dL=dL, X=X, test_year=2025)

# The fitted model predicts the target (delta emissions) using dL as the regressor.
predicted_dE = model.predict(X, dL=dL)
print(f"Test R²: {model.get_results().metrics['test_r2']:.3f}")

# Inspect the estimated marginal emissions factors for the held-out observations.
results = model.get_results()
print(results.mef.head())

model.save("linear_tree_model.joblib")
```

`fit` drops rows with missing target or feature values. By default it trains on years before `test_year` and evaluates on that year (2025 by default); pass `test_size` instead to use a chronological fraction of the observations as the test set, for example `model.fit(dE, dL, X, test_size=0.2)`. Hyperparameters are tuned using time-series cross-validation, so training requires enough observations for the configured number of splits (`n_splits`, default 5).

The results container returned by `get_results()` includes train/test metrics, feature importances, selected features, baseline metrics, and `g_k`, the estimated marginal emissions factor for each held-out row. A saved model can be restored with `LinearTreeModel.load("linear_tree_model.joblib")`.

---

## CO2 intensity factors

For each generator, the CO2 intensity factor is defined from IPCC data and are gathered in the `twig_emissions_indicators.config` module. Potential neighbors zones that do not have ENTSOE data are assigned a default value from the `EMISSION_FACTORS` dictionary. This default dataset can be overridden by providing a custom emissions factors dictionary when initializing the DataLoader, under the `emissions_factors` parameter.

```python
# Emission factors (kgCO2eq/MWh)
EMISSION_FACTORS = {
    "Fossil Hard coal": 820,
    "Fossil Gas":       490,
    "Nuclear":          12,
    "Wind Onshore":     11,
    "Wind Offshore":    12,
    "Solar":            48,
    "Hydro":            24,
    "Biomass":          230,
    "Fossil Oil":       840,
    "Fossil Brown coal/Lignite": 820,
    "Fossil Coal-derived gas": 820,
    "Fossil Peat": 820,
    "Geothermal": 38,
    "Waste": 230,
    "Other": 258,
    "Other renewable": 48,
    "Hydro Pumped Storage": 24,
    "Hydro Run-of-river and poundage": 24,
    "Hydro Water Reservoir": 24,
    "Energy storage": 358,
    "Marine": 10,
    # Imports # Electricity Maps 2024
    "Import_GB": 175,
    "Import_EE": 291,
    "Import_RU": 354.0,
}

```

---

## Dependencies

The package depends on the following libraries:
- pandas (>=3.0.5,<4.0.0)
- entsoe-py (>=0.8.0,<0.9.0)
- matplotlib (>=3.11.1,<4.0.0)
- linear-tree (fork compatible with scikit-learn >= 1.7.0)
