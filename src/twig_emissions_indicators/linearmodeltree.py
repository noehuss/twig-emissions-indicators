import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lineartree import LinearTreeRegressor

@dataclass
class LinearTreeResults:
    """Container for model results and diagnostics."""
    model: LinearTreeRegressor
    mef: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_importance: pd.Series
    selected_features: List[str]
    metrics: Dict[str, float]
    scaler: StandardScaler
    time_features: List[str]
    baseline_metrics: Dict[str, float]

class LinearTreeModel(BaseEstimator, RegressorMixin):
    """
    A Linear Tree Model for marginal emissions factor estimation.
    Wraps `LinearTreeRegressor` and provides a full pipeline for training and evaluation.

    Parameters
    ----------
    base_estimator : estimator, default=LinearRegression(fit_intercept=False)
        The base estimator for each leaf in the linear tree.
    time_features : List[str], default=["Hour", "Month", "Dow"]
        List of circular time features to transform using cos/sin encoding.
    n_splits : int, default=5
        Number of splits for TimeSeriesSplit during hyperparameter tuning.
    max_depth : int, default=10
        Maximum depth for the LinearTreeRegressor.
    min_samples_leaf : float, default=0.01
        Minimum samples required to be at a leaf node.
    linear_features : List[int], default=[0]
        Indices of features to use as linear features in each leaf.
    split_features : List[int], default=None
        Indices of features to use for splitting. If None, all non-linear features are used.
    importance_threshold : float, default=0.01
        Threshold for feature selection based on importance.
    random_state : int, default=None
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress messages.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        time_features: List[str] = ["Hour", "Month", "Dow"],
        n_splits: int = 5,
        max_depth: int = 10,
        min_samples_leaf: float = 0.01,
        linear_features: List[int] = [0],
        split_features: Optional[List[int]] = None,
        importance_threshold: float = 0.01,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ):
        self.base_estimator = base_estimator if base_estimator is not None else LinearRegression(fit_intercept=False)
        self.time_features = time_features
        self.n_splits = n_splits
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.linear_features = linear_features
        self.split_features = split_features
        self.importance_threshold = importance_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.scaler_ = None
        self.model_ = None
        self.feature_importance_ = None
        self.selected_features_ = None
        self.baseline_metrics_ = None
        self.results_ = None

    def _preprocess_features(
        self,
        X: pd.DataFrame,
        dL: pd.Series,
    ) -> pd.DataFrame:
        """
        Preprocess features:
        - Apply cos/sin transformation to circular time features.
        - Remove highly correlated features.
        - Drop problematic columns.
        - Scale features.
        - Combine with dL.
        """
        X_transformed = X.copy()

        # Cos/sin transformation for circular time features
        for col in self.time_features:
            if col in X_transformed.columns:
                max_val = X_transformed[col].max()
                X_transformed[f"cos_{col}"] = np.cos(X_transformed[col] * 2 * np.pi / max_val)
                X_transformed.drop(columns=[col], inplace=True)

        # Remove highly correlated features
        corr_matrix = X_transformed.corr().abs()
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.95:
                    colname = corr_matrix.columns[i]
                    to_drop.add(colname)
        X_transformed.drop(columns=list(to_drop), inplace=True)

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler_.fit_transform(X_transformed),
            columns=X_transformed.columns,
            index=X_transformed.index,
        )

        # Combine with dL
        dL_col = dL.rename("dL")
        features = pd.concat([dL_col, X_scaled], axis=1)

        return features

    def _train_baseline(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Train a baseline linear regression model on dL only."""
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_train[["dL"]], y_train)

        y_train_pred = lr.predict(X_train[["dL"]])
        y_test_pred = lr.predict(X_test[["dL"]])

        metrics = {
            "lr_train_r2": r2_score(y_train, y_train_pred),
            "lr_test_r2": r2_score(y_test, y_test_pred),
            "lr_train_rmse": root_mean_squared_error(y_train, y_train_pred),
            "lr_test_rmse": root_mean_squared_error(y_test, y_test_pred),
            "lr_train_mape": mean_absolute_percentage_error(y_train, y_train_pred),
            "lr_test_mape": mean_absolute_percentage_error(y_test, y_test_pred),
            "average_empirical_mef": lr.coef_[0],
        }
        return metrics

    def _tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV."""
        param_grid = {
            "max_depth": [8, 10, self.max_depth],
            "min_samples_leaf": [0.01, self.min_samples_leaf],
        }

        lt_base = LinearTreeRegressor(
            base_estimator=self.base_estimator,
            criterion="rmse",
            linear_features=self.linear_features,
            split_features=self.split_features or list(range(1, X_train.shape[1])),
            n_jobs=-1,
        )

        search = GridSearchCV(
            lt_base,
            param_grid,
            cv=TimeSeriesSplit(n_splits=self.n_splits),
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        return search.best_params_

    def _select_features(self, X_train: pd.DataFrame, model: LinearTreeRegressor) -> Tuple:
        """Select features based on importance threshold."""
        feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        selected_features = feature_importance[feature_importance > self.importance_threshold].index.tolist()
        if "dL" not in selected_features:
            selected_features.insert(0, "dL")
        return selected_features, feature_importance

    def fit(
        self,
        dE: pd.Series,
        dL: pd.Series,
        X: pd.DataFrame,
        test_size: Optional[float] = None,
        test_year: Optional[int] = 2025,
    ) -> "LinearTreeModel":
        """
        Fit the LinearTreeModel.

        Parameters
        ----------
        dE : pd.Series
            Target variable (delta emissions).
        dL : pd.Series
            Delta load.
        X : pd.DataFrame
            Input features.
        test_size : float, optional
            Fraction of data to use for testing. If None, uses `test_year`.
        test_year : int, optional
            Year to use for testing. If None, uses the last year in the data.

        Returns
        -------
        self : LinearTreeModel
            Fitted model.
        """
        # Preprocess features
        features = self._preprocess_features(X, dL)
        df = pd.concat([dE.rename("dG"), features], axis=1).dropna()
        X_processed = df.drop(columns="dG")
        y = df["dG"]

        # Train/test split
        if test_size is not None:
            split_idx = int(len(df) * (1 - test_size))
            train_idx = df.index[:split_idx]
            test_idx = df.index[split_idx:]
        else:
            train_idx = df.index.year < test_year
            test_idx = df.index.year == test_year

        X_train, y_train = X_processed[train_idx], y[train_idx]
        X_test, y_test = X_processed[test_idx], y[test_idx]

        # Train baseline
        self.baseline_metrics_ = self._train_baseline(X_train, y_train, X_test, y_test)

        # Hyperparameter tuning
        best_params = self._tune_hyperparameters(X_train, y_train)
        if self.verbose:
            print(f"Best hyperparameters: {best_params}")

        # Train initial model
        self.model_ = LinearTreeRegressor(
            base_estimator=self.base_estimator,
            criterion="rmse",
            linear_features=self.linear_features,
            split_features=self.split_features or list(range(1, X_train.shape[1])),
            max_depth=best_params.get("max_depth", self.max_depth),
            min_samples_leaf=best_params.get("min_samples_leaf", self.min_samples_leaf),
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(X_train, y_train)

        # Feature selection
        self.selected_features_, self.feature_importance_ = self._select_features(X_train, self.model_)
        X_train_selected = X_train[self.selected_features_]
        X_test_selected = X_test[self.selected_features_]

        # Retrain with selected features
        self.model_ = LinearTreeRegressor(
            base_estimator=self.base_estimator,
            criterion="rmse",
            linear_features=[0],  # dL is always first
            split_features=list(range(1, len(self.selected_features_))),
            max_depth=best_params.get("max_depth", self.max_depth),
            min_samples_leaf=best_params.get("min_samples_leaf", self.min_samples_leaf),
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(X_train_selected, y_train)

        # Store results
        self.results_ = LinearTreeResults(
            model=self.model_,
            mef=self._extract_mef(X_test_selected),
            X_train=X_train_selected,
            X_test=X_test_selected,
            y_train=y_train,
            y_test=y_test,
            feature_importance=self.feature_importance_,
            selected_features=self.selected_features_,
            metrics=self._compute_metrics(X_train_selected, y_train, X_test_selected, y_test),
            scaler=self.scaler_,
            time_features=self.time_features,
            baseline_metrics=self.baseline_metrics_,
        )

        return self

    def _extract_mef(self, X: pd.DataFrame) -> pd.Series:
        """Extract marginal emissions factors from the model's leaves."""
        leaves = self.model_.summary(only_leaves=True)
        leaves_index = self.model_.apply(X)
        mef = pd.Series(np.zeros(len(X)), index=X.index, name="g_k")
        for leaf_id in np.unique(leaves_index):
            mask = leaves_index == leaf_id
            coefs = leaves[leaf_id]["models"].coef_
            mef[mask] = coefs[0]  # mef is coefficient on dL
        return mef

    def _compute_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Compute evaluation metrics for the model."""
        y_train_pred = self.model_.predict(X_train)
        y_test_pred = self.model_.predict(X_test)
        return {
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "train_rmse": root_mean_squared_error(y_train, y_train_pred),
            "test_rmse": root_mean_squared_error(y_test, y_test_pred),
            "train_mape": mean_absolute_percentage_error(y_train, y_train_pred),
            "test_mape": mean_absolute_percentage_error(y_test, y_test_pred),
        }

    def predict(self, X: pd.DataFrame, dL: Optional[pd.Series] = None) -> pd.Series:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        dL : pd.Series, optional
            Delta load. If None, assumes X already includes dL.

        Returns
        -------
        pd.Series
            Predicted values.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")

        if dL is not None:
            features = self._preprocess_features(X, dL)
        else:
            features = X.copy()

        # Select features
        if self.selected_features_ is not None:
            features = features[self.selected_features_]

        return pd.Series(self.model_.predict(features), index=features.index)

    def score(self, X: pd.DataFrame, y: pd.Series, dL: Optional[pd.Series] = None) -> float:
        """
        Score the model using R².

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            True target values.
        dL : pd.Series, optional
            Delta load. If None, assumes X already includes dL.

        Returns
        -------
        float
            R² score.
        """
        y_pred = self.predict(X, dL)
        return r2_score(y, y_pred)

    def get_results(self) -> LinearTreeResults:
        """Return the full results container."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        return self.results_

    def save(self, path: Union[str, Path]) -> None:
        """Save the model and its metadata to disk."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        joblib.dump(
            {
                "model": self.model_,
                "scaler": self.scaler_,
                "selected_features": self.selected_features_,
                "feature_importance": self.feature_importance_,
                "baseline_metrics": self.baseline_metrics_,
                "time_features": self.time_features,
                "importance_threshold": self.importance_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearTreeModel":
        """Load a saved model from disk."""
        data = joblib.load(path)
        model = cls(
            time_features=data["time_features"],
            importance_threshold=data["importance_threshold"],
        )
        model.model_ = data["model"]
        model.scaler_ = data["scaler"]
        model.selected_features_ = data["selected_features"]
        model.feature_importance_ = data["feature_importance"]
        model.baseline_metrics_ = data["baseline_metrics"]
        return model
