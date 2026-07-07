"""
Baseline tabular-only regressors for comparison against MultimodalFusionNet.

All models share the same interface:
    X: (n_samples, seq_len × n_features)  — flattened 24-step tabular windows
    y: (n_samples,)                        — engineered target values

Models: HistoricalMeanModel, LinearRegressionModel, XGBoostModel.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _XGBOOST_AVAILABLE = False
    logger.warning("xgboost not installed — XGBoostModel will use GradientBoostingRegressor")

from sklearn.linear_model import LinearRegression


# ─── 1. Historical Mean ───────────────────────────────────────────────────────

class HistoricalMeanModel:
    """
    Predicts the training-set mean for every test sample.

    Serves as the naive benchmark for the historical-mean R²_OOS denominator.
    """

    name = "HistoricalMean"

    def __init__(self):
        self.train_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistoricalMeanModel":
        self.train_mean = float(np.mean(y))
        logger.info(f"  [{self.name}] train_mean = {self.train_mean:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.train_mean, dtype=np.float32)

    def get_train_mean(self) -> float:
        return self.train_mean


# ─── 2. Linear Regression ────────────────────────────────────────────────────

class LinearRegressionModel:
    """sklearn LinearRegression on flattened 24-step tabular windows (168 features)."""

    name = "LinearRegression"

    def __init__(self):
        self._model = LinearRegression(fit_intercept=True, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        self._model.fit(X, y)
        logger.info(f"  [{self.name}] fitted on {X.shape[0]} samples, {X.shape[1]} features")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(np.float32)


# ─── 3. XGBoost ──────────────────────────────────────────────────────────────

class XGBoostModel:
    """
    XGBRegressor on flattened 24-step tabular windows.

    Falls back to sklearn GradientBoostingRegressor if xgboost is not installed.
    Hyperparameters: n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8.
    """

    name = "XGBoost" if _XGBOOST_AVAILABLE else "GradientBoosting"

    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.05, seed: int = 42):
        if _XGBOOST_AVAILABLE:
            self._model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            )
        else:
            self._model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                random_state=seed,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self._model.fit(X, y)
        logger.info(f"  [{self.name}] fitted on {X.shape[0]} samples")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(np.float32)


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_all_models(seed: int = 42):
    """Return one fresh instance of each baseline model."""
    return [
        HistoricalMeanModel(),
        LinearRegressionModel(),
        XGBoostModel(seed=seed),
    ]
