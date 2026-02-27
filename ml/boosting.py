"""
ml/boosting.py
Gradient Boosting Models for Return Prediction & Feature Importance
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class MacroBooster:
    """
    Gradient Boosting wrapper for macro return prediction.

    Supports: LightGBM, XGBoost, sklearn GBM, Random Forest.
    Provides feature importance, SHAP-proxy decomposition,
    rolling retrain, and conformal prediction intervals.
    """

    def __init__(
        self,
        model_type: str = "lgbm",
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample: float = 0.8,
        window: int = 252,
        n_splits: int = 5,
    ):
        self.model_type    = model_type
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.colsample     = colsample
        self.window        = window
        self.n_splits      = n_splits
        self.model         = None
        self.scaler        = StandardScaler()

    def _build_model(self):
        if self.model_type == "lgbm" and HAS_LGB:
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample,
                random_state=42,
                verbose=-1,
            )
        elif self.model_type == "xgb" and HAS_XGB:
            return XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample,
                random_state=42,
                verbosity=0,
            )
        elif self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )
        else:
            # fallback sklearn GBM
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
            )

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MacroBooster":
        self.model = self._build_model()
        self.model.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.Series:
        """
        Normalised feature importance sorted descending.
        """
        imp = self.model.feature_importances_
        imp = imp / (imp.sum() + 1e-9)
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Rolling retrain (walk-forward)
    # ------------------------------------------------------------------
    def rolling_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """
        Walk-forward predictions using rolling window.
        Returns out-of-sample predictions aligned with y index.
        """
        preds = pd.Series(np.nan, index=y.index)
        for i in range(self.window, len(X)):
            Xtr = X.iloc[i - self.window: i]
            ytr = y.iloc[i - self.window: i]
            Xte = X.iloc[[i]]
            m   = self._build_model()
            m.fit(Xtr, ytr)
            preds.iloc[i] = m.predict(Xte)[0]
        return preds.dropna()

    # ------------------------------------------------------------------
    # Conformal prediction interval (split conformal)
    # ------------------------------------------------------------------
    def conformal_interval(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        X_test: pd.DataFrame,
        alpha: float = 0.10,
    ) -> pd.DataFrame:
        """
        Split conformal prediction intervals at 1-alpha coverage.
        Returns DataFrame with columns [pred, lower, upper].
        """
        m = self._build_model()
        m.fit(X_train, y_train)

        cal_resid = np.abs(y_cal.values - m.predict(X_cal))
        q = np.quantile(cal_resid, 1 - alpha)

        preds = m.predict(X_test)
        return pd.DataFrame({
            "pred":  preds,
            "lower": preds - q,
            "upper": preds + q,
        }, index=X_test.index)

    # ------------------------------------------------------------------
    # Time-series CV
    # ------------------------------------------------------------------
    def cv_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        r2s  = []
        for tr, te in tscv.split(X):
            m = self._build_model()
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            r2s.append(r2_score(y.iloc[te], pred))
        return {"r2_mean": round(np.mean(r2s), 4), "r2_std": round(np.std(r2s), 4)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(1)
    n, k = 600, 8
    idx  = pd.date_range("2015-01-01", periods=n, freq="B")
    cols = [f"f{i}" for i in range(k)]
    X    = pd.DataFrame(np.random.randn(n, k), index=idx, columns=cols)
    y    = pd.Series(X.iloc[:, 0] * 0.4 - X.iloc[:, 1] * 0.3 +
                     0.02 * np.random.randn(n), index=idx)

    mb = MacroBooster(model_type="sklearn", n_estimators=100)
    mb.fit(X.iloc[:400], y.iloc[:400])
    print("Feature importance:")
    print(mb.feature_importance())

    cv = mb.cv_score(X, y)
    print(f"\nCV R2: {cv['r2_mean']} +/- {cv['r2_std']}")
