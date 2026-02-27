"""
ml/panel_regression.py
Panel Regression & Factor Return Estimation
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")


class PanelRegression:
    """
    Panel (pooled) regression for factor return estimation.

    Features
    --------
    - Pooled OLS / Ridge / Lasso / ElasticNet
    - Rolling window estimation
    - Cross-sectional / time-series decomposition
    - Fama-MacBeth two-pass estimation
    - Information Coefficient (IC) analysis
    """

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        window: int = 252,
        n_splits: int = 5,
    ):
        self.model_type = model_type
        self.alpha      = alpha
        self.l1_ratio   = l1_ratio
        self.window     = window
        self.n_splits   = n_splits
        self.scaler     = StandardScaler()
        self.model      = self._build_model()

    def _build_model(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha)
        elif self.model_type == "lasso":
            return Lasso(alpha=self.alpha, max_iter=5000)
        elif self.model_type == "elasticnet":
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)
        else:
            raise ValueError(f"Unknown model: {self.model_type}")

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "PanelRegression":
        """
        Fit the panel model.
        X : feature matrix (observations x features)
        y : return vector
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def factor_loadings(self) -> pd.Series:
        """Return named factor loadings."""
        return pd.Series(
            self.model.coef_,
            index=self.feature_names_,
            name="loading",
        )

    # ------------------------------------------------------------------
    # Rolling estimation
    # ------------------------------------------------------------------
    def rolling_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Rolling window factor loadings over time.
        Returns DataFrame(index=dates, columns=factors).
        """
        results = []
        dates   = []
        for i in range(self.window, len(X)):
            Xi = X.iloc[i - self.window: i]
            yi = y.iloc[i - self.window: i]
            try:
                m = self._build_model()
                Xs = self.scaler.fit_transform(Xi)
                m.fit(Xs, yi)
                results.append(m.coef_)
                dates.append(X.index[i])
            except Exception:
                continue

        return pd.DataFrame(
            results, index=dates, columns=X.columns
        )

    # ------------------------------------------------------------------
    # Fama-MacBeth two-pass
    # ------------------------------------------------------------------
    def fama_macbeth(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        First pass:  time-series beta estimation per asset.
        Second pass: cross-sectional regression each period.

        Returns (risk_premia_mean, risk_premia_std)
        """
        # First pass: estimate betas per asset
        betas = {}
        for col in returns.columns:
            Xi = factors.loc[returns.index]
            yi = returns[col].dropna()
            Xi = Xi.loc[yi.index]
            if len(yi) < 60:
                continue
            m  = Ridge(alpha=self.alpha)
            Xs = StandardScaler().fit_transform(Xi)
            m.fit(Xs, yi)
            betas[col] = m.coef_

        beta_df = pd.DataFrame(betas, index=factors.columns).T

        # Second pass: cross-sectional each period
        lambdas = []
        for date in returns.index:
            ri = returns.loc[date].dropna()
            Bi = beta_df.loc[ri.index].dropna()
            ri = ri.loc[Bi.index]
            if len(ri) < 5:
                continue
            m  = Ridge(alpha=0.1)
            m.fit(Bi.values, ri.values)
            lambdas.append(m.coef_)

        if not lambdas:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        lam_df = pd.DataFrame(lambdas, columns=factors.columns)
        return lam_df.mean(), lam_df.std()

    # ------------------------------------------------------------------
    # IC analysis
    # ------------------------------------------------------------------
    def information_coefficient(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        periods: List[int] = [1, 5, 21],
    ) -> pd.DataFrame:
        """
        Rank IC between each signal and forward returns.
        Returns DataFrame(index=signal, columns=period).
        """
        from scipy.stats import spearmanr
        ics = {}
        for sig in signals.columns:
            ics[sig] = {}
            for p in periods:
                fwd = forward_returns.shift(-p).dropna()
                sig_aligned = signals[sig].loc[fwd.index].dropna()
                fwd_aligned = fwd.loc[sig_aligned.index]
                if len(fwd_aligned) < 20:
                    ics[sig][p] = np.nan
                    continue
                ic, _ = spearmanr(sig_aligned, fwd_aligned)
                ics[sig][p] = round(ic, 4)
        return pd.DataFrame(ics).T

    # ------------------------------------------------------------------
    # Cross-validation score
    # ------------------------------------------------------------------
    def cv_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        TimeSeriesSplit cross-validation R2 and MSE.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        r2s  = []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            m  = self._build_model()
            Xs = StandardScaler()
            m.fit(Xs.fit_transform(Xtr), ytr)
            pred = m.predict(Xs.transform(Xte))
            r2s.append(r2_score(yte, pred))
        return {"r2_mean": np.mean(r2s), "r2_std": np.std(r2s)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    n, k = 500, 5
    idx  = pd.date_range("2015-01-01", periods=n, freq="B")
    cols = [f"f{i}" for i in range(k)]
    X    = pd.DataFrame(np.random.randn(n, k), index=idx, columns=cols)
    y    = pd.Series(X @ np.array([0.5, -0.3, 0.2, 0.1, -0.4]) + 0.01 * np.random.randn(n), index=idx)

    pr = PanelRegression(model_type="ridge", alpha=1.0)
    pr.fit(X, y)
    print("Factor loadings:")
    print(pr.factor_loadings())

    cv = pr.cv_score(X, y)
    print(f"\nCV R2: {cv['r2_mean']:.4f} +/- {cv['r2_std']:.4f}")
