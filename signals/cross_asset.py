"""
signals/cross_asset.py
Cross-Asset Correlation & Regime-Conditional Signals
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr


class CrossAssetSignals:
    """
    Cross-asset signal generation:
    1. Rolling cross-asset correlation regime
    2. Risk-on / Risk-off composite score
    3. Equity-bond leadership signal
    4. Dollar-sensitivity (DXY beta) divergence
    5. Commodity-equity divergence (growth proxy)
    """

    def __init__(self, short_window: int = 21, long_window: int = 63, z_cap: float = 3.0):
        self.short_window = short_window
        self.long_window  = long_window
        self.z_cap        = z_cap

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _zscore(self, s: pd.Series, window: int) -> pd.Series:
        mu  = s.rolling(window, min_periods=max(10, window // 3)).mean()
        sig = s.rolling(window, min_periods=max(10, window // 3)).std()
        return ((s - mu) / (sig + 1e-9)).clip(-self.z_cap, self.z_cap)

    def _rolling_corr(self, a: pd.Series, b: pd.Series, window: int) -> pd.Series:
        return a.rolling(window).corr(b)

    # ------------------------------------------------------------------
    # 1. Correlation-regime shift
    # ------------------------------------------------------------------
    def corr_regime(
        self,
        eq_returns: pd.Series,
        bond_returns: pd.Series,
    ) -> pd.Series:
        """
        Short-minus-long rolling equity-bond correlation.
        Negative shift => risk-off regime emerging.
        """
        short_corr = self._rolling_corr(eq_returns, bond_returns, self.short_window)
        long_corr  = self._rolling_corr(eq_returns, bond_returns, self.long_window)
        regime     = short_corr - long_corr
        return self._zscore(regime, self.long_window).rename("corr_regime")

    # ------------------------------------------------------------------
    # 2. Risk-on / Risk-off composite
    # ------------------------------------------------------------------
    def risk_on_off(
        self,
        vix: pd.Series,
        hy_spread: pd.Series,
        em_fx_vol: pd.Series,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Composite risk appetite index.
        High positive => risk-on environment.
        """
        if weights is None:
            weights = {"vix": 1/3, "hy": 1/3, "em": 1/3}

        z_vix = -self._zscore(vix,       self.long_window)  # inverted: high VIX => risk-off
        z_hy  = -self._zscore(hy_spread, self.long_window)  # inverted: wide spread => risk-off
        z_em  = -self._zscore(em_fx_vol, self.long_window)  # inverted: high EM vol => risk-off

        composite = (weights["vix"] * z_vix +
                     weights["hy"]  * z_hy  +
                     weights["em"]  * z_em)
        return composite.rename("risk_on_off")

    # ------------------------------------------------------------------
    # 3. Equity-bond leadership
    # ------------------------------------------------------------------
    def eq_bond_leadership(
        self,
        eq_returns: pd.Series,
        bond_returns: pd.Series,
        lag: int = 5,
    ) -> pd.Series:
        """
        Granger-proxy: equity returns lagged vs current bond returns.
        Positive => equities leading bonds (growth expectation building).
        """
        leadership = eq_returns.shift(lag).rolling(self.short_window).corr(bond_returns)
        return self._zscore(leadership, self.long_window).rename("eq_bond_lead")

    # ------------------------------------------------------------------
    # 4. Dollar-sensitivity divergence
    # ------------------------------------------------------------------
    def dollar_sensitivity(
        self,
        dxy_returns: pd.Series,
        asset_returns: pd.Series,
    ) -> pd.Series:
        """
        Rolling beta of asset to DXY.
        Divergence from long-run beta = potential mean-reversion signal.
        """
        short_beta = asset_returns.rolling(self.short_window).cov(dxy_returns) / \
                     (dxy_returns.rolling(self.short_window).var() + 1e-9)
        long_beta  = asset_returns.rolling(self.long_window).cov(dxy_returns) / \
                     (dxy_returns.rolling(self.long_window).var() + 1e-9)
        divergence = short_beta - long_beta
        return self._zscore(divergence, self.long_window).rename("dxy_beta_div")

    # ------------------------------------------------------------------
    # 5. Commodity-equity divergence
    # ------------------------------------------------------------------
    def commodity_equity_div(
        self,
        comm_returns: pd.Series,
        eq_returns: pd.Series,
    ) -> pd.Series:
        """
        Rolling ratio of commodity to equity momentum.
        High ratio => commodity outperforming => inflation/growth signal.
        """
        comm_mom = comm_returns.rolling(self.short_window).mean()
        eq_mom   = eq_returns.rolling(self.short_window).mean()
        ratio    = comm_mom - eq_mom
        return self._zscore(ratio, self.long_window).rename("comm_eq_div")

    # ------------------------------------------------------------------
    # Master composite
    # ------------------------------------------------------------------
    def composite(
        self,
        signals: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        df = pd.DataFrame(signals)
        if weights is None:
            weights = {c: 1.0 / len(df.columns) for c in df.columns}
        w = pd.Series(weights) / sum(weights.values())
        return df.mul(w).sum(axis=1).rename("cross_asset_composite")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(7)
    n   = 500
    idx = pd.date_range("2018-01-01", periods=n, freq="B")

    eq   = pd.Series(np.random.normal(0.0003, 0.01, n), index=idx)
    bond = pd.Series(np.random.normal(0.0001, 0.005, n), index=idx)
    vix  = pd.Series(np.abs(np.random.normal(18, 5, n)), index=idx)
    hy   = pd.Series(np.abs(np.random.normal(0.04, 0.01, n)), index=idx)
    emfx = pd.Series(np.abs(np.random.normal(0.008, 0.002, n)), index=idx)
    dxy  = pd.Series(np.random.normal(0.0001, 0.005, n), index=idx)
    comm = pd.Series(np.random.normal(0.0002, 0.012, n), index=idx)

    ca = CrossAssetSignals()
    s1 = ca.corr_regime(eq, bond)
    s2 = ca.risk_on_off(vix, hy, emfx)
    s3 = ca.eq_bond_leadership(eq, bond)
    s4 = ca.dollar_sensitivity(dxy, eq)
    s5 = ca.commodity_equity_div(comm, eq)

    comp = ca.composite({"corr": s1, "roo": s2, "lead": s3, "dxy": s4, "comm": s5})
    print(comp.tail(10))
