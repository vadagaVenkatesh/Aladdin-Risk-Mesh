"""
ml/rank_model.py
Cross-Sectional Rank Model: signal ranking, portfolio tilts, IC decay
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RankModel:
    """
    Cross-sectional rank-based signal combination and portfolio tilting.

    Key methods
    -----------
    rank_signals       : Percentile-rank each signal cross-sectionally
    combine_signals    : Weighted combination of ranked signals
    rank_ic            : Rank IC over time (signal quality)
    ic_decay           : IC decay curve by lag
    rank_to_weights    : Convert composite rank signal to long/short weights
    turnover_analysis  : Rank persistence and implied turnover
    """

    def __init__(
        self,
        min_assets: int = 5,
        winsorize_pct: float = 0.05,
    ):
        self.min_assets    = min_assets
        self.winsorize_pct = winsorize_pct

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _winsorize(self, s: pd.Series, pct: float) -> pd.Series:
        lo = s.quantile(pct)
        hi = s.quantile(1 - pct)
        return s.clip(lo, hi)

    def _cross_rank(self, s: pd.Series) -> pd.Series:
        """Rank cross-sectionally, scaled to [-1, 1]."""
        r = s.rank(method="average", na_option="keep")
        r = (r - 1) / (r.count() - 1) * 2 - 1
        return r

    # ------------------------------------------------------------------
    # 1. Rank signals
    # ------------------------------------------------------------------
    def rank_signals(
        self,
        signals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank each signal cross-sectionally at each time step.
        Returns DataFrame of same shape with values in [-1, 1].
        """
        return signals.apply(self._cross_rank, axis=1)

    # ------------------------------------------------------------------
    # 2. Combine signals
    # ------------------------------------------------------------------
    def combine_signals(
        self,
        ranked: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Weighted combination of ranked signals to composite score.
        """
        if weights is None:
            weights = {c: 1.0 / len(ranked.columns) for c in ranked.columns}
        w = pd.Series(weights)
        w = w / w.sum()
        return ranked.mul(w).sum(axis=1).rename("composite")

    # ------------------------------------------------------------------
    # 3. Rank IC
    # ------------------------------------------------------------------
    def rank_ic(
        self,
        signal_ts: pd.DataFrame,
        returns_ts: pd.DataFrame,
        forward: int = 1,
    ) -> pd.DataFrame:
        """
        Rolling daily rank IC per signal.
        Returns DataFrame(index=dates, columns=signals).
        """
        fwd_ret = returns_ts.shift(-forward)
        ics = {}
        for sig in signal_ts.columns:
            ic_series = []
            for date in signal_ts.index:
                s = signal_ts.loc[date].dropna()
                r = fwd_ret.loc[date].dropna() if date in fwd_ret.index else pd.Series()
                common = s.index.intersection(r.index)
                if len(common) < self.min_assets:
                    ic_series.append(np.nan)
                    continue
                ic, _ = spearmanr(s[common], r[common])
                ic_series.append(ic)
            ics[sig] = pd.Series(ic_series, index=signal_ts.index)
        return pd.DataFrame(ics)

    # ------------------------------------------------------------------
    # 4. IC decay
    # ------------------------------------------------------------------
    def ic_decay(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        max_lag: int = 21,
    ) -> pd.Series:
        """
        Compute average rank IC at each lag from 1 to max_lag.
        """
        lags, ics = [], []
        for lag in range(1, max_lag + 1):
            fwd = returns.shift(-lag)
            ic_vals = []
            for date in signal.index:
                s = signal.loc[date].dropna()
                r = fwd.loc[date].dropna() if date in fwd.index else pd.Series()
                common = s.index.intersection(r.index)
                if len(common) < self.min_assets:
                    continue
                ic, _ = spearmanr(s[common], r[common])
                if not np.isnan(ic):
                    ic_vals.append(ic)
            lags.append(lag)
            ics.append(np.mean(ic_vals) if ic_vals else np.nan)
        return pd.Series(ics, index=lags, name="ic_decay")

    # ------------------------------------------------------------------
    # 5. Rank to long/short weights
    # ------------------------------------------------------------------
    def rank_to_weights(
        self,
        composite: pd.Series,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        leverage: float = 1.0,
    ) -> pd.Series:
        """
        Convert composite scores to long/short weights.
        Top long_pct => long; bottom short_pct => short.
        """
        n = len(composite.dropna())
        lo_cut = composite.quantile(short_pct)
        hi_cut = composite.quantile(1 - long_pct)

        weights = pd.Series(0.0, index=composite.index)
        longs  = composite >= hi_cut
        shorts = composite <= lo_cut

        n_long  = longs.sum()
        n_short = shorts.sum()

        if n_long  > 0: weights[longs]  =  leverage / 2 / n_long
        if n_short > 0: weights[shorts] = -leverage / 2 / n_short
        return weights

    # ------------------------------------------------------------------
    # 6. Turnover analysis
    # ------------------------------------------------------------------
    def turnover_analysis(
        self,
        weights_ts: pd.DataFrame,
    ) -> pd.Series:
        """
        One-way turnover at each rebalancing date.
        """
        diffs  = weights_ts.diff().abs()
        return (diffs.sum(axis=1) / 2).rename("turnover")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(5)
    dates   = pd.date_range("2020-01-01", periods=100, freq="B")
    assets  = ["A", "B", "C", "D", "E", "F"]
    signals = pd.DataFrame(np.random.randn(100, 6), index=dates, columns=assets)
    returns = pd.DataFrame(np.random.randn(100, 6) * 0.01, index=dates, columns=assets)

    rm     = RankModel()
    ranked = rm.rank_signals(signals)
    comp   = rm.combine_signals(ranked)
    w      = rm.rank_to_weights(comp.iloc[-1], long_pct=0.33, short_pct=0.33)
    print("Latest weights:")
    print(w.round(3))
