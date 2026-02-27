"""
regime/detector.py
Macro Regime Detection: HMM, Volatility Clustering, Trend-Filter
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


REGIME_LABELS = {
    0: "RISK_OFF_LOW_VOL",
    1: "RISK_ON_BULL",
    2: "CRISIS_HIGH_VOL",
    3: "TRANSITION",
}


class RegimeDetector:
    """
    Multi-method macro regime detection engine.

    Methods
    -------
    hmm_regimes        : Hidden Markov Model (2-4 states)
    gmm_regimes        : Gaussian Mixture Model clustering
    vol_regime         : Simple volatility-based regime (low/mid/high)
    trend_regime       : Dual moving-average trend filter
    composite_regime   : Ensemble vote across methods
    regime_transition  : Transition probability matrix
    """

    def __init__(
        self,
        n_regimes: int = 3,
        vol_window: int = 21,
        trend_short: int = 50,
        trend_long: int = 200,
        random_state: int = 42,
    ):
        self.n_regimes    = n_regimes
        self.vol_window   = vol_window
        self.trend_short  = trend_short
        self.trend_long   = trend_long
        self.random_state = random_state
        self.scaler       = StandardScaler()

    # ------------------------------------------------------------------
    # 1. HMM regimes
    # ------------------------------------------------------------------
    def hmm_regimes(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Fit Gaussian HMM to return series.
        Returns integer regime labels aligned with returns index.
        """
        if not HAS_HMM:
            # fallback to GMM
            return self.gmm_regimes(returns)

        X = returns.dropna().values.reshape(-1, 1)
        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
        )
        model.fit(X)
        states = model.predict(X)
        # reorder by volatility of each state (0=lowest vol)
        state_vol = {s: X[states == s].std() for s in range(self.n_regimes)}
        order = sorted(state_vol, key=state_vol.get)
        remap = {old: new for new, old in enumerate(order)}
        mapped = np.array([remap[s] for s in states])
        return pd.Series(mapped, index=returns.dropna().index, name="hmm_regime")

    # ------------------------------------------------------------------
    # 2. GMM regimes
    # ------------------------------------------------------------------
    def gmm_regimes(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        GMM clustering on (return, realised vol) feature space.
        """
        ret    = returns.dropna()
        rv     = ret.rolling(self.vol_window).std().dropna()
        common = ret.index.intersection(rv.index)
        X      = np.column_stack([ret[common], rv[common]])
        Xs     = self.scaler.fit_transform(X)

        gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_init=5,
        )
        gmm.fit(Xs)
        labels = gmm.predict(Xs)

        # sort by vol level
        comp_vol = {i: Xs[labels == i, 1].mean() for i in range(self.n_regimes)}
        order    = sorted(comp_vol, key=comp_vol.get)
        remap    = {old: new for new, old in enumerate(order)}
        mapped   = np.array([remap[s] for s in labels])
        return pd.Series(mapped, index=common, name="gmm_regime")

    # ------------------------------------------------------------------
    # 3. Volatility-based regime
    # ------------------------------------------------------------------
    def vol_regime(
        self,
        returns: pd.Series,
        low_pct: float = 0.33,
        high_pct: float = 0.67,
    ) -> pd.Series:
        """
        0 = low vol, 1 = mid vol, 2 = high vol.
        """
        rv   = returns.rolling(self.vol_window).std()
        lo   = rv.quantile(low_pct)
        hi   = rv.quantile(high_pct)
        reg  = pd.cut(rv, bins=[-np.inf, lo, hi, np.inf], labels=[0, 1, 2])
        return reg.astype(float).rename("vol_regime")

    # ------------------------------------------------------------------
    # 4. Trend regime (dual MA)
    # ------------------------------------------------------------------
    def trend_regime(
        self,
        prices: pd.Series,
    ) -> pd.Series:
        """
        1 = uptrend (fast MA > slow MA), 0 = downtrend.
        """
        fast = prices.rolling(self.trend_short).mean()
        slow = prices.rolling(self.trend_long).mean()
        return (fast > slow).astype(int).rename("trend_regime")

    # ------------------------------------------------------------------
    # 5. Composite regime (ensemble)
    # ------------------------------------------------------------------
    def composite_regime(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Combine GMM and vol regime into a composite label.
        Returns DataFrame with individual regime columns.
        """
        gmm = self.gmm_regimes(returns)
        vol = self.vol_regime(returns)

        df  = pd.DataFrame({"gmm": gmm, "vol": vol})
        if prices is not None:
            trend = self.trend_regime(prices)
            df["trend"] = trend

        df = df.dropna()
        # majority vote
        df["composite"] = df.mode(axis=1)[0].astype(int)
        return df

    # ------------------------------------------------------------------
    # 6. Transition probability matrix
    # ------------------------------------------------------------------
    def transition_matrix(
        self,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Empirical transition probability matrix between regime states.
        """
        states   = sorted(regimes.dropna().unique().astype(int))
        n        = len(states)
        counts   = np.zeros((n, n))
        reg_vals = regimes.dropna().astype(int).values
        for i in range(len(reg_vals) - 1):
            counts[reg_vals[i], reg_vals[i + 1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        probs    = np.where(row_sums > 0, counts / row_sums, 0.0)
        return pd.DataFrame(probs, index=states, columns=states)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    n   = 1000
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    ret = pd.Series(np.concatenate([
        np.random.normal(0.001, 0.005, 400),
        np.random.normal(-0.002, 0.020, 200),
        np.random.normal(0.001, 0.008, 400),
    ]), index=idx)
    prices = (1 + ret).cumprod()

    rd = RegimeDetector(n_regimes=3)
    comp = rd.composite_regime(ret, prices)
    print(comp.tail(10))
    print("\nTransition matrix:")
    print(rd.transition_matrix(comp["composite"]))
