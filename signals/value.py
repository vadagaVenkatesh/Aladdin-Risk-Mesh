"""
signals/value.py
Value & Valuation Signals for Global Macro Long/Short
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class ValueSignals:
    """
    Compute fundamental value and relative-value signals
    suitable for macro cross-asset positioning.

    Signals implemented
    -------------------
    1. Real yield spread  (bond vs equity earnings yield)
    2. FX purchasing-power-parity (PPP) deviation
    3. Commodity term-structure value (spot vs rolling futures)
    4. Equity CAPE-based value
    5. Credit spread carry-value composite
    """

    def __init__(self, lookback: int = 252, z_cap: float = 3.0):
        self.lookback = lookback
        self.z_cap   = z_cap

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _zscore(self, series: pd.Series) -> pd.Series:
        """Rolling z-score, clipped at z_cap."""
        mu  = series.rolling(self.lookback, min_periods=60).mean()
        sig = series.rolling(self.lookback, min_periods=60).std()
        z   = (series - mu) / (sig + 1e-9)
        return z.clip(-self.z_cap, self.z_cap)

    # ------------------------------------------------------------------
    # 1. Real yield spread
    # ------------------------------------------------------------------
    def real_yield_spread(
        self,
        nominal_yield: pd.Series,
        breakeven_inflation: pd.Series,
        earnings_yield: pd.Series,
    ) -> pd.Series:
        """
        Real bond yield minus equity earnings yield.
        Positive => bonds cheap vs equities (favour bonds).
        """
        real_yield = nominal_yield - breakeven_inflation
        spread     = real_yield - earnings_yield
        return self._zscore(spread).rename("real_yield_spread")

    # ------------------------------------------------------------------
    # 2. FX PPP deviation
    # ------------------------------------------------------------------
    def fx_ppp_deviation(
        self,
        spot_rate: pd.Series,
        ppp_rate: pd.Series,
    ) -> pd.Series:
        """
        Log deviation of spot from PPP fair value.
        Positive => currency is expensive vs PPP (short signal).
        """
        dev = np.log(spot_rate / ppp_rate)
        return self._zscore(dev).rename("fx_ppp_dev")

    # ------------------------------------------------------------------
    # 3. Commodity term-structure value
    # ------------------------------------------------------------------
    def commodity_ts_value(
        self,
        spot: pd.Series,
        futures_12m: pd.Series,
    ) -> pd.Series:
        """
        Annualised roll-yield (spot / futures_12m - 1).
        High roll-yield => commodity is in backwardation => buy signal.
        """
        roll_yield = spot / futures_12m - 1.0
        return self._zscore(roll_yield).rename("commodity_ts_value")

    # ------------------------------------------------------------------
    # 4. CAPE-based equity value
    # ------------------------------------------------------------------
    def cape_value(
        self,
        price: pd.Series,
        earnings_10y_avg: pd.Series,
    ) -> pd.Series:
        """
        Inverted CAPE (1/CAPE) z-scored over history.
        High inverse CAPE => cheap equities => positive value signal.
        """
        inv_cape = earnings_10y_avg / (price + 1e-9)
        return self._zscore(inv_cape).rename("cape_value")

    # ------------------------------------------------------------------
    # 5. Credit carry-value composite
    # ------------------------------------------------------------------
    def credit_carry_value(
        self,
        oas_spread: pd.Series,
        expected_default_loss: pd.Series,
    ) -> pd.Series:
        """
        Excess spread after expected loss.
        Higher excess spread => better carry-value => positive signal.
        """
        excess = oas_spread - expected_default_loss
        return self._zscore(excess).rename("credit_carry_value")

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------
    def composite(
        self,
        signals: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Weighted average of provided value signals.
        Default: equal weight.
        """
        df = pd.DataFrame(signals)
        if weights is None:
            weights = {c: 1.0 / len(df.columns) for c in df.columns}
        w  = pd.Series(weights)
        w  = w / w.sum()
        composite = df.mul(w).sum(axis=1)
        return composite.rename("value_composite")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    idx = pd.date_range("2010-01-01", periods=1000, freq="B")

    nominal_yield   = pd.Series(np.random.normal(0.03, 0.01, 1000), index=idx)
    breakeven       = pd.Series(np.random.normal(0.02, 0.005, 1000), index=idx)
    earnings_yield  = pd.Series(np.random.normal(0.05, 0.01, 1000), index=idx)
    spot_fx         = pd.Series(np.exp(np.random.normal(0, 0.1, 1000).cumsum() * 0.01 + np.log(1.2)), index=idx)
    ppp_fx          = pd.Series(np.full(1000, 1.15), index=idx)
    spot_comm       = pd.Series(np.random.lognormal(5, 0.3, 1000), index=idx)
    fut_comm        = spot_comm * np.random.uniform(0.95, 1.05, 1000)
    price           = pd.Series(np.random.lognormal(7, 0.3, 1000), index=idx)
    earnings_10y    = pd.Series(np.random.normal(100, 10, 1000), index=idx)
    oas             = pd.Series(np.random.normal(0.02, 0.005, 1000), index=idx)
    edl             = pd.Series(np.random.normal(0.005, 0.001, 1000), index=idx)

    vs = ValueSignals(lookback=252)

    s1 = vs.real_yield_spread(nominal_yield, breakeven, earnings_yield)
    s2 = vs.fx_ppp_deviation(spot_fx, ppp_fx)
    s3 = vs.commodity_ts_value(spot_comm, fut_comm)
    s4 = vs.cape_value(price, earnings_10y)
    s5 = vs.credit_carry_value(oas, edl)

    comp = vs.composite({"rys": s1, "ppp": s2, "comm": s3, "cape": s4, "credit": s5})
    print(comp.tail(10))
