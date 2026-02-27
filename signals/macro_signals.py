"""Macro Signal Engine for Aladdin-Risk-Mesh.

Generates signals from macro-economic data including:
- Yield curve (2s10s spread, real yields)
- PMI composite indicators
- Central bank policy surprise index
- Credit impulse (change in credit flow / GDP)
- Global growth momentum
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MacroSignalConfig:
    """Configuration for macro signal generation."""
    yield_curve_window: int = 63        # ~3 months
    pmi_smoothing: int = 3              # EWM span for PMI smoothing
    credit_impulse_window: int = 252    # 1-year window for credit impulse
    growth_momentum_window: int = 126   # 6-month growth momentum
    z_score_window: int = 252           # Lookback for z-score normalization
    signal_clip: float = 3.0            # Clip signals at +/- 3 sigma


class MacroSignalEngine:
    """
    Composite macro signal engine.

    Combines yield curve, PMI, credit impulse, and central bank
    policy signals into a unified directional macro score.

    Parameters
    ----------
    config : MacroSignalConfig
        Engine configuration parameters.
    """

    def __init__(self, config: Optional[MacroSignalConfig] = None):
        self.config = config or MacroSignalConfig()
        self._signal_history: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Yield Curve Signals
    # ------------------------------------------------------------------

    def yield_curve_signal(
        self,
        short_rate: pd.Series,
        long_rate: pd.Series,
    ) -> pd.Series:
        """
        Compute normalized 2s10s (or equivalent) yield curve signal.

        Positive values => steepening (risk-on / growth), negative => inversion.

        Parameters
        ----------
        short_rate : pd.Series  Short-end yield (e.g., 2Y Treasury)
        long_rate  : pd.Series  Long-end yield (e.g., 10Y Treasury)

        Returns
        -------
        pd.Series  Z-scored yield-curve spread signal
        """
        spread = long_rate - short_rate
        signal = self._z_score(spread)
        self._signal_history["yield_curve"] = signal
        return signal

    def real_yield_signal(self, nominal_yield: pd.Series, inflation_breakeven: pd.Series) -> pd.Series:
        """
        Real yield signal: negative real yields => risk-on / commodity bullish.

        Returns z-scored real yield (positive = tight financial conditions).
        """
        real_yield = nominal_yield - inflation_breakeven
        # Invert: higher real yield => bearish for risk assets
        signal = -self._z_score(real_yield)
        self._signal_history["real_yield"] = signal
        return signal

    # ------------------------------------------------------------------
    # PMI / Activity Signals
    # ------------------------------------------------------------------

    def pmi_signal(self, pmi_composite: pd.Series) -> pd.Series:
        """
        PMI-based growth signal.

        Uses:
          - Level relative to 50 (expansion / contraction)
          - Rate-of-change (acceleration / deceleration)
          - Z-score normalization over lookback window

        Returns
        -------
        pd.Series  Combined PMI signal
        """
        cfg = self.config
        # Smooth raw PMI
        pmi_smooth = pmi_composite.ewm(span=cfg.pmi_smoothing, adjust=False).mean()
        # Level component: deviation from 50
        level = pmi_smooth - 50.0
        # Momentum component: 3-month change
        momentum = pmi_smooth.diff(3)
        # Combine and normalize
        combined = 0.6 * level + 0.4 * momentum
        signal = self._z_score(combined)
        self._signal_history["pmi"] = signal
        return signal

    # ------------------------------------------------------------------
    # Credit Impulse
    # ------------------------------------------------------------------

    def credit_impulse_signal(
        self,
        credit_flow: pd.Series,
        nominal_gdp: pd.Series,
    ) -> pd.Series:
        """
        Credit impulse = change in credit extension as % of GDP.

        A positive impulse leads economic activity by 3-6 months.

        Parameters
        ----------
        credit_flow : pd.Series   New credit extended (absolute)
        nominal_gdp : pd.Series   Nominal GDP (same frequency)

        Returns
        -------
        pd.Series  Z-scored credit impulse signal
        """
        cfg = self.config
        # Credit as % of GDP
        credit_pct = credit_flow / nominal_gdp
        # Impulse = change in credit % GDP
        impulse = credit_pct.diff(cfg.credit_impulse_window // 4)
        signal = self._z_score(impulse)
        self._signal_history["credit_impulse"] = signal
        return signal

    # ------------------------------------------------------------------
    # Central Bank Policy Surprise
    # ------------------------------------------------------------------

    def policy_surprise_signal(
        self,
        actual_rate_decision: pd.Series,
        market_expected_rate: pd.Series,
    ) -> pd.Series:
        """
        Central bank policy surprise index.

        Positive surprise (higher-than-expected) => bearish for duration/equities.
        Negative surprise (dovish) => bullish for risk assets.

        Returns
        -------
        pd.Series  Smoothed, z-scored policy surprise (inverted for risk-on framing)
        """
        surprise = actual_rate_decision - market_expected_rate
        # Accumulate surprises with decay
        accumulated = surprise.ewm(span=20, adjust=False).mean()
        # Invert: dovish surprise => positive signal
        signal = -self._z_score(accumulated)
        self._signal_history["policy_surprise"] = signal
        return signal

    # ------------------------------------------------------------------
    # Composite Score
    # ------------------------------------------------------------------

    def composite_signal(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Build composite macro signal from all computed sub-signals.

        Parameters
        ----------
        weights : dict, optional
            Weight for each sub-signal key. Defaults to equal weighting.

        Returns
        -------
        pd.Series  Composite macro signal in [-3, +3] range
        """
        if not self._signal_history:
            raise ValueError("No sub-signals computed. Run individual signal methods first.")

        keys = list(self._signal_history.keys())
        if weights is None:
            w = {k: 1.0 / len(keys) for k in keys}
        else:
            total = sum(weights.values())
            w = {k: v / total for k, v in weights.items()}

        # Align all signals on common index
        df = pd.concat(
            [self._signal_history[k].rename(k) for k in keys if k in self._signal_history],
            axis=1,
        ).dropna(how="all")

        composite = sum(df[k] * w.get(k, 0.0) for k in df.columns)
        composite = composite.clip(-self.config.signal_clip, self.config.signal_clip)
        return composite

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _z_score(self, series: pd.Series) -> pd.Series:
        """Rolling z-score normalization."""
        window = self.config.z_score_window
        mu = series.rolling(window, min_periods=window // 2).mean()
        sigma = series.rolling(window, min_periods=window // 2).std()
        z = (series - mu) / sigma.replace(0, np.nan)
        return z.clip(-self.config.signal_clip, self.config.signal_clip)

    def signal_summary(self) -> pd.DataFrame:
        """Return latest cross-section of all computed signals."""
        data = {
            k: v.dropna().iloc[-1] if not v.dropna().empty else np.nan
            for k, v in self._signal_history.items()
        }
        return pd.DataFrame.from_dict(
            data, orient="index", columns=["latest_signal"]
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    idx = pd.date_range("2015-01-01", periods=2000, freq="B")

    engine = MacroSignalEngine()

    # Synthetic data
    short_rate = pd.Series(np.random.normal(0.02, 0.005, len(idx)).cumsum() + 0.01, index=idx).clip(0)
    long_rate  = pd.Series(np.random.normal(0.04, 0.005, len(idx)).cumsum() + 0.02, index=idx).clip(0)
    pmi        = pd.Series(np.random.normal(51, 2, len(idx)), index=idx)
    inflation  = pd.Series(np.random.normal(0.022, 0.003, len(idx)), index=idx)
    credit     = pd.Series(np.random.normal(1e12, 5e10, len(idx)), index=idx)
    gdp        = pd.Series(np.full(len(idx), 25e12), index=idx)
    actual_cb  = pd.Series(np.random.choice([-0.25, 0, 0.25, 0.50], len(idx)), index=idx)
    expected_cb = pd.Series(np.zeros(len(idx)), index=idx)

    engine.yield_curve_signal(short_rate, long_rate)
    engine.real_yield_signal(long_rate, inflation)
    engine.pmi_signal(pmi)
    engine.credit_impulse_signal(credit, gdp)
    engine.policy_surprise_signal(actual_cb, expected_cb)

    comp = engine.composite_signal()
    print("Composite macro signal (last 5):\n", comp.tail())
    print("\nSignal summary:\n", engine.signal_summary())
