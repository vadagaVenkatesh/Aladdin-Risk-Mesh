"""
signals/momentum.py
Aladdin Risk Mesh - Time-Series Momentum (TSMOM) Signals

Authors: VDG Venkatesh & Agentic AI
Description:
    Implements time-series momentum signals across multiple lookback
    windows. TSMOM has historically performed well during inflationary
    and crisis regimes, making it a core signal for 2026 macro investing.

    Signal construction:
        signal_t = sign(r_{t-L, t-1}) * vol_scaled_weight
    where L is the lookback window and vol_scaled weights by inverse
    recent volatility (risk parity at signal level).
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class TimeSeriesMomentum:
    """
    Time-series momentum (TSMOM) signal generator.

    Generates long/short signals based on trailing price performance
    across configurable lookback windows, with volatility scaling.
    Lookbacks: 1m (21d), 3m (63d), 6m (126d), 12m (252d).
    """

    DEFAULT_LOOKBACKS  = [21, 63, 126, 252]   # trading days
    VOL_LOOKBACK       = 60                    # days for vol scaling
    VOL_TARGET         = 0.15                  # 15% annualised vol target

    def __init__(self,
                 prices: pd.DataFrame,
                 lookbacks: Optional[List[int]] = None,
                 vol_scale: bool = True):
        """
        Args:
            prices    : DataFrame of adjusted close prices (dates x tickers).
            lookbacks : List of lookback windows in trading days.
            vol_scale : Whether to apply inverse-vol position scaling.
        """
        self.prices    = prices.copy()
        self.lookbacks = lookbacks or self.DEFAULT_LOOKBACKS
        self.vol_scale = vol_scale
        self.log_ret   = np.log(prices / prices.shift(1))

    def _vol_scale_factor(self) -> pd.DataFrame:
        """
        Compute daily volatility scaling factor: vol_target / realised_vol.
        Caps scaling at 2.0 to prevent extreme leverage.
        """
        daily_vol  = self.log_ret.rolling(self.VOL_LOOKBACK).std()
        annual_vol = daily_vol * np.sqrt(252)
        factor     = (self.VOL_TARGET / annual_vol).clip(upper=2.0)
        return factor

    def raw_signal(self, lookback: int) -> pd.DataFrame:
        """
        Compute sign-of-return signal for a given lookback window.
        +1 if trailing return is positive (long), -1 if negative (short).

        Args:
            lookback: Number of trading days.
        Returns:
            DataFrame of {-1, 0, +1} signals.
        """
        trailing_ret = self.log_ret.rolling(lookback).sum()
        signal = np.sign(trailing_ret)
        return signal

    def scaled_signal(self, lookback: int) -> pd.DataFrame:
        """
        Vol-scaled TSMOM signal.
        Scales raw ±1 signal by inverse realised volatility
        so each position targets vol_target annualised vol.
        """
        sig = self.raw_signal(lookback)
        if self.vol_scale:
            sig = sig * self._vol_scale_factor()
        return sig

    def composite_signal(self,
                         weights: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Blend signals across all lookback windows into a single composite.
        Equal-weights by default; can pass custom weights.

        Args:
            weights: List of weights (must sum to 1) aligned with lookbacks.
        Returns:
            Composite momentum signal DataFrame.
        """
        if weights is None:
            weights = [1.0 / len(self.lookbacks)] * len(self.lookbacks)
        assert len(weights) == len(self.lookbacks), "Weight-lookback mismatch."
        composite = pd.DataFrame(0.0,
                                 index=self.prices.index,
                                 columns=self.prices.columns)
        for w, lb in zip(weights, self.lookbacks):
            composite += w * self.scaled_signal(lb)
        return composite

    def signal_summary(self) -> pd.DataFrame:
        """
        Return summary DataFrame of raw signal values for all lookbacks.
        Useful for diagnostics and regime analysis.
        """
        summary = {}
        for lb in self.lookbacks:
            summary[f'tsmom_{lb}d'] = self.raw_signal(lb)
        return pd.concat(summary, axis=1)

    def sharpe_by_window(self) -> pd.Series:
        """
        Estimate annualised Sharpe ratio of the TSMOM strategy per window.
        Uses next-day returns as the forward return.
        """
        fwd_ret = self.log_ret.shift(-1)  # next day return
        sharpes = {}
        for lb in self.lookbacks:
            sig      = self.scaled_signal(lb)
            strat_r  = (sig * fwd_ret).dropna()
            ann_ret  = strat_r.mean().mean() * 252
            ann_vol  = strat_r.std().mean()  * np.sqrt(252)
            sharpes[f'tsmom_{lb}d'] = ann_ret / ann_vol if ann_vol > 0 else 0.0
        return pd.Series(sharpes)


if __name__ == '__main__':
    from data.loaders import MultiAssetLoader
    loader = MultiAssetLoader(start='2015-01-01', end='2025-12-31')
    prices = loader.get_universe_prices()
    mom    = TimeSeriesMomentum(prices)
    comp   = mom.composite_signal()
    print('Composite TSMOM signal (last 5 rows):')
    print(comp.tail())
    print('\nSharpe by window:')
    print(mom.sharpe_by_window())
