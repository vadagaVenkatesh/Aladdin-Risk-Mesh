"""
signals/carry.py
Aladdin Risk Mesh - Carry & Roll-Down Signals

Authors: VDG Venkatesh & Agentic AI
Description:
    Implements carry signals across three asset classes:
    1. Interest rate carry: short-end vs long-end yield differential
    2. FX carry: interest rate parity implied by short rates
    3. Commodity roll yield: contango vs backwardation

    Carry is a risk premium that captures compensation for holding
    risk assets over the risk-free rate. It performs well in stable
    inflationary growth regimes but requires careful risk management.
"""

import pandas as pd
import numpy as np
from typing import Optional


class CarrySignal:
    """
    Multi-asset carry signal generator.

    Carry signals represent the expected return from holding an
    asset assuming no price change, i.e., the "carry" of the position.
    """

    VOL_LOOKBACK  = 63   # 3-month vol for normalisation
    VOL_TARGET    = 0.10 # 10% target vol per carry strategy

    def __init__(self, prices: pd.DataFrame):
        """
        Args:
            prices: Wide DataFrame of adjusted close prices.
        """
        self.prices  = prices.copy()
        self.log_ret = np.log(prices / prices.shift(1))

    # ------------------------------------------------------------------
    # Rate Carry
    # ------------------------------------------------------------------

    def rate_carry(
        self,
        short_end_tickers: list,
        long_end_tickers: list
    ) -> pd.Series:
        """
        Compute rate carry as the difference between long-end and
        short-end bond ETF trailing returns (3m). Positive carry =
        long bond outperforms cash, implying term premium.

        Args:
            short_end_tickers : Tickers of short-duration bonds (e.g. SHY).
            long_end_tickers  : Tickers of long-duration bonds (e.g. TLT).
        Returns:
            Daily series of rate carry signal (long minus short).
        """
        short_ret = self.log_ret[short_end_tickers].mean(axis=1).rolling(63).sum()
        long_ret  = self.log_ret[long_end_tickers].mean(axis=1).rolling(63).sum()
        carry     = long_ret - short_ret
        return self._vol_normalise(carry.to_frame('rate_carry'))['rate_carry']

    # ------------------------------------------------------------------
    # FX Carry
    # ------------------------------------------------------------------

    def fx_carry(
        self,
        fx_tickers: list,
        high_yield_tickers: list,
        low_yield_tickers: list
    ) -> pd.DataFrame:
        """
        FX carry signal: long high-yielding, short low-yielding currencies.
        Proxied as total-return differences between high vs low yield FX.

        Args:
            fx_tickers         : All FX pair tickers.
            high_yield_tickers : High carry currencies (e.g. AUD, BRL).
            low_yield_tickers  : Low carry currencies (e.g. JPY, CHF).
        Returns:
            Long/short FX carry signal DataFrame.
        """
        available_hy = [t for t in high_yield_tickers if t in self.prices.columns]
        available_ly = [t for t in low_yield_tickers  if t in self.prices.columns]

        if not available_hy or not available_ly:
            return pd.DataFrame(index=self.prices.index)

        hy_ret = self.log_ret[available_hy].rolling(21).sum()
        ly_ret = self.log_ret[available_ly].rolling(21).sum()

        carry = hy_ret.mean(axis=1) - ly_ret.mean(axis=1)
        signal = pd.DataFrame({'fx_carry': carry})
        return self._vol_normalise(signal)

    # ------------------------------------------------------------------
    # Commodity Roll Yield (Carry)
    # ------------------------------------------------------------------

    def commodity_roll_yield(
        self,
        spot_tickers: list,
        futures_tickers: list
    ) -> pd.DataFrame:
        """
        Commodity carry from roll yield:
        - Backwardation (spot > futures) => positive carry, go long.
        - Contango      (spot < futures) => negative carry, go short.

        Proxied here as the z-score of trailing 3m price momentum
        relative to the commodity's own vol (since we don't have
        direct front/back contract data in yfinance).

        Args:
            spot_tickers    : Spot ETF tickers (e.g. GLD, USO).
            futures_tickers : Futures tickers  (e.g. GC=F, CL=F).
        Returns:
            Commodity carry signal DataFrame.
        """
        signals = {}
        pairs = list(zip(spot_tickers, futures_tickers))
        for spot, fut in pairs:
            if spot not in self.prices.columns or fut not in self.prices.columns:
                continue
            spot_ret  = self.log_ret[spot].rolling(63).sum()
            fut_ret   = self.log_ret[fut].rolling(63).sum()
            roll_yield = spot_ret - fut_ret
            mu  = roll_yield.rolling(252).mean()
            std = roll_yield.rolling(252).std()
            signals[f'comm_carry_{spot}'] = (roll_yield - mu) / (std + 1e-8)
        return pd.DataFrame(signals, index=self.prices.index)

    # ------------------------------------------------------------------
    # Composite Carry Signal
    # ------------------------------------------------------------------

    def composite_carry(
        self,
        rate_carry: Optional[pd.Series] = None,
        fx_carry: Optional[pd.DataFrame] = None,
        comm_carry: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Assemble a single composite carry DataFrame combining
        rate, FX, and commodity carry signals.
        """
        frames = {}
        if rate_carry is not None:
            frames['rate_carry'] = rate_carry
        if fx_carry is not None and not fx_carry.empty:
            for col in fx_carry.columns:
                frames[col] = fx_carry[col]
        if comm_carry is not None and not comm_carry.empty:
            for col in comm_carry.columns:
                frames[col] = comm_carry[col]
        return pd.DataFrame(frames, index=self.prices.index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vol_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise signals to target annualised vol."""
        daily_vol  = df.rolling(self.VOL_LOOKBACK).std()
        annual_vol = daily_vol * np.sqrt(252)
        return df * (self.VOL_TARGET / annual_vol.replace(0, np.nan))


if __name__ == '__main__':
    from data.loaders import MultiAssetLoader
    loader = MultiAssetLoader(start='2015-01-01', end='2025-12-31')
    prices = loader.get_universe_prices()
    carry  = CarrySignal(prices)
    rc     = carry.rate_carry(['SHY'], ['TLT'])
    print('Rate carry signal tail:')
    print(rc.tail())
