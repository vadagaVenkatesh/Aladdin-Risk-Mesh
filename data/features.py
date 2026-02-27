"""
data/features.py
Aladdin Risk Mesh - Feature Engineering & Signal Preprocessing

Authors: VDG Venkatesh & Agentic AI
Description:
    Computes a comprehensive feature set from raw price/return series:
    - Rolling returns (1m, 3m, 6m, 12m)
    - Z-score normalization (cross-sectional and time-series)
    - Volatility estimates (rolling std, EWMA)
    - Cross-sectional ranks and percentile scores
    - Macro regime features (yield curve slope, credit spreads)
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Transforms raw price / return data into a rich feature matrix
    suitable for signal generation and ML models.
    """

    MOMENTUM_WINDOWS = [21, 63, 126, 252]   # ~1m, 3m, 6m, 12m in trading days
    VOL_WINDOWS      = [21, 63]              # ~1m, 3m
    EWMA_SPAN        = 30

    def __init__(self, prices: pd.DataFrame):
        """
        Args:
            prices: Wide DataFrame (dates x tickers) of adjusted close prices.
        """
        self.prices = prices.copy()
        self.log_returns = np.log(prices / prices.shift(1))
        self.simple_returns = prices.pct_change()

    # ------------------------------------------------------------------
    # Momentum Features
    # ------------------------------------------------------------------

    def rolling_returns(self) -> pd.DataFrame:
        """
        Compute trailing total log returns over each momentum window.
        Returns a MultiLevel column DataFrame: (window, ticker).
        """
        frames = {}
        for w in self.MOMENTUM_WINDOWS:
            frames[f'ret_{w}d'] = self.log_returns.rolling(w).sum()
        return pd.concat(frames, axis=1)

    def momentum_zscore(self, window: int = 252) -> pd.DataFrame:
        """
        Cross-sectional z-score of trailing returns over given window.
        Normalises signals so they are comparable across assets.
        """
        trailing = self.log_returns.rolling(window).sum()
        mu  = trailing.mean(axis=1)
        std = trailing.std(axis=1)
        return trailing.sub(mu, axis=0).div(std, axis=0)

    # ------------------------------------------------------------------
    # Volatility Features
    # ------------------------------------------------------------------

    def rolling_volatility(self) -> pd.DataFrame:
        """
        Annualised rolling volatility for each window in VOL_WINDOWS.
        """
        frames = {}
        for w in self.VOL_WINDOWS:
            frames[f'vol_{w}d'] = self.log_returns.rolling(w).std() * np.sqrt(252)
        return pd.concat(frames, axis=1)

    def ewma_volatility(self) -> pd.DataFrame:
        """
        EWMA volatility (annualised) using span = EWMA_SPAN.
        Responds faster to recent volatility spikes.
        """
        return (self.log_returns
                    .ewm(span=self.EWMA_SPAN)
                    .std() * np.sqrt(252))

    # ------------------------------------------------------------------
    # Cross-Sectional Features
    # ------------------------------------------------------------------

    def cross_sectional_rank(self, df: Optional[pd.DataFrame] = None,
                             window: int = 252) -> pd.DataFrame:
        """
        Cross-sectional percentile rank [0, 1] of rolling returns.
        Used by learn-to-rank models and relative-value signals.
        """
        if df is None:
            df = self.log_returns.rolling(window).sum()
        return df.rank(axis=1, pct=True)

    # ------------------------------------------------------------------
    # Macro/Rate Features
    # ------------------------------------------------------------------

    @staticmethod
    def yield_curve_slope(short_yield: pd.Series,
                          long_yield: pd.Series) -> pd.Series:
        """
        10Y - 3M yield spread (or any long-short pair).
        Positive = normal curve; negative = inversion (recession signal).
        """
        return long_yield - short_yield

    @staticmethod
    def credit_spread(hy_price: pd.Series,
                      ig_price: pd.Series) -> pd.Series:
        """
        HY - IG spread proxy from total return price series.
        Widens during risk-off; narrows in risk-on regimes.
        """
        hy_ret = np.log(hy_price / hy_price.shift(1))
        ig_ret = np.log(ig_price / ig_price.shift(1))
        return (hy_ret - ig_ret).rolling(21).sum()

    # ------------------------------------------------------------------
    # Composite Feature Matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(self) -> pd.DataFrame:
        """
        Assemble a flat feature matrix combining momentum, volatility,
        and cross-sectional rank features.
        Returns:
            DataFrame with MultiIndex columns: (feature_group, ticker).
        """
        rets    = self.rolling_returns()
        vols    = self.rolling_volatility()
        ewma_v  = self.ewma_volatility()
        ranks   = self.cross_sectional_rank()

        ewma_v  = pd.concat({'vol_ewma': ewma_v}, axis=1)
        ranks   = pd.concat({'cs_rank': ranks},   axis=1)

        feature_matrix = pd.concat([rets, vols, ewma_v, ranks], axis=1)
        feature_matrix = feature_matrix.sort_index()
        return feature_matrix


if __name__ == '__main__':
    from data.loaders import MultiAssetLoader
    loader  = MultiAssetLoader(start='2018-01-01', end='2024-12-31')
    prices  = loader.load_equity()
    fe      = FeatureEngineer(prices)
    fm      = fe.build_feature_matrix()
    print(f"Feature matrix shape: {fm.shape}")
    print(fm.columns.get_level_values(0).unique())
