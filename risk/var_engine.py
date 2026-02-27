"""
risk/var_engine.py
Aladdin Risk Mesh - VaR and Expected Shortfall Engine

Authors: VDG Venkatesh & Agentic AI
Description:
    Computes Value-at-Risk (VaR) and Expected Shortfall (ES/CVaR)
    using three methodologies:
    1. Historical Simulation (non-parametric, preferred for fat tails)
    2. Parametric (Gaussian assumption, fast intraday approximation)
    3. Monte Carlo Simulation (draws from historical covariance)

    Supports 1-day to 10-day horizons via square-root-of-time scaling.
    Multiple lookback windows capture both recent and long-run regimes.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


class VaREngine:
    """
    Cross-asset VaR and Expected Shortfall engine.

    Supports Historical Simulation, Parametric, and Monte Carlo methods.
    Designed for a macro long/short portfolio with equity, bond, FX,
    and commodity exposures.
    """

    DEFAULT_CONFIDENCE = 0.95    # 95% VaR
    DEFAULT_HORIZON    = 1       # 1-day horizon
    DEFAULT_LOOKBACK   = 252     # 1 year of daily returns
    MC_SIMULATIONS     = 10_000  # Monte Carlo paths

    def __init__(
        self,
        returns: pd.DataFrame,
        confidence: float = DEFAULT_CONFIDENCE,
        horizon: int = DEFAULT_HORIZON,
        lookback: int = DEFAULT_LOOKBACK
    ):
        """
        Args:
            returns    : DataFrame of daily log returns (dates x assets).
            confidence : VaR confidence level (e.g. 0.95 for 95% VaR).
            horizon    : Holding period in trading days.
            lookback   : Number of historical days to use.
        """
        self.returns    = returns.dropna(how='all')
        self.confidence = confidence
        self.horizon    = horizon
        self.lookback   = lookback
        self.alpha      = 1 - confidence   # tail probability

    def _recent_returns(self, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Get the most recent `lookback` rows of returns."""
        r = self.returns if as_of is None else self.returns.loc[:as_of]
        return r.iloc[-self.lookback:]

    # ------------------------------------------------------------------
    # Portfolio P&L Series
    # ------------------------------------------------------------------

    def portfolio_pnl(
        self,
        weights: pd.Series,
        as_of: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Compute daily portfolio P&L series given asset weights.
        Args:
            weights : Series of asset weights (index = tickers).
        Returns:
            Daily portfolio return series.
        """
        r    = self._recent_returns(as_of)
        cols = [c for c in weights.index if c in r.columns]
        w    = weights[cols]
        w    = w / w.abs().sum()   # normalise to gross=1
        return (r[cols] * w).sum(axis=1)

    # ------------------------------------------------------------------
    # Historical Simulation VaR / ES
    # ------------------------------------------------------------------

    def historical_var(
        self,
        weights: pd.Series,
        as_of: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Historical Simulation VaR:
            VaR = -quantile(portfolio_pnl, alpha)

        Scaled to `horizon`-day horizon via sqrt-of-time rule.
        Returns positive number (loss magnitude).
        """
        pnl  = self.portfolio_pnl(weights, as_of)
        var1 = -np.percentile(pnl.dropna(), self.alpha * 100)
        return var1 * np.sqrt(self.horizon)

    def historical_es(
        self,
        weights: pd.Series,
        as_of: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Historical Expected Shortfall (CVaR):
            ES = -mean(portfolio_pnl | pnl < VaR_threshold)

        Returns positive number (expected loss beyond VaR).
        """
        pnl       = self.portfolio_pnl(weights, as_of).dropna()
        threshold = np.percentile(pnl, self.alpha * 100)
        tail_pnl  = pnl[pnl <= threshold]
        es        = -tail_pnl.mean()
        return es * np.sqrt(self.horizon)

    # ------------------------------------------------------------------
    # Parametric VaR / ES (Gaussian)
    # ------------------------------------------------------------------

    def parametric_var(
        self,
        weights: pd.Series,
        as_of: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Parametric VaR assuming normally distributed returns:
            VaR = -( mu - z_alpha * sigma ) * sqrt(horizon)
        """
        pnl   = self.portfolio_pnl(weights, as_of).dropna()
        mu    = pnl.mean()
        sigma = pnl.std()
        z     = stats.norm.ppf(self.alpha)
        var1  = -(mu + z * sigma)
        return var1 * np.sqrt(self.horizon)

    def parametric_es(
        self,
        weights: pd.Series,
        as_of: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Parametric ES (Gaussian):
            ES = -(mu - sigma * phi(z_alpha) / alpha)
        where phi is the standard normal PDF.
        """
        pnl   = self.portfolio_pnl(weights, as_of).dropna()
        mu    = pnl.mean()
        sigma = pnl.std()
        z     = stats.norm.ppf(self.alpha)
        phi_z = stats.norm.pdf(z)
        es    = -(mu - sigma * phi_z / self.alpha)
        return es * np.sqrt(self.horizon)

    # ------------------------------------------------------------------
    # Monte Carlo VaR / ES
    # ------------------------------------------------------------------

    def monte_carlo_var(
        self,
        weights: pd.Series,
        as_of: Optional[pd.Timestamp] = None,
        n_sims: int = MC_SIMULATIONS
    ) -> float:
        """
        Monte Carlo VaR using multivariate normal draws from
        historical mean and covariance.
        """
        r    = self._recent_returns(as_of).dropna()
        cols = [c for c in weights.index if c in r.columns]
        w    = (weights[cols] / weights[cols].abs().sum()).values
        mu   = r[cols].mean().values
        cov  = r[cols].cov().values
        sims = np.random.multivariate_normal(mu, cov, n_sims)
        port = sims @ w
        var1 = -np.percentile(port, self.alpha * 100)
        return var1 * np.sqrt(self.horizon)

    # ------------------------------------------------------------------
    # Rolling VaR Timeseries
    # ------------------------------------------------------------------

    def rolling_var_series(
        self,
        weights: pd.Series,
        method: str = 'historical'
    ) -> pd.Series:
        """
        Compute rolling VaR over the full return history.
        Useful for backtesting and regime analysis.

        Args:
            weights : Asset weights.
            method  : 'historical' or 'parametric'.
        Returns:
            Series of daily rolling VaR estimates.
        """
        dates  = self.returns.index[self.lookback:]
        result = {}
        for date in dates:
            tmp_engine = VaREngine(
                self.returns.loc[:date],
                confidence=self.confidence,
                horizon=self.horizon,
                lookback=self.lookback
            )
            if method == 'historical':
                result[date] = tmp_engine.historical_var(weights)
            else:
                result[date] = tmp_engine.parametric_var(weights)
        return pd.Series(result, name=f'var_{int(self.confidence*100)}')

    # ------------------------------------------------------------------
    # Summary Report
    # ------------------------------------------------------------------

    def var_report(
        self,
        weights: pd.Series,
        as_of: Optional[pd.Timestamp] = None
    ) -> dict:
        """
        Produce a comprehensive VaR/ES summary across all methods.
        Returns:
            dict with keys: hist_var, hist_es, param_var, param_es, mc_var.
        """
        return {
            'hist_var':   self.historical_var(weights, as_of),
            'hist_es':    self.historical_es(weights, as_of),
            'param_var':  self.parametric_var(weights, as_of),
            'param_es':   self.parametric_es(weights, as_of),
            'mc_var':     self.monte_carlo_var(weights, as_of),
            'confidence': self.confidence,
            'horizon':    self.horizon,
            'lookback':   self.lookback,
        }


if __name__ == '__main__':
    from data.loaders import MultiAssetLoader
    loader  = MultiAssetLoader(start='2015-01-01', end='2025-12-31')
    prices  = loader.load_equity()
    returns = np.log(prices / prices.shift(1)).dropna()
    weights = pd.Series({
        'SPY':  0.30, 'QQQ':  0.15, 'EFA':  0.15,
        'EEM':  0.10, 'VGK':  0.10, 'EWJ':  0.10,
        'IWM': -0.10, 'VNQ': -0.10,
    })
    engine  = VaREngine(returns, confidence=0.95, horizon=1)
    report  = engine.var_report(weights)
    print('VaR Report:')
    for k, v in report.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
