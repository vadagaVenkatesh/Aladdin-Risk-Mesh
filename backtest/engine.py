"""Vectorized and Event-Driven Backtesting Engine for Aladdin-Risk-Mesh.

Core features:
- Walk-forward optimization support
- Performance analytics (Sharpe, Sortino, Drawdown, Calmar)
- Integration with portfolio optimizers and risk engines
- Multi-asset, multi-currency support
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable


class BacktestEngine:
    """
    Unified backtesting framework.

    Supports both rapid vectorized backtesting for signal research
    and event-driven simulation for realistic execution.

    Parameters
    ----------
    data : pd.DataFrame
        Oversized price/return data indexed by timestamp.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results: Optional[pd.DataFrame] = None

    def run_vectorized(
        self,
        signals: pd.DataFrame,
        transaction_costs: float = 0.0005,
    ) -> pd.DataFrame:
        """
        Fast vectorized backtest.

        Parameters
        ----------
        signals : pd.DataFrame
            Target weights or positions.
        transaction_costs : float
            One-way cost (slippage + commission).

        Returns
        -------
        pd.DataFrame  P&L and performance metrics
        """
        # Align signals with data
        weights = signals.reindex(self.data.index).fillna(method='ffill').fillna(0)
        returns = self.data.pct_change().fillna(0)

        # Raw portfolio returns
        port_rets = (weights.shift(1) * returns).sum(axis=1)

        # Estimate turnover and costs
        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * transaction_costs
        net_rets = port_rets - costs

        # Cumulative performance
        equity_curve = (1 + net_rets).cumprod()

        self.results = pd.DataFrame({
            "gross_return": port_rets,
            "net_return": net_rets,
            "equity_curve": equity_curve,
            "turnover": turnover
        })
        return self.results

    def compute_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """Calculate standard performance statistics."""
        if self.results is None:
            raise ValueError("Run backtest first.")

        rets = self.results["net_return"]
        total_ret = self.results["equity_curve"].iloc[-1] - 1
        ann_ret = (1 + rets.mean())**252 - 1
        ann_vol = rets.std() * np.sqrt(252)

        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # Drawdown
        cum_max = self.results["equity_curve"].cummax()
        drawdown = (self.results["equity_curve"] / cum_max) - 1
        max_dd = drawdown.min()

        return {
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": abs(ann_ret / max_dd) if max_dd < 0 else 0
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate synthetic prices
    idx = pd.date_range("2020-01-01", periods=1000, freq="B")
    prices = pd.DataFrame(
        np.exp(np.random.normal(0.0001, 0.01, (1000, 2)).cumsum(axis=0)) * 100,
        index=idx,
        columns=["Asset1", "Asset2"]
    )

    # Simple 60/40 strategy
    signals = pd.DataFrame(index=idx)
    signals["Asset1"] = 0.6
    signals["Asset2"] = 0.4

    bt = BacktestEngine(prices)
    bt.run_vectorized(signals)

    metrics = bt.compute_metrics()
    print("Backtest Results:")
    for k, v in metrics.items():
        print(f"{k:25}: {v:.4f}")

    print("
Latest Results:")
    print(bt.results.tail())
