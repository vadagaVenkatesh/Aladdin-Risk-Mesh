"""
portfolio/rebalancer.py
Rebalancing Engine: threshold, calendar, and risk-triggered rebalance
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RebalanceConfig:
    rebalance_type: str  = "threshold"  # "threshold" | "calendar" | "risk"
    drift_threshold: float = 0.05       # 5% weight drift triggers rebalance
    calendar_freq: str    = "M"         # pandas offset alias
    min_trade_size: float = 0.001       # ignore tiny trades
    transaction_cost: float = 0.0010    # 10 bps per unit of turnover
    max_turnover: float   = 0.30        # 30% one-way turnover cap
    var_trigger_multiplier: float = 1.5 # rebalance if VaR > 1.5x limit


class Rebalancer:
    """
    Portfolio rebalancing engine.

    Supports
    --------
    - Threshold (band) rebalancing
    - Calendar rebalancing
    - Risk-triggered (VaR breach) rebalancing
    - Turnover-constrained trade generation
    - Cost-adjusted net benefit analysis
    """

    def __init__(self, config: Optional[RebalanceConfig] = None):
        self.config = config or RebalanceConfig()

    # ------------------------------------------------------------------
    # Core: compute trades from current to target weights
    # ------------------------------------------------------------------
    def compute_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> pd.Series:
        """
        Net trade vector = target - current.
        Zeroes out trades below min_trade_size.
        """
        trades = target_weights.subtract(current_weights, fill_value=0.0)
        trades = trades[trades.abs() >= self.config.min_trade_size]
        return trades

    # ------------------------------------------------------------------
    # Turnover cap
    # ------------------------------------------------------------------
    def cap_turnover(
        self,
        trades: pd.Series,
    ) -> pd.Series:
        """
        Scale trades proportionally if one-way turnover exceeds cap.
        """
        one_way = trades.abs().sum() / 2.0
        if one_way > self.config.max_turnover:
            scale  = self.config.max_turnover / one_way
            trades = trades * scale
        return trades

    # ------------------------------------------------------------------
    # Threshold check
    # ------------------------------------------------------------------
    def needs_rebalance_threshold(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> bool:
        """
        Returns True if any weight drifted beyond drift_threshold.
        """
        drift = (current_weights - target_weights).abs()
        return bool(drift.max() >= self.config.drift_threshold)

    # ------------------------------------------------------------------
    # Calendar rebalance schedule
    # ------------------------------------------------------------------
    def rebalance_dates(
        self,
        start: str,
        end: str,
    ) -> pd.DatetimeIndex:
        """
        Generate calendar rebalance dates.
        """
        return pd.date_range(start, end, freq=self.config.calendar_freq)

    # ------------------------------------------------------------------
    # Risk-triggered rebalance
    # ------------------------------------------------------------------
    def needs_rebalance_risk(
        self,
        current_var: float,
        var_limit: float,
    ) -> bool:
        """
        Trigger rebalance if current VaR exceeds limit * multiplier.
        """
        return current_var > var_limit * self.config.var_trigger_multiplier

    # ------------------------------------------------------------------
    # Cost-adjusted benefit
    # ------------------------------------------------------------------
    def net_benefit(
        self,
        current_sharpe: float,
        target_sharpe: float,
        trades: pd.Series,
    ) -> float:
        """
        Estimate net benefit of rebalancing after transaction costs.
        Returns improvement in Sharpe (approximate, annualised).
        """
        cost = trades.abs().sum() * self.config.transaction_cost
        improvement = target_sharpe - current_sharpe
        return improvement - cost * 252  # annualise cost

    # ------------------------------------------------------------------
    # Full rebalance step
    # ------------------------------------------------------------------
    def rebalance_step(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        date: Optional[pd.Timestamp] = None,
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
        current_var: float = 0.0,
        var_limit: float = 0.02,
    ) -> Tuple[bool, pd.Series, float]:
        """
        Decide whether to rebalance and compute trade list.

        Returns (do_rebalance, trades, estimated_cost)
        """
        do_rebalance = False

        if self.config.rebalance_type == "threshold":
            do_rebalance = self.needs_rebalance_threshold(
                current_weights, target_weights)

        elif self.config.rebalance_type == "calendar":
            if date is not None and rebalance_dates is not None:
                do_rebalance = date in rebalance_dates

        elif self.config.rebalance_type == "risk":
            do_rebalance = self.needs_rebalance_risk(current_var, var_limit)

        if not do_rebalance:
            return False, pd.Series(dtype=float), 0.0

        trades = self.compute_trades(current_weights, target_weights)
        trades = self.cap_turnover(trades)
        cost   = trades.abs().sum() * self.config.transaction_cost
        return True, trades, cost

    # ------------------------------------------------------------------
    # Backtest helper: rolling rebalance over time series
    # ------------------------------------------------------------------
    def simulate_rebalancing(
        self,
        returns: pd.DataFrame,
        initial_weights: pd.Series,
        target_weights_ts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate portfolio with periodic rebalancing.
        Returns DataFrame with columns: [date, portfolio_return, turnover, cost]
        """
        weights = initial_weights.copy()
        records = []

        for date in returns.index:
            # drift weights with returns
            daily_ret = returns.loc[date]
            port_ret  = (weights * daily_ret).sum()

            # update weights for price impact
            new_w = weights * (1 + daily_ret)
            new_w = new_w / new_w.sum()

            # check if target provided
            target = target_weights_ts.loc[date] if date in target_weights_ts.index else weights

            rebaled, trades, cost = self.rebalance_step(new_w, target)
            if rebaled:
                new_w = target.copy()
                turnover = trades.abs().sum() / 2
            else:
                turnover = 0.0
                cost     = 0.0

            weights = new_w
            records.append({
                "date":             date,
                "portfolio_return": port_ret,
                "turnover":         turnover,
                "cost":             cost,
            })

        return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(99)
    tickers = ["SPY", "TLT", "EEM", "GLD"]
    current = pd.Series([0.30, 0.20, 0.28, 0.22], index=tickers)
    target  = pd.Series([0.25, 0.25, 0.25, 0.25], index=tickers)

    rb = Rebalancer(RebalanceConfig(drift_threshold=0.03))
    do_reb, trades, cost = rb.rebalance_step(current, target)
    print(f"Rebalance: {do_reb}")
    print(f"Trades:\n{trades}")
    print(f"Estimated cost: {cost:.4f}")
