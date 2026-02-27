"""
risk/stress_test.py
Historical & Hypothetical Stress Testing Engine
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StressScenario:
    name: str
    shocks: Dict[str, float]          # asset -> shock magnitude (e.g. -0.20)
    description: str = ""
    correlation_override: Optional[np.ndarray] = None


HISTORICAL_SCENARIOS: List[StressScenario] = [
    StressScenario(
        name="GFC_2008",
        shocks={"equity": -0.40, "credit": -0.25, "em": -0.45,
                "commodities": -0.35, "rates": 0.05},
        description="Global Financial Crisis peak-to-trough drawdowns",
    ),
    StressScenario(
        name="COVID_2020",
        shocks={"equity": -0.34, "credit": -0.15, "em": -0.28,
                "commodities": -0.40, "rates": 0.08},
        description="COVID-19 March 2020 shock",
    ),
    StressScenario(
        name="EUR_CRISIS_2011",
        shocks={"equity": -0.20, "credit": -0.18, "em": -0.15,
                "commodities": -0.10, "rates": -0.02},
        description="European Sovereign Debt Crisis",
    ),
    StressScenario(
        name="RATE_SHOCK_2022",
        shocks={"equity": -0.25, "credit": -0.20, "em": -0.22,
                "commodities": 0.15, "rates": -0.15},
        description="Fed hiking cycle 2022 bond drawdown",
    ),
    StressScenario(
        name="OIL_SHOCK_1990",
        shocks={"equity": -0.15, "credit": -0.05, "em": -0.10,
                "commodities": 0.60, "rates": -0.03},
        description="Gulf War oil price spike",
    ),
]


class StressTester:
    """
    Apply historical and hypothetical stress scenarios to a portfolio.
    Supports:
    - Single-scenario P&L estimation
    - Batch scenario sweep
    - Factor-sensitivity stress (parallel-shift, twist, tilt)
    - Reverse stress test: find shock that breaches a loss threshold
    """

    def __init__(
        self,
        weights: pd.Series,
        cov: pd.DataFrame,
        asset_map: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        weights   : portfolio weights indexed by ticker
        cov       : covariance matrix of returns (annualised)
        asset_map : maps ticker -> asset class (equity/credit/em/commodities/rates)
        """
        self.weights   = weights
        self.cov       = cov
        self.asset_map = asset_map or {}

    # ------------------------------------------------------------------
    # Core scenario P&L
    # ------------------------------------------------------------------
    def scenario_pnl(
        self,
        scenario: StressScenario,
    ) -> Tuple[float, pd.Series]:
        """
        Returns (total_pnl, per_asset_pnl) under a stress scenario.
        Assets not in the shock dict receive 0 shock.
        """
        shocks = pd.Series(0.0, index=self.weights.index)
        for ticker, w in self.weights.items():
            asset_class = self.asset_map.get(ticker, "equity")
            shocks[ticker] = scenario.shocks.get(asset_class, 0.0)

        per_asset = self.weights * shocks
        total     = per_asset.sum()
        return total, per_asset

    # ------------------------------------------------------------------
    # Batch sweep
    # ------------------------------------------------------------------
    def run_all_scenarios(
        self,
        scenarios: Optional[List[StressScenario]] = None,
    ) -> pd.DataFrame:
        """
        Run all scenarios and return a summary DataFrame.
        """
        scenarios = scenarios or HISTORICAL_SCENARIOS
        rows = []
        for sc in scenarios:
            total, per_asset = self.scenario_pnl(sc)
            rows.append({
                "scenario":    sc.name,
                "description": sc.description,
                "total_pnl":   round(total, 4),
            })
        return pd.DataFrame(rows).set_index("scenario")

    # ------------------------------------------------------------------
    # Factor sensitivity stress
    # ------------------------------------------------------------------
    def factor_stress(
        self,
        factor_shocks: Dict[str, float],
        factor_loadings: pd.DataFrame,
    ) -> Tuple[float, pd.Series]:
        """
        P&L given factor shocks and a loadings matrix.
        factor_loadings: DataFrame(index=tickers, columns=factors)
        factor_shocks  : dict {factor: shock_size}
        """
        shock_vec = pd.Series(factor_shocks)
        asset_shocks = factor_loadings[shock_vec.index].dot(shock_vec)
        per_asset    = self.weights * asset_shocks
        return per_asset.sum(), per_asset

    # ------------------------------------------------------------------
    # Reverse stress test
    # ------------------------------------------------------------------
    def reverse_stress(
        self,
        loss_threshold: float,
        asset_class: str = "equity",
        step: float = 0.01,
        max_shock: float = 0.80,
    ) -> float:
        """
        Find minimum uniform shock to the given asset class that
        breaches the loss threshold.
        Returns the shock level (absolute value).
        """
        shock = 0.0
        while shock <= max_shock:
            shock += step
            sc = StressScenario(
                name="reverse",
                shocks={asset_class: -shock},
            )
            total, _ = self.scenario_pnl(sc)
            if total <= -abs(loss_threshold):
                return round(shock, 4)
        return max_shock

    # ------------------------------------------------------------------
    # Marginal VaR contribution under stress
    # ------------------------------------------------------------------
    def stressed_var(
        self,
        scenario: StressScenario,
        confidence: float = 0.99,
    ) -> float:
        """
        Simple parametric VaR scaled by the scenario shock magnitude.
        """
        w   = self.weights.values
        sig = np.sqrt(w @ self.cov.values @ w)
        from scipy.stats import norm
        base_var = sig * norm.ppf(confidence)
        avg_shock = np.mean(list(scenario.shocks.values()))
        stress_multiplier = max(1.0, 1 + abs(avg_shock) * 2)
        return round(base_var * stress_multiplier, 6)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    tickers = ["SPY", "TLT", "EEM", "GLD", "HYG"]
    weights = pd.Series([0.30, 0.20, 0.20, 0.15, 0.15], index=tickers)
    asset_map = {"SPY": "equity", "TLT": "rates", "EEM": "em",
                 "GLD": "commodities", "HYG": "credit"}
    cov = pd.DataFrame(
        np.diag([0.04, 0.01, 0.06, 0.03, 0.02]),
        index=tickers, columns=tickers
    )

    st = StressTester(weights, cov, asset_map)
    results = st.run_all_scenarios()
    print(results)

    rev = st.reverse_stress(loss_threshold=0.10)
    print(f"Reverse stress equity shock to breach -10%: {rev:.1%}")
