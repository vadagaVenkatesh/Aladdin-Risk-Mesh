"""
risk/position_limits.py
Position & Exposure Limit Framework for Global Macro L/S Fund
Authors: VDG Venkatesh & Agentic AI
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LimitConfig:
    """Per-fund limit configuration."""
    # Single-name limits
    max_single_long:  float = 0.10   # 10% of NAV
    max_single_short: float = 0.10

    # Sector / asset-class limits
    max_sector_net:   float = 0.25
    max_sector_gross: float = 0.40

    # Portfolio-level limits
    max_gross_leverage: float = 2.0   # 200% gross
    max_net_exposure:   float = 0.50  # +/-50% net

    # VaR limits (as fraction of NAV)
    max_daily_var_99:   float = 0.02  # 2% daily VaR
    max_stress_loss:    float = 0.15  # 15% max stress loss

    # Liquidity limit (days to liquidate)
    max_days_to_liq:    float = 10.0

    # Country/region concentration
    max_country_gross:  float = 0.30


@dataclass
class LimitBreach:
    limit_name: str
    ticker: Optional[str]
    current_value: float
    limit_value: float
    severity: str   # "WARNING" | "BREACH"

    def __str__(self):
        return (f"[{self.severity}] {self.limit_name} "
                f"({'@' + self.ticker if self.ticker else 'PORTFOLIO'}: "
                f"current={self.current_value:.4f}, limit={self.limit_value:.4f})")


class PositionLimitChecker:
    """
    Real-time position and risk limit checker.

    Usage
    -----
    checker = PositionLimitChecker(config)
    breaches = checker.run_all_checks(
        weights, sector_map, country_map, var_99, stress_loss
    )
    """

    def __init__(self, config: Optional[LimitConfig] = None):
        self.config  = config or LimitConfig()
        self.breaches: List[LimitBreach] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _add(self, name, ticker, current, limit, warn_pct=0.90):
        if abs(current) >= abs(limit):
            self.breaches.append(LimitBreach(name, ticker, current, limit, "BREACH"))
        elif abs(current) >= abs(limit) * warn_pct:
            self.breaches.append(LimitBreach(name, ticker, current, limit, "WARNING"))

    # ------------------------------------------------------------------
    # 1. Single-name checks
    # ------------------------------------------------------------------
    def check_single_name(
        self,
        weights: pd.Series,
    ) -> None:
        for ticker, w in weights.items():
            if w > 0:
                self._add("single_long", ticker, w, self.config.max_single_long)
            else:
                self._add("single_short", ticker, abs(w), self.config.max_single_short)

    # ------------------------------------------------------------------
    # 2. Sector / asset-class checks
    # ------------------------------------------------------------------
    def check_sector(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
    ) -> None:
        sectors = pd.Series(sector_map)
        df = pd.DataFrame({"w": weights, "sector": sectors})
        for sec, grp in df.groupby("sector"):
            net   = grp["w"].sum()
            gross = grp["w"].abs().sum()
            self._add("sector_net",   sec, abs(net),   self.config.max_sector_net)
            self._add("sector_gross", sec, gross,      self.config.max_sector_gross)

    # ------------------------------------------------------------------
    # 3. Portfolio-level leverage
    # ------------------------------------------------------------------
    def check_leverage(
        self,
        weights: pd.Series,
    ) -> None:
        gross = weights.abs().sum()
        net   = weights.sum()
        self._add("gross_leverage", None, gross, self.config.max_gross_leverage)
        self._add("net_exposure",   None, abs(net), self.config.max_net_exposure)

    # ------------------------------------------------------------------
    # 4. VaR limit
    # ------------------------------------------------------------------
    def check_var(
        self,
        var_99: float,
    ) -> None:
        self._add("daily_var_99", None, var_99, self.config.max_daily_var_99)

    # ------------------------------------------------------------------
    # 5. Stress loss limit
    # ------------------------------------------------------------------
    def check_stress(
        self,
        stress_loss: float,
    ) -> None:
        self._add("stress_loss", None, abs(stress_loss), self.config.max_stress_loss)

    # ------------------------------------------------------------------
    # 6. Country concentration
    # ------------------------------------------------------------------
    def check_country(
        self,
        weights: pd.Series,
        country_map: Dict[str, str],
    ) -> None:
        countries = pd.Series(country_map)
        df = pd.DataFrame({"w": weights, "country": countries})
        for cty, grp in df.groupby("country"):
            gross = grp["w"].abs().sum()
            self._add("country_gross", cty, gross, self.config.max_country_gross)

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all_checks(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        country_map: Optional[Dict[str, str]] = None,
        var_99: float = 0.0,
        stress_loss: float = 0.0,
    ) -> List[LimitBreach]:
        self.breaches = []
        self.check_single_name(weights)
        self.check_sector(weights, sector_map)
        self.check_leverage(weights)
        self.check_var(var_99)
        self.check_stress(stress_loss)
        if country_map:
            self.check_country(weights, country_map)
        return self.breaches

    def summary(self) -> pd.DataFrame:
        if not self.breaches:
            return pd.DataFrame(columns=["limit", "ticker", "current", "limit_val", "severity"])
        rows = [{
            "limit":     b.limit_name,
            "ticker":    b.ticker or "PORTFOLIO",
            "current":   round(b.current_value, 4),
            "limit_val": round(b.limit_value, 4),
            "severity":  b.severity,
        } for b in self.breaches]
        return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
class PositionLimiter:
    """
    Convenience wrapper around PositionLimitChecker.
    Accepts top-level limit parameters directly as keyword arguments
    so that callers can instantiate without building a LimitConfig first.

    Example
    -------
    limiter = PositionLimiter(max_leverage=2.0)
    limiter.check_leverage(weights)
    """

    def __init__(
        self,
        max_leverage: float = 2.0,
        max_single_long: float = 0.10,
        max_single_short: float = 0.10,
        max_sector_net: float = 0.25,
        max_sector_gross: float = 0.40,
        max_net_exposure: float = 0.50,
        max_daily_var_99: float = 0.02,
        max_stress_loss: float = 0.15,
        max_country_gross: float = 0.30,
    ):
        config = LimitConfig(
            max_gross_leverage=max_leverage,
            max_single_long=max_single_long,
            max_single_short=max_single_short,
            max_sector_net=max_sector_net,
            max_sector_gross=max_sector_gross,
            max_net_exposure=max_net_exposure,
            max_daily_var_99=max_daily_var_99,
            max_stress_loss=max_stress_loss,
            max_country_gross=max_country_gross,
        )
        self._checker = PositionLimitChecker(config)
        self.max_leverage = max_leverage

    def check_leverage(self, weights: pd.Series) -> List[LimitBreach]:
        """Check gross and net leverage limits."""
        self._checker.breaches = []
        self._checker.check_leverage(weights)
        return self._checker.breaches

    def run_all_checks(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        country_map: Optional[Dict[str, str]] = None,
        var_99: float = 0.0,
        stress_loss: float = 0.0,
    ) -> List[LimitBreach]:
        """Run full suite of limit checks."""
        return self._checker.run_all_checks(
            weights, sector_map, country_map, var_99, stress_loss
        )

    def summary(self) -> pd.DataFrame:
        """Return breach summary as DataFrame."""
        return self._checker.summary()
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tickers    = ["SPY", "TLT", "EEM", "GLD", "HYG", "XOM", "AAPL"]
    weights    = pd.Series([0.25, -0.15, 0.20, 0.12, -0.08, 0.15, -0.12], index=tickers)
    sector_map = {"SPY": "equity", "TLT": "rates", "EEM": "em",
                  "GLD": "commodity", "HYG": "credit", "XOM": "equity", "AAPL": "equity"}
    country_map = {t: "US" for t in tickers}
    country_map["EEM"] = "EM"

    checker  = PositionLimitChecker(LimitConfig(max_single_long=0.20))
    breaches = checker.run_all_checks(
        weights, sector_map, country_map,
        var_99=0.018, stress_loss=0.12
    )
    print(checker.summary().to_string())
    print(f"\nTotal breaches/warnings: {len(breaches)}")
