"""Risk-Aware Execution Module

Execution algorithms with regulatory-style risk controls:
- Intraday VaR limits
- Stress loss constraints
- Liquidity horizon constraints
- What-if stress testing before execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml


class RiskAwareExecutionEngine:
    """Execution engine with integrated risk controls aligned with Basel/FRTB standards."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize execution engine with risk constraints.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.exec_config = self.config.get('execution', {})
        self.max_volume_participation = self.exec_config.get('max_daily_volume_participation', 0.10)
        self.max_intraday_var_breach = self.exec_config.get('max_intraday_var_breach', 0.02)
        self.max_stress_loss_daily = self.exec_config.get('stress_loss_limit_daily', 0.05)
        self.liquidity_horizons = self.exec_config.get('liquidity_horizon_days', {})
        
    def calculate_intraday_var(self,
                              position_changes: pd.Series,
                              asset_volatilities: pd.Series,
                              correlations: pd.DataFrame,
                              confidence: float = 0.99) -> float:
        """Calculate intraday VaR for proposed trades.
        
        Args:
            position_changes: Proposed position changes by asset
            asset_volatilities: Intraday volatility per asset
            correlations: Correlation matrix
            confidence: VaR confidence level
            
        Returns:
            Intraday VaR estimate
        """
        from scipy.stats import norm
        
        # Position change vector
        position_vector = position_changes.values
        
        # Volatility as diagonal matrix
        vol_matrix = np.diag(asset_volatilities.values)
        
        # Covariance = vol * corr * vol
        covariance = vol_matrix @ correlations.values @ vol_matrix
        
        # Portfolio variance
        portfolio_variance = position_vector.T @ covariance @ position_vector
        portfolio_std = np.sqrt(portfolio_variance)
        
        # VaR
        z_score = norm.ppf(confidence)
        var = z_score * portfolio_std
        
        return var
    
    def run_what_if_stress(self,
                          proposed_positions: pd.Series,
                          stress_scenario: str = "2020_Covid") -> Dict[str, float]:
        """Run what-if stress test on proposed positions before execution.
        
        Args:
            proposed_positions: Proposed position sizes
            stress_scenario: Stress scenario name
            
        Returns:
            Dictionary with stressed P&L and risk metrics
        """
        # Simplified stress shocks by asset class
        stress_shocks = {
            "2020_Covid": {
                'equities': -0.30,
                'rates': -0.05,
                'fx': 0.15,
                'commodities': -0.25,
                'credit': -0.20
            },
            "2022_Energy_Spike": {
                'equities': -0.15,
                'rates': 0.10,
                'fx': 0.10,
                'commodities': 0.40,
                'credit': -0.10
            },
            "Global_Market_Shock": {
                'equities': -0.20,
                'rates': -0.10,
                'fx': 0.20,
                'commodities': -0.15,
                'credit': -0.25
            }
        }
        
        if stress_scenario not in stress_shocks:
            stress_scenario = "Global_Market_Shock"
        
        shocks = stress_shocks[stress_scenario]
        
        # Apply shocks based on inferred asset class from position name
        stressed_pnl = 0.0
        for asset, position in proposed_positions.items():
            asset_lower = asset.lower()
            
            # Infer asset class
            if 'eq' in asset_lower or 'stock' in asset_lower or 'spy' in asset_lower:
                shock = shocks['equities']
            elif 'bond' in asset_lower or 'tlt' in asset_lower or 'rate' in asset_lower:
                shock = shocks['rates']
            elif 'fx' in asset_lower or 'usd' in asset_lower or 'eur' in asset_lower:
                shock = shocks['fx']
            elif 'gold' in asset_lower or 'oil' in asset_lower or 'commodity' in asset_lower:
                shock = shocks['commodities']
            else:
                shock = shocks.get('credit', -0.10)
            
            stressed_pnl += position * shock
        
        # Calculate stress loss as percentage of total exposure
        total_exposure = proposed_positions.abs().sum()
        stress_loss_pct = abs(stressed_pnl) / total_exposure if total_exposure > 0 else 0
        
        return {
            'scenario': stress_scenario,
            'stressed_pnl': stressed_pnl,
            'stress_loss_pct': stress_loss_pct,
            'total_exposure': total_exposure,
            'breach_limit': stress_loss_pct > self.max_stress_loss_daily
        }
    
    def check_liquidity_constraints(self,
                                   order_sizes: pd.Series,
                                   asset_classes: pd.Series,
                                   execution_horizon_days: int = 1) -> Dict[str, Any]:
        """Check if execution can be completed within liquidity horizons.
        
        Args:
            order_sizes: Order sizes by asset
            asset_classes: Asset class labels
            execution_horizon_days: Proposed execution horizon
            
        Returns:
            Dictionary with liquidity constraint checks
        """
        violations = []
        
        for asset in order_sizes.index:
            asset_class = asset_classes[asset]
            
            # Get regulatory liquidity horizon for asset class
            required_horizon = self.liquidity_horizons.get(asset_class, 10)
            
            if execution_horizon_days < required_horizon:
                violations.append({
                    'asset': asset,
                    'asset_class': asset_class,
                    'proposed_horizon': execution_horizon_days,
                    'required_horizon': required_horizon,
                    'shortfall_days': required_horizon - execution_horizon_days
                })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'total_violations': len(violations)
        }
    
    def pre_trade_risk_check(self,
                            proposed_positions: pd.Series,
                            asset_classes: pd.Series,
                            current_portfolio_value: float,
                            asset_volatilities: Optional[pd.Series] = None,
                            correlations: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Comprehensive pre-trade risk check.
        
        Args:
            proposed_positions: Proposed position changes
            asset_classes: Asset class labels
            current_portfolio_value: Current portfolio value
            asset_volatilities: Intraday volatilities (optional)
            correlations: Correlation matrix (optional)
            
        Returns:
            Dictionary with all risk checks and approval status
        """
        checks = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'checks_passed': [],
            'checks_failed': [],
            'warnings': []
        }
        
        # 1. Liquidity constraint check
        liquidity_check = self.check_liquidity_constraints(
            proposed_positions,
            asset_classes,
            execution_horizon_days=1
        )
        
        if liquidity_check['compliant']:
            checks['checks_passed'].append('liquidity_constraints')
        else:
            checks['checks_failed'].append('liquidity_constraints')
            checks['warnings'].append(f"Liquidity violations: {liquidity_check['total_violations']}")
        
        # 2. Stress test
        stress_result = self.run_what_if_stress(proposed_positions, "Global_Market_Shock")
        
        if not stress_result['breach_limit']:
            checks['checks_passed'].append('stress_test')
        else:
            checks['checks_failed'].append('stress_test')
            checks['warnings'].append(
                f"Stress loss {stress_result['stress_loss_pct']:.2%} exceeds limit {self.max_stress_loss_daily:.2%}"
            )
        
        # 3. Intraday VaR check (if volatility data provided)
        if asset_volatilities is not None and correlations is not None:
            intraday_var = self.calculate_intraday_var(
                proposed_positions,
                asset_volatilities,
                correlations
            )
            
            var_pct_of_portfolio = intraday_var / current_portfolio_value
            
            if var_pct_of_portfolio <= self.max_intraday_var_breach:
                checks['checks_passed'].append('intraday_var')
            else:
                checks['checks_failed'].append('intraday_var')
                checks['warnings'].append(
                    f"Intraday VaR {var_pct_of_portfolio:.2%} exceeds limit {self.max_intraday_var_breach:.2%}"
                )
            
            checks['intraday_var'] = intraday_var
            checks['var_pct_of_portfolio'] = var_pct_of_portfolio
        
        # Overall approval
        checks['approved'] = len(checks['checks_failed']) == 0
        checks['liquidity_check'] = liquidity_check
        checks['stress_result'] = stress_result
        
        return checks
    
    def schedule_vwap_execution(self,
                               order_size: float,
                               expected_volume: float,
                               n_intervals: int = 10) -> pd.DataFrame:
        """Schedule VWAP execution with volume participation limits.
        
        Args:
            order_size: Total order size to execute
            expected_volume: Expected daily volume
            n_intervals: Number of execution intervals
            
        Returns:
            DataFrame with execution schedule
        """
        # Apply volume participation constraint
        max_execution = expected_volume * self.max_volume_participation
        
        if abs(order_size) > max_execution:
            print(f"Warning: Order size {order_size} exceeds max execution {max_execution}")
            print(f"Capping to {self.max_volume_participation:.0%} of daily volume")
            order_size = np.sign(order_size) * max_execution
        
        # Distribute across intervals
        interval_size = order_size / n_intervals
        
        schedule = pd.DataFrame({
            'interval': range(1, n_intervals + 1),
            'quantity': [interval_size] * n_intervals,
            'cumulative': [interval_size * i for i in range(1, n_intervals + 1)],
            'pct_complete': [(i / n_intervals) * 100 for i in range(1, n_intervals + 1)]
        })
        
        return schedule
    
    def log_execution_event(self,
                           event_type: str,
                           details: Dict,
                           severity: str = "info") -> None:
        """Log execution events for governance and audit.
        
        Args:
            event_type: Type of event (risk_violation, order_halt, execution_complete)
            details: Event details
            severity: Severity level (info, warning, critical)
        """
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        # In production, this would write to a proper logging system
        print(f"\n[EXECUTION LOG - {severity.upper()}]")
        print(f"Event: {event_type}")
        print(f"Details: {details}")


if __name__ == "__main__":
    # Example usage
    engine = RiskAwareExecutionEngine()
    
    # Proposed trades
    proposed_positions = pd.Series({
        'US_Equity_SPY': 500000,
        'EU_Bond_TLT': -300000,
        'Gold_GLD': 200000,
        'USD_JPY_FX': 150000
    })
    
    asset_classes = pd.Series({
        'US_Equity_SPY': 'equities',
        'EU_Bond_TLT': 'rates',
        'Gold_GLD': 'commodities',
        'USD_JPY_FX': 'fx'
    })
    
    # Run pre-trade risk check
    print("=== Pre-Trade Risk Check ===")
    risk_check = engine.pre_trade_risk_check(
        proposed_positions=proposed_positions,
        asset_classes=asset_classes,
        current_portfolio_value=10000000
    )
    
    print(f"\nApproval Status: {'APPROVED' if risk_check['approved'] else 'REJECTED'}")
    print(f"Checks Passed: {risk_check['checks_passed']}")
    print(f"Checks Failed: {risk_check['checks_failed']}")
    
    if risk_check['warnings']:
        print("\nWarnings:")
        for warning in risk_check['warnings']:
            print(f"  - {warning}")
    
    # Stress test details
    print(f"\nStress Test ({risk_check['stress_result']['scenario']}):")
    print(f"  Stressed P&L: ${risk_check['stress_result']['stressed_pnl']:,.2f}")
    print(f"  Loss %: {risk_check['stress_result']['stress_loss_pct']:.2%}")
    
    # Schedule execution if approved
    if risk_check['approved']:
        print("\n=== VWAP Execution Schedule ===")
        schedule = engine.schedule_vwap_execution(
            order_size=500000,
            expected_volume=10000000,
            n_intervals=10
        )
        print(schedule.head())
