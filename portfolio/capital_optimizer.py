"""Capital-Efficient Portfolio Optimizer

Extends traditional MVO and Black-Litterman with explicit capital cost considerations,
aligned with Basel IV capital efficiency objectives.

Objective: maximize (expected_return - lambda * capital_cost) subject to risk constraints
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import yaml


class CapitalEfficientOptimizer:
    """Portfolio optimizer with capital efficiency as explicit objective component."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize optimizer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.capital_config = self.config.get('capital_optimization', {})
        self.lambda_capital = self.capital_config.get('lambda_capital_cost', 0.10)
        self.capital_budget_total = self.capital_config.get('capital_budget_total', 1e6)
        
    def calculate_capital_cost_vector(self,
                                     expected_returns: np.ndarray,
                                     capital_charges: np.ndarray) -> np.ndarray:
        """Calculate marginal capital cost per unit of investment.
        
        Args:
            expected_returns: Expected return vector
            capital_charges: Capital requirement per unit invested
            
        Returns:
            Capital cost vector (cost per dollar invested)
        """
        # Capital cost = capital_charge * cost_of_capital
        # Simplified: use risk-free rate + capital hurdle
        cost_of_capital = 0.10  # 10% hurdle rate for capital
        
        return capital_charges * cost_of_capital
    
    def objective_risk_adjusted_return_net_capital(self,
                                                  weights: np.ndarray,
                                                  expected_returns: np.ndarray,
                                                  covariance: np.ndarray,
                                                  capital_costs: np.ndarray,
                                                  lambda_risk: float = 1.0) -> float:
        """Objective function: maximize return - risk - capital cost.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected return vector
            covariance: Covariance matrix
            capital_costs: Capital cost vector
            lambda_risk: Risk aversion parameter
            
        Returns:
            Negative objective (for minimization)
        """
        # Portfolio return
        port_return = weights.dot(expected_returns)
        
        # Portfolio risk (variance)
        port_variance = weights.T.dot(covariance).dot(weights)
        
        # Capital cost
        port_capital_cost = weights.dot(capital_costs)
        
        # Objective: return - lambda_risk * variance - lambda_capital * capital_cost
        objective = port_return - lambda_risk * port_variance - self.lambda_capital * port_capital_cost
        
        return -objective  # Negative for minimization
    
    def optimize_with_capital_budget(self,
                                    expected_returns: np.ndarray,
                                    covariance: pd.DataFrame,
                                    capital_charges: np.ndarray,
                                    asset_names: List[str],
                                    capital_budget: Optional[float] = None,
                                    gross_limit: float = 2.0,
                                    net_limit: float = 1.0,
                                    long_only: bool = False) -> Dict:
        """Optimize portfolio with capital budget constraint.
        
        Args:
            expected_returns: Expected return vector
            covariance: Covariance matrix
            capital_charges: Capital requirement per unit (from FRTB/SA-CCR)
            asset_names: List of asset names
            capital_budget: Maximum capital allocation (defaults to config)
            gross_limit: Maximum gross exposure
            net_limit: Maximum net exposure
            long_only: If True, constrain to long-only positions
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        if capital_budget is None:
            capital_budget = self.capital_budget_total
        
        n_assets = len(expected_returns)
        
        # Calculate capital costs
        capital_costs = self.calculate_capital_cost_vector(expected_returns, capital_charges)
        
        # Initial guess: equal weight
        w0 = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = []
        
        # Net exposure constraint: sum(weights) <= net_limit
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: net_limit - np.sum(w)
        })
        
        # Gross exposure constraint: sum(abs(weights)) <= gross_limit
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: gross_limit - np.sum(np.abs(w))
        })
        
        # Capital budget constraint: sum(abs(weights) * capital_charges) <= budget
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: capital_budget - np.sum(np.abs(w) * capital_charges)
        })
        
        # Bounds
        if long_only:
            bounds = [(0, 1) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]  # Allow short positions
        
        # Optimize
        result = minimize(
            self.objective_risk_adjusted_return_net_capital,
            w0,
            args=(expected_returns, covariance.values, capital_costs, 1.0),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        port_return = optimal_weights.dot(expected_returns)
        port_variance = optimal_weights.T.dot(covariance.values).dot(optimal_weights)
        port_volatility = np.sqrt(port_variance)
        port_sharpe = port_return / port_volatility if port_volatility > 0 else 0
        
        port_capital_usage = np.sum(np.abs(optimal_weights) * capital_charges)
        port_capital_cost = optimal_weights.dot(capital_costs)
        
        capital_efficiency = port_return / port_capital_usage if port_capital_usage > 0 else 0
        
        return {
            'weights': pd.Series(optimal_weights, index=asset_names),
            'expected_return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': port_sharpe,
            'capital_usage': port_capital_usage,
            'capital_cost': port_capital_cost,
            'capital_efficiency': capital_efficiency,
            'capital_utilization_pct': (port_capital_usage / capital_budget) * 100,
            'gross_exposure': np.sum(np.abs(optimal_weights)),
            'net_exposure': np.sum(optimal_weights),
            'optimization_success': result.success
        }
    
    def optimize_by_sleeve(self,
                          expected_returns_by_sleeve: Dict[str, np.ndarray],
                          covariance_by_sleeve: Dict[str, pd.DataFrame],
                          capital_charges_by_sleeve: Dict[str, np.ndarray],
                          asset_names_by_sleeve: Dict[str, List[str]],
                          sleeve_capital_budgets: Optional[Dict[str, float]] = None) -> Dict:
        """Optimize each sleeve separately with individual capital budgets.
        
        Args:
            expected_returns_by_sleeve: Returns by sleeve (equity, rates, fx, etc.)
            covariance_by_sleeve: Covariance by sleeve
            capital_charges_by_sleeve: Capital charges by sleeve
            asset_names_by_sleeve: Asset names by sleeve
            sleeve_capital_budgets: Capital budget per sleeve (from config if None)
            
        Returns:
            Dictionary with results by sleeve
        """
        if sleeve_capital_budgets is None:
            sleeve_capital_budgets = {
                'equities': self.capital_config.get('capital_budget_equity_sleeve', 400000),
                'rates': self.capital_config.get('capital_budget_rates_sleeve', 300000),
                'fx': self.capital_config.get('capital_budget_fx_sleeve', 200000),
                'commodities': self.capital_config.get('capital_budget_commodity_sleeve', 100000)
            }
        
        results_by_sleeve = {}
        
        for sleeve_name in expected_returns_by_sleeve.keys():
            print(f"\nOptimizing sleeve: {sleeve_name}")
            
            result = self.optimize_with_capital_budget(
                expected_returns=expected_returns_by_sleeve[sleeve_name],
                covariance=covariance_by_sleeve[sleeve_name],
                capital_charges=capital_charges_by_sleeve[sleeve_name],
                asset_names=asset_names_by_sleeve[sleeve_name],
                capital_budget=sleeve_capital_budgets.get(sleeve_name, 100000)
            )
            
            results_by_sleeve[sleeve_name] = result
        
        # Aggregate metrics
        total_capital_usage = sum(r['capital_usage'] for r in results_by_sleeve.values())
        total_expected_return = sum(r['expected_return'] for r in results_by_sleeve.values())
        
        aggregate = {
            'sleeves': results_by_sleeve,
            'total_capital_usage': total_capital_usage,
            'total_expected_return': total_expected_return,
            'aggregate_capital_efficiency': total_expected_return / total_capital_usage if total_capital_usage > 0 else 0
        }
        
        return aggregate
    
    def check_rebalance_trigger(self,
                               current_capital_efficiency: float,
                               baseline_capital_efficiency: float,
                               threshold: Optional[float] = None) -> Dict[str, Any]:
        """Check if capital efficiency drop triggers rebalancing.
        
        Args:
            current_capital_efficiency: Current return/capital ratio
            baseline_capital_efficiency: Historical or target efficiency
            threshold: Trigger threshold (default from config)
            
        Returns:
            Dictionary with trigger status and details
        """
        if threshold is None:
            threshold = self.capital_config.get('capital_efficiency_rebal_trigger', 0.20)
        
        efficiency_drop_pct = (baseline_capital_efficiency - current_capital_efficiency) / baseline_capital_efficiency
        
        trigger_rebalance = efficiency_drop_pct >= threshold
        
        return {
            'trigger_rebalance': trigger_rebalance,
            'current_efficiency': current_capital_efficiency,
            'baseline_efficiency': baseline_capital_efficiency,
            'efficiency_drop_pct': efficiency_drop_pct,
            'threshold': threshold,
            'recommendation': 'rebalance' if trigger_rebalance else 'hold'
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Mock data for equity sleeve
    n_assets = 5
    assets = ['US_Eq', 'EU_Eq', 'EM_Eq', 'Japan_Eq', 'UK_Eq']
    
    expected_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.09])
    cov_matrix = pd.DataFrame(
        np.random.uniform(0.01, 0.03, (n_assets, n_assets)),
        index=assets,
        columns=assets
    )
    # Make symmetric
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    np.fill_diagonal(cov_matrix.values, 0.04)
    
    # Capital charges from FRTB (higher for EM, lower for developed)
    capital_charges = np.array([0.30, 0.28, 0.40, 0.25, 0.30])
    
    optimizer = CapitalEfficientOptimizer()
    
    result = optimizer.optimize_with_capital_budget(
        expected_returns=expected_returns,
        covariance=cov_matrix,
        capital_charges=capital_charges,
        asset_names=assets,
        capital_budget=500000,
        gross_limit=1.5,
        net_limit=1.0
    )
    
    print("=== Capital-Efficient Portfolio Optimization ===")
    print(f"\nOptimal Weights:")
    print(result['weights'])
    print(f"\nExpected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"\nCapital Usage: ${result['capital_usage']:,.2f}")
    print(f"Capital Utilization: {result['capital_utilization_pct']:.1f}%")
    print(f"Capital Efficiency (Return/Capital): {result['capital_efficiency']:.4f}")
    print(f"\nGross Exposure: {result['gross_exposure']:.2f}")
    print(f"Net Exposure: {result['net_exposure']:.2f}")
