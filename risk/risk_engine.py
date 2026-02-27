import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Union

class RiskEngine:
    """
    Cross-asset Risk Management Engine.
    Provides VaR, CVaR, and Stress Testing capabilities.
    """
    
    def __init__(self, confidence_level: float = 0.99):
        self.alpha = confidence_level

    def calculate_var_historical(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """
        Calculate Historical Value at Risk.
        """
        portfolio_returns = returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - self.alpha) * 100)
        return -var

    def calculate_cvar_historical(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        """
        portfolio_returns = returns.dot(weights)
        var = self.calculate_var_historical(returns, weights)
        cvar = portfolio_returns[portfolio_returns <= -var].mean()
        return -cvar

    def calculate_parametric_var(self, 
                                 expected_returns: np.ndarray, 
                                 covariance: pd.DataFrame, 
                                 weights: np.ndarray) -> float:
        """
        Calculate Parametric (Delta-Normal) VaR.
        """
        port_variance = weights.T.dot(covariance).dot(weights)
        port_std = np.sqrt(port_variance)
        z_score = norm.ppf(1 - self.alpha)
        var = (weights.dot(expected_returns) + z_score * port_std)
        return -var

    def run_stress_test(self, weights: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """
        Simulate portfolio impact under historical stress scenarios.
        """
        scenarios = {
            "2008_Lehman": -0.45,
            "2020_Covid": -0.30,
            "1987_Black_Monday": -0.22,
            "2026_Hyperinflation_Shock": -0.15
        }
        
        results = {}
        for name, shock in scenarios.items():
            results[name] = np.sum(weights * shock)
            
        return results

    def marginal_contribution_to_risk(self, weights: np.ndarray, covariance: pd.DataFrame) -> pd.Series:
        """
        Calculate Marginal Contribution to Risk (MCR) for each asset.
        """
        port_std = np.sqrt(weights.T.dot(covariance).dot(weights))
        mcr = covariance.dot(weights) / port_std
        return mcr

if __name__ == "__main__":
    # Example usage
    assets = ['SPY', 'TLT', 'GLD', 'BTC']
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    
    # Mock data
    np.random.seed(42)
    mock_returns = pd.DataFrame(
        np.random.normal(0.0005, 0.01, (1000, 4)), 
        columns=assets
    )
    mock_cov = mock_returns.cov()
    
    re = RiskEngine()
    var_h = re.calculate_var_historical(mock_returns, weights)
    cvar_h = re.calculate_cvar_historical(mock_returns, weights)
    
    print(f"Historical VaR (99%): {var_h:.2%}")
    print(f"Historical CVaR (99%): {cvar_h:.2%}")
    
    stress_results = re.run_stress_test(weights, assets)
    print("
Stress Test Results:")
    for scenario, impact in stress_results.items():
        print(f"{scenario}: {impact:.2%}")
