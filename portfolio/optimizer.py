
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization Engine.
    Implements Black-Litterman and Mean-Variance Optimization with shrinkage.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate

    def black_litterman(self, 
                       market_caps: pd.Series, 
                       sigma: pd.DataFrame, 
                       views: List[Dict], 
                       tau: float = 0.05, 
                       risk_aversion: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Black-Litterman Model for expected returns.
        """
        # 1. Market Equilibrium Returns (Prior)
        market_weights = market_caps / market_caps.sum()
        pi = risk_aversion * sigma.dot(market_weights)
        
        if not views:
            return pi, sigma

        # 2. View Matrix (P) and View Vector (Q)
        n_assets = len(market_weights)
        P = np.zeros((len(views), n_assets))
        Q = np.zeros(len(views))
        omega_diag = []
        
        for i, view in enumerate(views):
            # Example: {'assets': ['AAPL', 'MSFT'], 'weights': [1, -1], 'return': 0.05, 'conf': 0.1}
            for asset, weight in zip(view['assets'], view['weights']):
                idx = market_weights.index.get_loc(asset)
                P[i, idx] = weight
            Q[i] = view['return']
            omega_diag.append(view.get('conf', 0.1))
            
        Omega = np.diag(omega_diag)
        
        # 3. Posterior Estimates
        # Mean: [ (tau*Sigma)^-1 + P' * Omega^-1 * P ]^-1 * [ (tau*Sigma)^-1 * pi + P' * Omega^-1 * Q ]
        tau_sigma_inv = np.linalg.inv(tau * sigma)
        omega_inv = np.linalg.inv(Omega)
        
        middle_term = np.linalg.inv(tau_sigma_inv + P.T.dot(omega_inv).dot(P))
        posterior_mu = middle_term.dot(tau_sigma_inv.dot(pi) + P.T.dot(omega_inv).dot(Q))
        
        # Posterior Covariance (simplified)
        posterior_sigma = sigma + middle_term
        
        return posterior_mu, posterior_sigma

    def mean_variance_optimize(self, 
                               expected_returns: np.ndarray, 
                               covariance: np.ndarray, 
                               constraints: Optional[List] = None,
                               target_vol: Optional[float] = None) -> np.ndarray:
        """
        Standard Markowitz Mean-Variance Optimization.
        """
        n = len(expected_returns)
        
        def objective(weights):
            port_return = weights.dot(expected_returns)
            port_vol = np.sqrt(weights.dot(covariance).dot(weights))
            # Maximize Sharpe (negative for minimization)
            return -(port_return - self.rf) / (port_vol + 1e-9)

        if constraints is None:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Fully invested
            ]
        
        bounds = [(0, 1) for _ in range(n)] # No short selling by default
        initial_weights = np.ones(n) / n
        
        res = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return res.x if res.success else initial_weights

if __name__ == "__main__":
    # Demo code
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    mcap = pd.Series([2.5e12, 2.3e12, 1.5e12, 1.3e12, 0.8e12], index=assets)
    
    # Mock covariance
    cov = pd.DataFrame(np.diag([0.04, 0.05, 0.06, 0.07, 0.15]), index=assets, columns=assets)
    
    opt = PortfolioOptimizer()
    
    # Absolute view: TSLA will return 15%
    # Relative view: AAPL will outperform MSFT by 3%
    views = [
        {'assets': ['TSLA'], 'weights': [1], 'return': 0.15, 'conf': 0.01},
        {'assets': ['AAPL', 'MSFT'], 'weights': [1, -1], 'return': 0.03, 'conf': 0.01}
    ]
    
    mu_bl, sigma_bl = opt.black_litterman(mcap, cov, views)
    weights = opt.mean_variance_optimize(mu_bl, sigma_bl)
    
    print("Optimized Weights:")
    for a, w in zip(assets, weights):
        print(f"{a}: {w:.2%}")
