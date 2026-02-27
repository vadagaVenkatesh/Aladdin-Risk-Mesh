import numpy as np
import pandas as pd

class RiskEngine:
    def __init__(self, confidence_level=0.99):
        self.confidence_level = confidence_level
        
    def calculate_var(self, returns):
        """Calculates Value at Risk (VaR) using the historical method."""
        return np.percentile(returns, (1 - self.confidence_level) * 100)
    
    def monte_carlo_simulation(self, current_price, days, iterations, mu, sigma):
        """Performs a Monte Carlo simulation for price paths."""
        dt = 1 / 252
        prices = np.zeros((days, iterations))
        prices[0] = current_price
        
        for t in range(1, days):
            z = np.random.standard_normal(iterations)
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            
        return prices

if __name__ == "__main__":
    engine = RiskEngine()
    print("RiskEngine initialized.")
