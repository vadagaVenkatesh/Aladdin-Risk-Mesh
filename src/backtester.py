import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        
    def run_backtest(self, price_data, signals):
        """Runs a basic backtest of the strategy."""
        portfolio = pd.DataFrame(index=price_data.index)
        portfolio['holdings'] = signals['signal'] * price_data
        
        # Simple cash calculation
        pos_diff = signals['positions']
        portfolio['cash'] = self.initial_capital - (pos_diff * price_data).cumsum()
        
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio

if __name__ == "__main__":
    tester = Backtester()
    print("Backtester initialized.")
