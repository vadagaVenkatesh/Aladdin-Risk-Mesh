import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import os

class DataLoader:
    def __init__(self, config_path='config/settings.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def fetch_equity_data(self, tickers, start_date, end_date):
        """Fetches historical equity data from Yahoo Finance."""
        print(f"Fetching data for: {tickers}")
        data = yf.download(tickers, start=start_date, end=end_date)
        return data['Adj Close']
    
    def fetch_macro_data(self, indicator):
        """Placeholder for fetching macro indicators."""
        # For MVP, returning mock data or simple FRED/WB integration
        print(f"Fetching macro data for: {indicator}")
        return pd.Series(np.random.randn(100), name=indicator)

if __name__ == "__main__":
    # Test initialization
    # Assuming the config file is in the parent directory for local testing
    # In the repo it will be at config/settings.yaml
    print("DataLoader script ready.")
