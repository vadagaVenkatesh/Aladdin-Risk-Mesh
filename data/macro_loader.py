"""
macro_loader.py

PROPRIETARY AND CONFIDENTIAL
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

This software and associated documentation files are the proprietary 
and confidential information of VDG Venkatesh. Unauthorized copying,
modification, distribution, or use of this software, via any medium,
is strictly prohibited without express written permission.

NO LICENSE IS GRANTED. This code may not be used, reproduced, or 
incorporated into any other projects without explicit authorization.
For licensing inquiries, contact the copyright holder.

Macro Data Loader for Global Macro Strategy
- Load equity indices, FX, rates, commodities
- FRED integration for macro indicators
- Bloomberg/Reuters API wrappers
- Real-time and historical data
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import yfinance as yf

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


class MacroDataLoader:
    """
    Unified data loader for macro assets and indicators.
    Supports Yahoo Finance, FRED, and extensible to Bloomberg.
    """
    
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_dir: str = "./data_cache"
    ):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        fred_api_key : str, optional
            FRED API key (defaults to env var FRED_API_KEY)
        cache_dir : str
            Local cache directory for storing downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize FRED client
        if FRED_AVAILABLE:
            self.fred_key = fred_api_key or os.getenv('FRED_API_KEY')
            if self.fred_key:
                self.fred = fredapi.Fred(api_key=self.fred_key)
            else:
                self.fred = None
        else:
            self.fred = None
    
    def load_equity_indices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load equity index data.
        
        Parameters:
        -----------
        tickers : list
            Yahoo Finance tickers (e.g., ['SPY', 'QQQ', 'EEM'])
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        use_cache : bool
            Use local cache if available
        
        Returns:
        --------
        pd.DataFrame : Multi-index DataFrame (date, ticker) with OHLCV
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"equity_{'-'.join(tickers)}_{start_date}_{end_date}.parquet"
        )
        
        if use_cache and os.path.exists(cache_file):
            return pd.read_parquet(cache_file)
        
        data = yf.download(
            tickers=' '.join(tickers),
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if use_cache:
            data.to_parquet(cache_file)
        
        return data
    
    def load_fx_rates(
        self,
        pairs: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load FX rates from Yahoo Finance.
        
        Parameters:
        -----------
        pairs : list
            FX pairs in Yahoo format (e.g., ['EURUSD=X', 'GBPUSD=X'])
        start_date : str
        end_date : str
        
        Returns:
        --------
        pd.DataFrame : FX rates (date index, columns = pairs)
        """
        data = yf.download(
            tickers=' '.join(pairs),
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        return data
    
    def load_commodities(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load commodity prices.
        
        Parameters:
        -----------
        tickers : list
            Commodity tickers (e.g., ['GC=F' (gold), 'CL=F' (oil)])
        start_date : str
        end_date : str
        
        Returns:
        --------
        pd.DataFrame : Commodity prices
        """
        data = yf.download(
            tickers=' '.join(tickers),
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        return data
    
    def load_fred_series(
        self,
        series_ids: Dict[str, str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load macro indicators from FRED.
        
        Parameters:
        -----------
        series_ids : dict
            {label: FRED_SERIES_ID}
            Examples:
            - 'CPI': 'CPIAUCSL'
            - 'GDP': 'GDP'
            - 'Unemployment': 'UNRATE'
            - 'Fed_Funds': 'DFF'
        start_date : str
        end_date : str
        
        Returns:
        --------
        pd.DataFrame : FRED data (date index, columns = labels)
        """
        if not self.fred:
            raise ValueError("FRED client not initialized. Provide API key.")
        
        data = {}
        for label, series_id in series_ids.items():
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                data[label] = series
            except Exception as e:
                print(f"Warning: Failed to load {label} ({series_id}): {e}")
        
        df = pd.DataFrame(data)
        return df
    
    def load_vix(
        self,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """
        Load VIX index.
        
        Parameters:
        -----------
        start_date : str
        end_date : str
        
        Returns:
        --------
        pd.Series : VIX values
        """
        vix = yf.download(
            '^VIX',
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        return vix
    
    def load_treasury_yields(
        self,
        maturities: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load US Treasury yields from FRED.
        
        Parameters:
        -----------
        maturities : list
            ['2Y', '10Y', '30Y', etc.]
            Maps to FRED series: DGS2, DGS10, DGS30
        start_date : str
        end_date : str
        
        Returns:
        --------
        pd.DataFrame : Treasury yields
        """
        if not self.fred:
            raise ValueError("FRED client not initialized.")
        
        maturity_map = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }
        
        series_ids = {mat: maturity_map[mat] for mat in maturities if mat in maturity_map}
        return self.load_fred_series(series_ids, start_date, end_date)
    
    def load_portfolio_universe(
        self,
        start_date: str,
        end_date: str,
        asset_universe: str = "macro_standard"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load a pre-defined asset universe.
        
        Parameters:
        -----------
        start_date : str
        end_date : str
        asset_universe : str
            - 'macro_standard': SPY, TLT, GLD, DXY, VIX
            - 'global_macro': Expanded set with EM, commodities
        
        Returns:
        --------
        dict : {asset_class: DataFrame}
        """
        if asset_universe == "macro_standard":
            equities = self.load_equity_indices(
                ['SPY', 'QQQ', 'EEM'],
                start_date,
                end_date
            )
            
            bonds = self.load_equity_indices(
                ['TLT', 'IEF', 'SHY'],
                start_date,
                end_date
            )
            
            commodities = self.load_equity_indices(
                ['GLD', 'USO'],
                start_date,
                end_date
            )
            
            vix = self.load_vix(start_date, end_date)
            
            fx = self.load_fx_rates(
                ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X'],
                start_date,
                end_date
            )
            
            return {
                'equities': equities,
                'bonds': bonds,
                'commodities': commodities,
                'vix': vix,
                'fx': fx
            }
        
        else:
            raise ValueError(f"Unknown asset universe: {asset_universe}")
    
    def resample_to_daily(
        self,
        df: pd.DataFrame,
        method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Resample lower-frequency data (e.g., monthly) to daily.
        
        Parameters:
        -----------
        df : pd.DataFrame
        method : str
            'ffill', 'bfill', 'interpolate'
        
        Returns:
        --------
        pd.DataFrame : Daily-resampled data
        """
        df_daily = df.resample('D').asfreq()
        
        if method == 'ffill':
            df_daily = df_daily.ffill()
        elif method == 'bfill':
            df_daily = df_daily.bfill()
        elif method == 'interpolate':
            df_daily = df_daily.interpolate()
        
        return df_daily


# ========================================
# Demo
# ========================================
if __name__ == "__main__":
    loader = MacroDataLoader()
    
    print("=" * 80)
    print("Macro Data Loader Demo")
    print("=" * 80)
    print()
    
    # Load standard universe
    start = '2023-01-01'
    end = '2024-01-01'
    
    print(f"Loading macro universe from {start} to {end}...")
    universe = loader.load_portfolio_universe(start, end)
    
    print(f"\nEquities: {universe['equities'].shape}")
    print(f"Bonds: {universe['bonds'].shape}")
    print(f"Commodities: {universe['commodities'].shape}")
    print(f"VIX: {universe['vix'].shape}")
    print(f"FX: {universe['fx'].shape}")
    
    print("\nSample equity data:")
    print(universe['equities']['Close'].head())
    print("\n" + "=" * 80)
  
