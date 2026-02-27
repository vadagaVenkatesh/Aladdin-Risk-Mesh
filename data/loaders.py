"""
data/loaders.py
Aladdin Risk Mesh - Multi-Asset Data Loader

Authors: VDG Venkatesh & Agentic AI
Description:
    Fetches daily OHLCV prices and macro data for:
    - Equity indices (SPY, EFA, EEM, VGK, EWJ)
    - Bond futures (TLT, IEF, SHY, EMB, HYG)
    - FX pairs (DX-Y.NYB, EURUSD, GBPUSD, JPYUSD, AUDUSD)
    - Commodity futures (GC=F, CL=F, NG=F, ZC=F, ZS=F)
    - Macro proxies (inflation, rates via FRED)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MultiAssetLoader:
    """
    Unified multi-asset data loader for equity, bond, FX,
    commodity, and macro instrument universes.
    """

    EQUITY_TICKERS = {
        'SPY':  'S&P 500 ETF',
        'QQQ':  'NASDAQ 100 ETF',
        'EFA':  'MSCI EAFE ETF',
        'EEM':  'MSCI EM ETF',
        'VGK':  'FTSE Europe ETF',
        'EWJ':  'MSCI Japan ETF',
        'EWZ':  'MSCI Brazil ETF',
        'FXI':  'FTSE China ETF',
        'IWM':  'Russell 2000 ETF',
        'VNQ':  'Real Estate ETF',
    }

    BOND_TICKERS = {
        'TLT':  'iShares 20Y+ Treasury',
        'IEF':  'iShares 7-10Y Treasury',
        'SHY':  'iShares 1-3Y Treasury',
        'TIP':  'iShares TIPS Bond',
        'EMB':  'iShares EM Bond',
        'HYG':  'iShares High Yield Bond',
        'LQD':  'iShares Investment Grade Corp',
        'BND':  'Vanguard Total Bond',
    }

    COMMODITY_TICKERS = {
        'GC=F':  'Gold Futures',
        'CL=F':  'WTI Crude Oil Futures',
        'NG=F':  'Natural Gas Futures',
        'ZC=F':  'Corn Futures',
        'ZS=F':  'Soybean Futures',
        'SI=F':  'Silver Futures',
        'HG=F':  'Copper Futures',
        'GLD':   'SPDR Gold Shares',
        'USO':   'US Oil Fund',
        'DBA':   'Invesco DB Agriculture',
    }

    FX_TICKERS = {
        'DX-Y.NYB': 'USD Index',
        'EURUSD=X':  'EUR/USD',
        'GBPUSD=X':  'GBP/USD',
        'JPY=X':     'USD/JPY',
        'AUDUSD=X':  'AUD/USD',
        'CADUSD=X':  'CAD/USD',
        'CHF=X':     'USD/CHF',
        'CNY=X':     'USD/CNY',
    }

    MACRO_TICKERS = {
        '^VIX':  'CBOE Volatility Index',
        '^TNX':  'US 10Y Treasury Yield',
        '^FVX':  'US 5Y Treasury Yield',
        '^IRX':  'US 3M Treasury Bill',
        '^TYX':  'US 30Y Treasury Yield',
    }

    def __init__(self, start: str = '2010-01-01', end: str = '2025-12-31'):
        self.start = start
        self.end = end
        self._cache: Dict[str, pd.DataFrame] = {}

    def _fetch(self, tickers: List[str]) -> pd.DataFrame:
        """Download adjusted close prices for a list of tickers."""
        key = tuple(sorted(tickers))
        if key in self._cache:
            return self._cache[key]
        data = yf.download(
            tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data[['Close']]
            prices.columns = [tickers[0]]
        prices = prices.dropna(how='all')
        self._cache[key] = prices
        return prices

    def load_equity(self) -> pd.DataFrame:
        """Load equity index ETF price series."""
        return self._fetch(list(self.EQUITY_TICKERS.keys()))

    def load_bonds(self) -> pd.DataFrame:
        """Load bond ETF / fixed income price series."""
        return self._fetch(list(self.BOND_TICKERS.keys()))

    def load_commodities(self) -> pd.DataFrame:
        """Load commodity futures and ETF price series."""
        return self._fetch(list(self.COMMODITY_TICKERS.keys()))

    def load_fx(self) -> pd.DataFrame:
        """Load FX pair price series."""
        return self._fetch(list(self.FX_TICKERS.keys()))

    def load_macro(self) -> pd.DataFrame:
        """Load macro proxies (VIX, treasury yields)."""
        return self._fetch(list(self.MACRO_TICKERS.keys()))

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all asset classes and return a dict keyed by asset class.
        Returns:
            dict with keys: 'equity', 'bonds', 'commodities', 'fx', 'macro'
        """
        print("[Loader] Fetching equity...")
        equity = self.load_equity()
        print("[Loader] Fetching bonds...")
        bonds = self.load_bonds()
        print("[Loader] Fetching commodities...")
        commodities = self.load_commodities()
        print("[Loader] Fetching FX...")
        fx = self.load_fx()
        print("[Loader] Fetching macro proxies...")
        macro = self.load_macro()
        return {
            'equity': equity,
            'bonds': bonds,
            'commodities': commodities,
            'fx': fx,
            'macro': macro,
        }

    def get_universe_prices(self) -> pd.DataFrame:
        """
        Combine all asset classes into a single wide DataFrame of closing prices.
        Columns: ticker symbols; Index: datetime.
        """
        all_data = self.load_all()
        combined = pd.concat(all_data.values(), axis=1)
        combined = combined.sort_index().ffill().dropna(how='all')
        return combined

    def compute_returns(self, prices: Optional[pd.DataFrame] = None,
                        method: str = 'log') -> pd.DataFrame:
        """
        Compute daily returns from price series.
        Args:
            prices: DataFrame of prices; if None, loads universe prices.
            method: 'log' for log returns, 'simple' for arithmetic.
        Returns:
            DataFrame of daily returns.
        """
        if prices is None:
            prices = self.get_universe_prices()
        if method == 'log':
            return np.log(prices / prices.shift(1)).dropna(how='all')
        else:
            return prices.pct_change().dropna(how='all')


if __name__ == '__main__':
    loader = MultiAssetLoader(start='2015-01-01', end='2025-12-31')
    prices = loader.get_universe_prices()
    returns = loader.compute_returns(prices)
    print(f"Universe shape: {prices.shape}")
    print(f"Returns shape:  {returns.shape}")
    print(prices.tail())
