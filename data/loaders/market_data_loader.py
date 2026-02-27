"""
Aladdin Risk Mesh — Market Data Loader (Bloomberg + Yahoo Fallback)
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL
This source code is the exclusive intellectual property of VDG Venkatesh.
Unauthorized use, reproduction, distribution, or modification of this code,
in whole or in part, without the express written consent of VDG Venkatesh
is strictly prohibited and may result in civil and criminal penalties.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import warnings

from environment.env_config import (
    BLOOMBERG_ENABLED,
    BLOOMBERG_HOST,
    BLOOMBERG_PORT,
    BLOOMBERG_TIMEOUT_SEC,
    YAHOO_FINANCE_ENABLED,
    YAHOO_INTRADAY_INTERVAL,
    YAHOO_LOOKBACK_DAYS,
    MAX_ASSETS,
)

log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Conditional imports
if BLOOMBERG_ENABLED:
    try:
        import blpapi
        BLOOMBERG_AVAILABLE = True
    except ImportError:
        log.warning("[DataLoader] Bloomberg blpapi not installed. Using Yahoo fallback only.")
        BLOOMBERG_AVAILABLE = False
else:
    BLOOMBERG_AVAILABLE = False

if YAHOO_FINANCE_ENABLED:
    try:
        import yfinance as yf
        YAHOO_AVAILABLE = True
    except ImportError:
        log.error("[DataLoader] yfinance not installed and Yahoo is enabled!")
        YAHOO_AVAILABLE = False
else:
    YAHOO_AVAILABLE = False


# ============================================================
# BLOOMBERG DATA LOADER
# ============================================================
class BloombergLoader:
    """Load market data from Bloomberg Terminal via blpapi."""

    def __init__(self, host: str = BLOOMBERG_HOST, port: int = BLOOMBERG_PORT):
        if not BLOOMBERG_AVAILABLE:
            raise RuntimeError("Bloomberg blpapi not available.")
        self.host = host
        self.port = port
        self.session = None

    def connect(self) -> None:
        """Establish connection to Bloomberg Terminal."""
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)
        self.session = blpapi.Session(session_options)
        if not self.session.start():
            raise ConnectionError("Failed to start Bloomberg session.")
        if not self.session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open Bloomberg refdata service.")
        log.info(f"[Bloomberg] Connected to {self.host}:{self.port}")

    def fetch_historical(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        fields: List[str] = ["PX_LAST"],
    ) -> pd.DataFrame:
        """
        Fetch historical daily data from Bloomberg.

        Parameters
        ----------
        tickers    : List of Bloomberg tickers (e.g., ['SPY US Equity', 'TLT US Equity'])
        start_date : Start date
        end_date   : End date
        fields     : Bloomberg fields (default: ['PX_LAST'])

        Returns
        -------
        DataFrame with DatetimeIndex and columns = tickers
        """
        if not self.session:
            self.connect()

        ref_data_service = self.session.getService("//blp/refdata")
        request = ref_data_service.createRequest("HistoricalDataRequest")

        for ticker in tickers:
            request.append("securities", ticker)
        for field in fields:
            request.append("fields", field)

        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")

        self.session.sendRequest(request)

        data = {}
        while True:
            event = self.session.nextEvent(BLOOMBERG_TIMEOUT_SEC * 1000)
            if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                for msg in event:
                    security_data = msg.getElement("securityData")
                    ticker = security_data.getElementAsString("security")
                    field_data = security_data.getElement("fieldData")

                    prices = []
                    dates = []
                    for point in field_data.values():
                        dates.append(point.getElementAsDatetime("date"))
                        prices.append(point.getElementAsFloat(fields[0]))

                    data[ticker] = pd.Series(prices, index=pd.to_datetime(dates))

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        df = pd.DataFrame(data)
        log.info(f"[Bloomberg] Fetched {len(df)} rows x {len(tickers)} tickers")
        return df

    def disconnect(self) -> None:
        """Close Bloomberg session."""
        if self.session:
            self.session.stop()
            log.info("[Bloomberg] Disconnected")


# ============================================================
# YAHOO FINANCE LOADER (FALLBACK)
# ============================================================
class YahooLoader:
    """Load market data from Yahoo Finance (free, no auth required)."""

    def fetch_historical(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.

        Parameters
        ----------
        tickers    : List of Yahoo tickers (e.g., ['SPY', 'TLT', 'GLD'])
        start_date : Start date
        end_date   : End date
        interval   : '1d' (daily), '1h' (hourly), '1m' (minute)

        Returns
        -------
        DataFrame with DatetimeIndex and columns = tickers (Close prices)
        """
        if not YAHOO_AVAILABLE:
            raise RuntimeError("Yahoo Finance (yfinance) not available.")

        data = {}
        failed = []

        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                if not df.empty:
                    data[ticker] = df["Close"]
                else:
                    failed.append(ticker)
            except Exception as e:
                log.warning(f"[Yahoo] Failed to fetch {ticker}: {e}")
                failed.append(ticker)

        df = pd.DataFrame(data)
        log.info(
            f"[Yahoo] Fetched {len(df)} rows x {len(data)} tickers | "
            f"Failed: {len(failed)}"
        )
        return df

    def fetch_intraday(
        self,
        tickers: List[str],
        lookback_days: int = YAHOO_LOOKBACK_DAYS,
        interval: str = YAHOO_INTRADAY_INTERVAL,
    ) -> pd.DataFrame:
        """
        Fetch intraday minute-bar data from Yahoo Finance.

        Parameters
        ----------
        tickers        : List of Yahoo tickers
        lookback_days  : Number of days to look back (max 30 for 1m bars)
        interval       : '1m', '5m', '15m', '1h'

        Returns
        -------
        DataFrame with DatetimeIndex and columns = tickers
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        return self.fetch_historical(tickers, start_date, end_date, interval=interval)


# ============================================================
# UNIFIED DATA LOADER (BLOOMBERG PRIMARY, YAHOO FALLBACK)
# ============================================================
class MarketDataLoader:
    """Unified market data loader with Bloomberg primary and Yahoo fallback."""

    def __init__(self, prefer_bloomberg: bool = BLOOMBERG_ENABLED):
        self.prefer_bloomberg = prefer_bloomberg and BLOOMBERG_AVAILABLE
        self.bloomberg = BloombergLoader() if self.prefer_bloomberg else None
        self.yahoo = YahooLoader() if YAHOO_AVAILABLE else None

        if not self.bloomberg and not self.yahoo:
            raise RuntimeError("No data sources available (Bloomberg and Yahoo both unavailable).")

        log.info(
            f"[MarketDataLoader] Initialized | "
            f"Bloomberg={self.prefer_bloomberg} | Yahoo={YAHOO_AVAILABLE}"
        )

    def fetch(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        use_yahoo_tickers: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch market data with automatic fallback.

        Parameters
        ----------
        tickers            : List of tickers
        start_date         : Start date
        end_date           : End date
        use_yahoo_tickers  : If True, assume tickers are Yahoo format (SPY, not SPY US Equity)

        Returns
        -------
        DataFrame with DatetimeIndex and columns = tickers
        """
        if len(tickers) > MAX_ASSETS:
            log.warning(f"[MarketDataLoader] Requested {len(tickers)} tickers exceeds MAX_ASSETS={MAX_ASSETS}")

        # Try Bloomberg first
        if self.prefer_bloomberg and not use_yahoo_tickers:
            try:
                df = self.bloomberg.fetch_historical(tickers, start_date, end_date)
                if not df.empty:
                    return df
                log.warning("[MarketDataLoader] Bloomberg returned empty data, falling back to Yahoo")
            except Exception as e:
                log.error(f"[MarketDataLoader] Bloomberg failed: {e}, falling back to Yahoo")

        # Fallback to Yahoo
        if self.yahoo:
            df = self.yahoo.fetch_historical(tickers, start_date, end_date)
            return df
        else:
            raise RuntimeError("All data sources failed.")

    def disconnect(self) -> None:
        """Disconnect all data sources."""
        if self.bloomberg:
            self.bloomberg.disconnect()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Example: Load SPY data (Yahoo fallback since Bloomberg likely unavailable)
    loader = MarketDataLoader(prefer_bloomberg=False)
    df = loader.fetch(
        tickers=["SPY", "TLT", "GLD"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        use_yahoo_tickers=True,
    )
    print(df.head())
    print(f"\nShape: {df.shape}")
  
