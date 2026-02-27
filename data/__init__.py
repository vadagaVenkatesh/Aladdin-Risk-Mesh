"""data — Market Data Layer

Handles Bloomberg (primary) and Yahoo Finance (fallback) ingestion.
All data is cached as Parquet files for fast re-reads.
Supports equities, FX, rates, commodities, and crypto.
"""
from data.loader import DataLoader

__all__ = ["DataLoader"]
