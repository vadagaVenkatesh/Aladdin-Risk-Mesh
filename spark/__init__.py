"""
spark/ — Distributed Computing & Large-Scale Data Processing

Provides PySpark-based distributed pipelines for large-scale
market data ingestion, risk aggregation across portfolios,
and parallel Monte Carlo simulation at enterprise scale.

Optimized for Azure Databricks and HDInsight environments.
"""

from .risk_aggregator import SparkRiskAggregator
from .market_loader import SparkMarketLoader

__all__ = ["SparkRiskAggregator", "SparkMarketLoader"]
