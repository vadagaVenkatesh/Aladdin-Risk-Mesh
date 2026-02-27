"""
Aladdin Risk Mesh — Environment Configuration
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL
This source code is the exclusive intellectual property of VDG Venkatesh.
Unauthorized use, reproduction, distribution, or modification of this code,
in whole or in part, without the express written consent of VDG Venkatesh
is strictly prohibited and may result in civil and criminal penalties.

This software is provided solely for authorized use within the Aladdin Risk
Mesh project. No license, express or implied, is granted to any third party.
"""

import os
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# AZURE INFRASTRUCTURE — Azure D16s_v5 (16 vCPUs, 64 GB RAM)
# ============================================================
AZURE_VM_SKU               = "Standard_D16s_v5"
AZURE_TOTAL_VCPUS          = 16
AZURE_RAM_GB               = 64
AZURE_REGION               = "eastus"

# CPU parallelism — leave 2 cores for OS/system overhead
CPU_CORES_TOTAL            = multiprocessing.cpu_count()        # auto-detect
CPU_CORES_WORKER           = max(1, CPU_CORES_TOTAL - 2)        # 14 workers
CPU_CORES_NUMBA            = CPU_CORES_WORKER                   # @njit(parallel=True)
CPU_CORES_SPARK            = CPU_CORES_WORKER                   # Spark local[N]
CPU_CORES_AIRFLOW          = 4                                  # reserved for Airflow

# Thread tuning
NUMBA_NUM_THREADS          = CPU_CORES_NUMBA
NUMPY_NUM_THREADS          = CPU_CORES_WORKER

# ============================================================
# GPU — NOT AVAILABLE ON THIS INSTANCE (Azure D16s_v5 = CPU only)
# ============================================================
GPU_AVAILABLE              = False
# import cupy as cp                    # FUTURE: enable when GPU instance is provisioned
# CUDA_DEVICE_ID             = 0
# CUDA_MEMORY_POOL_GB        = 16

# ============================================================
# DATA SOURCES
# ============================================================
# Primary: Bloomberg (production)
# Fallback: Yahoo Finance (dev/testing/Bloomberg unavailable)

BLOOMBERG_ENABLED          = bool(os.getenv("BLOOMBERG_ENABLED", "false").lower() == "true")
BLOOMBERG_HOST             = os.getenv("BLOOMBERG_HOST", "localhost")
BLOOMBERG_PORT             = int(os.getenv("BLOOMBERG_PORT", 8194))
BLOOMBERG_TIMEOUT_SEC      = 30

YAHOO_FINANCE_ENABLED      = True                               # always-on fallback
YAHOO_INTRADAY_INTERVAL    = "1m"                              # 1-minute bars
YAHOO_LOOKBACK_DAYS        = 30                                # intraday history window

FRED_API_KEY               = os.getenv("FRED_API_KEY", "")     # FRED macro data
QUANTLIB_ENABLED           = True                              # QuantLib for derivatives

# Maximum number of assets in the universe
MAX_ASSETS                 = 10_000

# ============================================================
# STRATEGIES (run simultaneously)
# ============================================================
STRATEGY_GLOBAL_MACRO      = True
STRATEGY_LONG_SHORT_EQUITY = True

# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================
PORTFOLIO_METHOD           = "BLACK_LITTERMAN"                  # primary optimizer
MVO_ENABLED                = True                               # also run MVO in parallel
KELLY_ENABLED              = True                               # Kelly criterion sizing

# Black-Litterman parameters
BL_RISK_AVERSION           = 2.5
BL_TAU                     = 0.05

# ============================================================
# MONTE CARLO
# ============================================================
MONTE_CARLO_SIMULATIONS    = 10_000
MONTE_CARLO_HORIZON_DAYS   = 252
MONTE_CARLO_USE_SPARK      = True                              # Spark-distributed MC
MONTE_CARLO_SEED           = 42

# ============================================================
# AIRFLOW SCHEDULING
# ============================================================
AIRFLOW_DAILY_RUN_HOUR_EST     = 6                             # 6:00 AM EST (hard-coded)
AIRFLOW_DAILY_RUN_MINUTE_EST   = 0
AIRFLOW_TIMEZONE               = "America/New_York"

AIRFLOW_BLOOMBERG_INTERVAL_MIN = 5                             # every 5 min Bloomberg intraday
AIRFLOW_BLOOMBERG_START_HOUR   = 9                             # 9:30 AM EST market open
AIRFLOW_BLOOMBERG_START_MIN    = 30
AIRFLOW_BLOOMBERG_END_HOUR     = 16                            # 4:00 PM EST market close
AIRFLOW_BLOOMBERG_END_MIN      = 0

# ============================================================
# SPARK CONFIGURATION
# ============================================================
SPARK_MASTER               = f"local[{CPU_CORES_SPARK}]"
SPARK_APP_NAME             = "AladdinRiskMesh"
SPARK_DRIVER_MEMORY        = "24g"                             # ~40% of 64 GB RAM
SPARK_EXECUTOR_MEMORY      = "24g"
SPARK_SHUFFLE_PARTITIONS   = CPU_CORES_SPARK * 4
SPARK_DEFAULT_PARALLELISM  = CPU_CORES_SPARK * 2
SPARK_LOG_LEVEL            = "WARN"

# ============================================================
# PARQUET DATA BUS — inter-module communication
# ============================================================
PARQUET_BASE_DIR           = os.getenv("PARQUET_BASE_DIR", "/tmp/aladdin_bus")
PARQUET_SIGNALS_PATH       = f"{PARQUET_BASE_DIR}/signals"
PARQUET_RISK_PATH          = f"{PARQUET_BASE_DIR}/risk"
PARQUET_PORTFOLIO_PATH     = f"{PARQUET_BASE_DIR}/portfolio"
PARQUET_REGIME_PATH        = f"{PARQUET_BASE_DIR}/regime"
PARQUET_EXECUTION_PATH     = f"{PARQUET_BASE_DIR}/execution"
PARQUET_BACKTEST_PATH      = f"{PARQUET_BASE_DIR}/backtest"
PARQUET_MVO_PATH           = f"{PARQUET_BASE_DIR}/mvo"        # persisted between runs
PARQUET_KELLY_PATH         = f"{PARQUET_BASE_DIR}/kelly"      # persisted between runs
PARQUET_COMPRESSION        = "snappy"                          # fast read/write

# ============================================================
# RISK ENGINE
# ============================================================
VAR_CONFIDENCE             = 0.99
VAR_LOOKBACK_DAYS          = 252
STRESS_SCENARIOS_ENABLED   = True
BASEL_IV_ENABLED           = True
FRTB_ENABLED               = True
KELLY_FRACTION             = 0.25                              # fractional Kelly

# ============================================================
# EXECUTION
# ============================================================
EXECUTION_INTRADAY_ENABLED = True
INTRADAY_BAR_INTERVAL      = "1m"
EXECUTION_MAX_SLIPPAGE_BPS = 5
EXECUTION_MAX_IMPACT_BPS   = 10

# ============================================================
# LOGGING & MONITORING
# ============================================================
LOG_LEVEL                  = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR                    = os.getenv("LOG_DIR", "./logs")
MONITOR_DRAWDOWN_LIMIT     = -0.10                             # -10% max drawdown alert
MONITOR_VAR_BREACH_LIMIT   = 0.05                             # 5% VaR breach threshold

# ============================================================
# DOCKER / AIRFLOW SERVICE PORTS
# ============================================================
AIRFLOW_WEBSERVER_PORT     = 8080
FASTAPI_PORT               = 8000
FLOWER_PORT                = 5555                              # Celery Flower monitoring
REDIS_PORT                 = 6379
POSTGRES_PORT              = 5432

# ============================================================
# ENVIRONMENT DATACLASS (typed access)
# ============================================================
@dataclass
class AladdinEnvConfig:
    """Typed environment configuration for Aladdin Risk Mesh."""
    # Azure compute
    azure_vm_sku: str              = AZURE_VM_SKU
    cpu_cores_total: int           = CPU_CORES_TOTAL
    cpu_cores_worker: int          = CPU_CORES_WORKER
    gpu_available: bool            = GPU_AVAILABLE

    # Data
    bloomberg_enabled: bool        = BLOOMBERG_ENABLED
    yahoo_enabled: bool            = YAHOO_FINANCE_ENABLED
    max_assets: int                = MAX_ASSETS
    fred_api_key: str              = FRED_API_KEY

    # Strategies
    strategy_global_macro: bool    = STRATEGY_GLOBAL_MACRO
    strategy_ls_equity: bool       = STRATEGY_LONG_SHORT_EQUITY

    # Portfolio
    portfolio_method: str          = PORTFOLIO_METHOD
    bl_risk_aversion: float        = BL_RISK_AVERSION
    bl_tau: float                  = BL_TAU

    # Monte Carlo
    mc_simulations: int            = MONTE_CARLO_SIMULATIONS
    mc_use_spark: bool             = MONTE_CARLO_USE_SPARK

    # Airflow
    airflow_daily_hour: int        = AIRFLOW_DAILY_RUN_HOUR_EST
    airflow_bbg_interval: int      = AIRFLOW_BLOOMBERG_INTERVAL_MIN

    # Parquet bus
    parquet_base_dir: str          = PARQUET_BASE_DIR
    parquet_compression: str       = PARQUET_COMPRESSION

    # Risk
    var_confidence: float          = VAR_CONFIDENCE
    kelly_fraction: float          = KELLY_FRACTION


# Singleton config instance
ENV = AladdinEnvConfig()


def get_spark_config() -> dict:
    """Return PySpark configuration dict."""
    return {
        "spark.master":                        SPARK_MASTER,
        "spark.app.name":                      SPARK_APP_NAME,
        "spark.driver.memory":                 SPARK_DRIVER_MEMORY,
        "spark.executor.memory":               SPARK_EXECUTOR_MEMORY,
        "spark.sql.shuffle.partitions":        str(SPARK_SHUFFLE_PARTITIONS),
        "spark.default.parallelism":           str(SPARK_DEFAULT_PARALLELISM),
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.serializer":                    "org.apache.spark.serializer.KryoSerializer",
    }


def print_env_summary() -> None:
    """Print a summary of the active environment configuration."""
    print("=" * 60)
    print("  Aladdin Risk Mesh — Environment Summary")
    print("=" * 60)
    print(f"  VM SKU         : {AZURE_VM_SKU}")
    print(f"  CPU Cores      : {CPU_CORES_TOTAL} total / {CPU_CORES_WORKER} workers")
    print(f"  GPU Available  : {GPU_AVAILABLE}")
    print(f"  Bloomberg      : {BLOOMBERG_ENABLED}")
    print(f"  Max Assets     : {MAX_ASSETS:,}")
    print(f"  MC Simulations : {MONTE_CARLO_SIMULATIONS:,}")
    print(f"  Spark Master   : {SPARK_MASTER}")
    print(f"  Airflow Daily  : {AIRFLOW_DAILY_RUN_HOUR_EST}:00 EST")
    print(f"  BBG Interval   : every {AIRFLOW_BLOOMBERG_INTERVAL_MIN} min")
    print(f"  Parquet Bus    : {PARQUET_BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_env_summary()
