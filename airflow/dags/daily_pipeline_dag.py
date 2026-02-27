"""
Aladdin Risk Mesh — Daily Pipeline DAG (6:00 AM EST)
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import logging

from environment.env_config import (
    AIRFLOW_DAILY_RUN_HOUR_EST,
    AIRFLOW_DAILY_RUN_MINUTE_EST,
    AIRFLOW_TIMEZONE,
)

log = logging.getLogger(__name__)

# ===========================================================
# TASK FUNCTIONS
# ===========================================================

def task_load_market_data():
    """Load daily market data from Bloomberg/Yahoo."""
    from data.loaders.market_data_loader import MarketDataLoader
    from core.data_bus import write, init_bus
    
    init_bus()
    loader = MarketDataLoader()
    
    # Define universe (example: top 100 liquid assets)
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO"]  # expand to 10k in production
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df = loader.fetch(tickers, start_date, end_date, use_yahoo_tickers=True)
    write("signals", df, filename="market_prices.parquet")
    log.info(f"[DAG] Loaded {len(df)} rows x {len(tickers)} tickers")


def task_generate_signals():
    """Generate momentum/macro signals."""
    from core.data_bus import read, write
    from signals.momentum import TimeSeriesMomentum
    
    prices = read("signals", filename="market_prices.parquet")
    tsmom = TimeSeriesMomentum(prices)
    composite = tsmom.composite_signal()
    
    write("signals", composite, filename="tsmom_signals.parquet")
    log.info(f"[DAG] Generated TSMOM signals for {len(composite.columns)} assets")


def task_compute_risk():
    """Compute risk metrics with Numba-optimized engine."""
    from core.data_bus import read, write
    from risk.risk_engine import compute_risk_metrics
    import pandas as pd
    
    prices = read("signals", filename="market_prices.parquet")
    returns = prices.pct_change().dropna()
    weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    
    risk_metrics = compute_risk_metrics(returns, weights, confidence=0.99)
    
    # Save results
    write("risk", risk_metrics["cov"], filename="covariance.parquet")
    write("risk", risk_metrics["marginal_var"].to_frame(), filename="marginal_var.parquet")
    log.info(f"[DAG] Computed risk: VaR={risk_metrics['var']:.4f}")


def task_optimize_portfolio():
    """Run Black-Litterman portfolio optimization."""
    from core.data_bus import read, write
    from portfolio.optimizer import BlackLittermanOptimizer
    import pandas as pd
    
    cov = read("risk", filename="covariance.parquet")
    signals = read("signals", filename="tsmom_signals.parquet")
    
    # Use latest signal as views
    views = signals.iloc[-1]
    
    optimizer = BlackLittermanOptimizer(cov, views)
    weights = optimizer.optimize()
    
    write("portfolio", weights.to_frame(name="weight"), filename="optimal_weights.parquet")
    log.info(f"[DAG] Optimized portfolio: top asset={weights.idxmax()} ({weights.max():.2%})")


def task_run_monte_carlo():
    """Run Spark-distributed Monte Carlo simulations."""
    from core.data_bus import read, write
    from spark.monte_carlo_spark import run_monte_carlo_portfolio, compute_mc_statistics
    
    weights = read("portfolio", filename="optimal_weights.parquet")["weight"]
    cov = read("risk", filename="covariance.parquet")
    
    mc_results = run_monte_carlo_portfolio(weights, cov, n_simulations=10_000, horizon_days=252)
    stats = compute_mc_statistics(mc_results)
    
    write("backtest", mc_results, filename="monte_carlo_paths.parquet")
    log.info(f"[DAG] Monte Carlo complete: VaR99={stats['var_99']:.4f}")


def task_persist_state():
    """Persist MVO/Kelly state for restart survival."""
    from core.data_bus import read, persist_mvo, persist_kelly
    
    weights = read("portfolio", filename="optimal_weights.parquet")["weight"]
    persist_mvo(weights)
    persist_kelly(weights * 0.25)  # fractional Kelly
    log.info("[DAG] Persisted MVO/Kelly state to Parquet bus")


# ===========================================================
# DAG DEFINITION
# ===========================================================

default_args = {
    "owner": "VDG_Venkatesh",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="aladdin_daily_pipeline",
    default_args=default_args,
    description="Daily 6AM EST: Load data, generate signals, optimize portfolio, run Monte Carlo",
    schedule_interval=f"{AIRFLOW_DAILY_RUN_MINUTE_EST} {AIRFLOW_DAILY_RUN_HOUR_EST} * * *",  # 6:00 AM EST daily
    start_date=datetime(2026, 1, 1, tzinfo=None),
    catchup=False,
    tags=["aladdin", "production", "daily"],
)

# Task instances
t1 = PythonOperator(task_id="load_market_data", python_callable=task_load_market_data, dag=dag)
t2 = PythonOperator(task_id="generate_signals", python_callable=task_generate_signals, dag=dag)
t3 = PythonOperator(task_id="compute_risk", python_callable=task_compute_risk, dag=dag)
t4 = PythonOperator(task_id="optimize_portfolio", python_callable=task_optimize_portfolio, dag=dag)
t5 = PythonOperator(task_id="run_monte_carlo", python_callable=task_run_monte_carlo, dag=dag)
t6 = PythonOperator(task_id="persist_state", python_callable=task_persist_state, dag=dag)

# Task dependencies
t1 >> t2 >> t3 >> t4 >> t5 >> t6
