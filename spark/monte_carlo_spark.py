"""
Aladdin Risk Mesh — Spark-Distributed Monte Carlo Engine
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL
This source code is the exclusive intellectual property of VDG Venkatesh.
Unauthorized use, reproduction, distribution, or modification of this code,
in whole or in part, without the express written consent of VDG Venkatesh
is strictly prohibited and may result in civil and criminal penalties.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType

from environment.env_config import (
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_HORIZON_DAYS,
    MONTE_CARLO_SEED,
    CPU_CORES_SPARK,
    get_spark_config,
)

log = logging.getLogger(__name__)


# ============================================================
# SPARK SESSION INITIALIZATION
# ============================================================
def get_or_create_spark() -> SparkSession:
    """Initialize or return existing Spark session with optimized config."""
    spark_conf = get_spark_config()
    builder = SparkSession.builder
    for key, val in spark_conf.items():
        builder = builder.config(key, val)
    spark = builder.getOrCreate()
    log.info(f"[Spark] Session active: {spark.sparkContext.appName} | {spark.sparkContext.master}")
    return spark


# ============================================================
# MONTE CARLO SIMULATION (DISTRIBUTED)
# ============================================================
def run_monte_carlo_portfolio(
    portfolio_weights: pd.Series,
    returns_cov: pd.DataFrame,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
    horizon_days: int = MONTE_CARLO_HORIZON_DAYS,
    seed: int = MONTE_CARLO_SEED,
) -> pd.DataFrame:
    """
    Run Monte Carlo portfolio simulations using PySpark for distribution.

    Parameters
    ----------
    portfolio_weights : Series with asset weights
    returns_cov       : Covariance matrix (N x N)
    n_simulations     : Number of Monte Carlo paths (default: 10,000)
    horizon_days      : Simulation horizon in trading days (default: 252)
    seed              : Random seed for reproducibility

    Returns
    -------
    DataFrame with columns: [simulation_id, day, portfolio_value, portfolio_return]
    """
    spark = get_or_create_spark()

    # Align inputs
    assets = portfolio_weights.index.tolist()
    weights_np = portfolio_weights.values
    cov_np = returns_cov.loc[assets, assets].values

    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(cov_np)

    # Broadcast parameters to all Spark workers
    bc_weights = spark.sparkContext.broadcast(weights_np)
    bc_L = spark.sparkContext.broadcast(L)
    bc_horizon = spark.sparkContext.broadcast(horizon_days)
    bc_seed = spark.sparkContext.broadcast(seed)

    log.info(
        f"[Spark MC] Starting {n_simulations} simulations | "
        f"{len(assets)} assets | {horizon_days} days horizon"
    )

    # Create RDD of simulation IDs
    sim_ids = spark.sparkContext.range(n_simulations, numSlices=CPU_CORES_SPARK * 4)

    def simulate_path(sim_id: int) -> list:
        """Simulate one Monte Carlo path."""
        np.random.seed(bc_seed.value + sim_id)  # unique seed per sim
        weights = bc_weights.value
        L_chol = bc_L.value
        horizon = bc_horizon.value

        portfolio_value = 1.0
        results = []

        for day in range(horizon):
            # Generate correlated asset returns: L @ Z
            Z = np.random.randn(len(weights))
            asset_returns = L_chol @ Z

            # Portfolio return = w^T * r
            port_return = np.dot(weights, asset_returns)
            portfolio_value *= (1.0 + port_return)

            results.append((int(sim_id), day, float(portfolio_value), float(port_return)))

        return results

    # Run distributed simulations
    rdd_results = sim_ids.flatMap(simulate_path)

    # Convert to Spark DataFrame
    schema = StructType([
        StructField("simulation_id", IntegerType(), False),
        StructField("day", IntegerType(), False),
        StructField("portfolio_value", DoubleType(), False),
        StructField("portfolio_return", DoubleType(), False),
    ])
    df_spark = spark.createDataFrame(rdd_results, schema=schema)

    # Convert to pandas
    df_pandas = df_spark.toPandas()

    log.info(
        f"[Spark MC] Completed {n_simulations} simulations | "
        f"{len(df_pandas)} total rows"
    )

    return df_pandas


# ============================================================
# MONTE CARLO STATISTICS
# ============================================================
def compute_mc_statistics(mc_results: pd.DataFrame) -> dict:
    """
    Compute summary statistics from Monte Carlo simulation results.

    Parameters
    ----------
    mc_results : DataFrame with [simulation_id, day, portfolio_value, portfolio_return]

    Returns
    -------
    dict with keys:
        'mean_final_value'       : Mean portfolio value at horizon
        'median_final_value'     : Median portfolio value at horizon
        'var_95'                 : 95% VaR (5th percentile loss)
        'var_99'                 : 99% VaR (1st percentile loss)
        'max_drawdown'           : Maximum drawdown across all paths
        'prob_loss'              : Probability of loss at horizon
        'percentiles'            : 1%, 5%, 25%, 50%, 75%, 95%, 99%
    """
    # Final values (last day of each simulation)
    final_values = mc_results.groupby("simulation_id")["portfolio_value"].last()

    # VaR (percentile of losses)
    var_95 = np.percentile(final_values, 5)   # 5th percentile
    var_99 = np.percentile(final_values, 1)   # 1st percentile

    # Percentiles
    percentiles = {
        "p01": np.percentile(final_values, 1),
        "p05": np.percentile(final_values, 5),
        "p25": np.percentile(final_values, 25),
        "p50": np.percentile(final_values, 50),
        "p75": np.percentile(final_values, 75),
        "p95": np.percentile(final_values, 95),
        "p99": np.percentile(final_values, 99),
    }

    # Maximum drawdown (per simulation)
    def max_dd(sim_df):
        vals = sim_df["portfolio_value"].values
        peak = np.maximum.accumulate(vals)
        dd = (vals - peak) / peak
        return dd.min()

    max_drawdowns = mc_results.groupby("simulation_id").apply(max_dd)
    worst_drawdown = max_drawdowns.min()

    # Probability of loss
    prob_loss = (final_values < 1.0).mean()

    stats = {
        "mean_final_value": final_values.mean(),
        "median_final_value": final_values.median(),
        "var_95": var_95,
        "var_99": var_99,
        "max_drawdown": worst_drawdown,
        "prob_loss": prob_loss,
        "percentiles": percentiles,
    }

    log.info(
        f"[MC Stats] Mean={stats['mean_final_value']:.4f} | "
        f"VaR95={var_95:.4f} | VaR99={var_99:.4f} | "
        f"MaxDD={worst_drawdown:.2%} | ProbLoss={prob_loss:.2%}"
    )

    return stats


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Toy example
    np.random.seed(42)
    n_assets = 100
    weights = pd.Series(1.0 / n_assets, index=[f"Asset{i}" for i in range(n_assets)])
    cov = pd.DataFrame(
        np.eye(n_assets) * 0.0001,  # diagonal covariance
        index=weights.index,
        columns=weights.index,
    )

    # Run Monte Carlo (small scale for testing)
    mc_df = run_monte_carlo_portfolio(
        portfolio_weights=weights,
        returns_cov=cov,
        n_simulations=100,  # small test
        horizon_days=21,    # 1 month
    )

    stats = compute_mc_statistics(mc_df)
    print("\nMonte Carlo Stats:")
    for k, v in stats.items():
        if k != "percentiles":
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print("\nPercentiles:")
    for k, v in stats["percentiles"].items():
        print(f"  {k}: {v:.4f}")
