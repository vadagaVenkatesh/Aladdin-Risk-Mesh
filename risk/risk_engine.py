"""
Aladdin Risk Mesh — Numba-Optimized Risk Engine
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL
This source code is the exclusive intellectual property of VDG Venkatesh.
Unauthorized use, reproduction, distribution, or modification of this code,
in whole or in part, without the express written consent of VDG Venkatesh
is strictly prohibited and may result in civil and criminal penalties.
"""

import numpy as np
import pandas as pd
from numba import njit, prange
import logging

from environment.env_config import (
    CPU_CORES_NUMBA,
    VAR_CONFIDENCE,
    VAR_LOOKBACK_DAYS,
    NUMBA_NUM_THREADS,
)

log = logging.getLogger(__name__)

# Set Numba thread count
import numba
numba.set_num_threads(NUMBA_NUM_THREADS)


# ============================================================
# CORE VaR CALCULATION (NJIT-ACCELERATED)
# ============================================================
@njit(parallel=True, fastmath=True)
def _compute_var_hist_njit(
    returns: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """
    Compute historical VaR for a portfolio.

    Parameters
    ----------
    returns    : (T, N) array of asset returns
    weights    : (N,) portfolio weights
    confidence : VaR confidence level (0.99 = 99% VaR)

    Returns
    -------
    float : VaR (negative value = expected loss)
    """
    T, N = returns.shape
    portfolio_returns = np.empty(T, dtype=np.float64)

    # Matrix multiply: portfolio_returns = returns @ weights
    for t in prange(T):
        s = 0.0
        for i in range(N):
            s += returns[t, i] * weights[i]
        portfolio_returns[t] = s

    # Sort and compute quantile
    sorted_rets = np.sort(portfolio_returns)
    idx = int((1.0 - confidence) * T)
    return sorted_rets[idx]


@njit(parallel=True, fastmath=True)
def _compute_cvar_hist_njit(
    returns: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """
    Compute historical CVaR (Expected Shortfall) for a portfolio.

    Parameters
    ----------
    returns    : (T, N) array of asset returns
    weights    : (N,) portfolio weights
    confidence : CVaR confidence level

    Returns
    -------
    float : CVaR (mean of tail losses)
    """
    T, N = returns.shape
    portfolio_returns = np.empty(T, dtype=np.float64)

    for t in prange(T):
        s = 0.0
        for i in range(N):
            s += returns[t, i] * weights[i]
        portfolio_returns[t] = s

    sorted_rets = np.sort(portfolio_returns)
    cutoff_idx = int((1.0 - confidence) * T)
    tail = sorted_rets[:cutoff_idx]
    return np.mean(tail)


# ============================================================
# COVARIANCE MATRIX (PARALLEL)
# ============================================================
@njit(parallel=True, fastmath=True)
def _compute_cov_njit(returns: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix using parallel loops.

    Parameters
    ----------
    returns : (T, N) array

    Returns
    -------
    (N, N) covariance matrix
    """
    T, N = returns.shape
    means = np.empty(N, dtype=np.float64)

    # Compute means
    for i in prange(N):
        s = 0.0
        for t in range(T):
            s += returns[t, i]
        means[i] = s / T

    # Compute covariance
    cov = np.zeros((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(i, N):
            s = 0.0
            for t in range(T):
                s += (returns[t, i] - means[i]) * (returns[t, j] - means[j])
            c = s / (T - 1)
            cov[i, j] = c
            cov[j, i] = c  # symmetry

    return cov


# ============================================================
# PORTFOLIO VOLATILITY (NJIT)
# ============================================================
@njit(fastmath=True)
def _compute_portfolio_vol_njit(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Portfolio volatility: sqrt(w^T * Cov * w)

    Parameters
    ----------
    weights : (N,) array
    cov     : (N, N) covariance matrix

    Returns
    -------
    float : annualized volatility
    """
    N = len(weights)
    variance = 0.0
    for i in range(N):
        for j in range(N):
            variance += weights[i] * cov[i, j] * weights[j]
    return np.sqrt(variance * 252.0)  # annualized


# ============================================================
# MARGINAL VAR (PARALLEL)
# ============================================================
@njit(parallel=True, fastmath=True)
def _compute_marginal_var_njit(
    returns: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    confidence: float = 0.99,
) -> np.ndarray:
    """
    Compute marginal VaR: ∂VaR/∂w_i (approximated by finite difference).

    Parameters
    ----------
    returns    : (T, N) returns
    weights    : (N,) portfolio weights
    cov        : (N, N) covariance
    confidence : VaR confidence

    Returns
    -------
    (N,) marginal VaR per asset
    """
    N = len(weights)
    base_var = _compute_var_hist_njit(returns, weights, confidence)
    marginal = np.empty(N, dtype=np.float64)
    epsilon = 1e-5

    for i in prange(N):
        w_perturb = weights.copy()
        w_perturb[i] += epsilon
        w_perturb = w_perturb / np.sum(w_perturb)  # re-normalize
        var_perturb = _compute_var_hist_njit(returns, w_perturb, confidence)
        marginal[i] = (var_perturb - base_var) / epsilon

    return marginal


# ============================================================
# COMPONENT VAR (PARALLEL)
# ============================================================
@njit(parallel=True, fastmath=True)
def _compute_component_var_njit(
    marginal_var: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Component VaR: w_i * MarginalVaR_i

    Parameters
    ----------
    marginal_var : (N,) marginal VaR
    weights      : (N,) portfolio weights

    Returns
    -------
    (N,) component VaR per asset
    """
    N = len(weights)
    comp = np.empty(N, dtype=np.float64)
    for i in prange(N):
        comp[i] = weights[i] * marginal_var[i]
    return comp


# ============================================================
# HIGH-LEVEL INTERFACE
# ============================================================
def compute_risk_metrics(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    confidence: float = VAR_CONFIDENCE,
) -> dict:
    """
    Compute a comprehensive risk report for a portfolio.

    Parameters
    ----------
    returns_df : DataFrame with columns = asset tickers, index = dates
    weights    : Series with index = tickers, values = portfolio weights
    confidence : VaR confidence level

    Returns
    -------
    dict with keys:
        'var'             : Portfolio VaR
        'cvar'            : Portfolio CVaR
        'volatility'      : Annualized portfolio volatility
        'cov'             : Covariance matrix (DataFrame)
        'marginal_var'    : Marginal VaR (Series)
        'component_var'   : Component VaR (Series)
    """
    # Align weights with returns
    weights = weights.reindex(returns_df.columns, fill_value=0.0)
    returns_np = returns_df.values.astype(np.float64)
    weights_np = weights.values.astype(np.float64)

    # Compute covariance
    cov_np = _compute_cov_njit(returns_np)
    cov_df = pd.DataFrame(cov_np, index=returns_df.columns, columns=returns_df.columns)

    # VaR & CVaR
    var = _compute_var_hist_njit(returns_np, weights_np, confidence)
    cvar = _compute_cvar_hist_njit(returns_np, weights_np, confidence)

    # Portfolio volatility
    vol = _compute_portfolio_vol_njit(weights_np, cov_np)

    # Marginal VaR
    marginal_var_np = _compute_marginal_var_njit(returns_np, weights_np, cov_np, confidence)
    marginal_var = pd.Series(marginal_var_np, index=returns_df.columns)

    # Component VaR
    component_var_np = _compute_component_var_njit(marginal_var_np, weights_np)
    component_var = pd.Series(component_var_np, index=returns_df.columns)

    log.info(
        f"[RiskEngine] VaR={var:.4f} CVaR={cvar:.4f} Vol={vol:.2%} "
        f"(confidence={confidence}, T={len(returns_df)})"
    )

    return {
        "var": var,
        "cvar": cvar,
        "volatility": vol,
        "cov": cov_df,
        "marginal_var": marginal_var,
        "component_var": component_var,
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Toy example
    np.random.seed(42)
    T = 252
    N = 1000
    rets = np.random.randn(T, N) * 0.01
    returns_df = pd.DataFrame(rets, columns=[f"Asset{i}" for i in range(N)])
    weights = pd.Series(1.0 / N, index=returns_df.columns)

    metrics = compute_risk_metrics(returns_df, weights, confidence=0.99)
    print(f"VaR (99%): {metrics['var']:.4f}")
    print(f"CVaR (99%): {metrics['cvar']:.4f}")
    print(f"Volatility: {metrics['volatility']:.2%}")
