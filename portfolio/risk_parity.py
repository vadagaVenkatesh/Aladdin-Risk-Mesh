"""Risk Parity and Hierarchical Risk Parity (HRP) for Aladdin-Risk-Mesh.

Implements allocation strategies that focus on risk contribution:
- Naive Risk Parity (Inverse Volatility)
- Equal Risk Contribution (ERC)
- Hierarchical Risk Parity (HRP) based on Lopez de Prado
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import minimize
from typing import List, Optional


class RiskParityEngine:
    """
    Risk-based allocation engine.

    Focuses on equalizing risk contributions across assets
    rather than maximizing expected returns.
    """

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.cov = returns.cov() * 252
        self.assets = returns.columns.tolist()
        self.n = len(self.assets)

    # ------------------------------------------------------------------
    # Inverse Volatility (Naive Risk Parity)
    # ------------------------------------------------------------------

    def inverse_volatility_weights(self) -> pd.Series:
        """Weights proportional to 1/volatility."""
        vols = np.sqrt(np.diag(self.cov))
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)
        return pd.Series(weights, index=self.assets)

    # ------------------------------------------------------------------
    # Equal Risk Contribution (ERC)
    # ------------------------------------------------------------------

    def equal_risk_contribution_weights(self) -> pd.Series:
        """
        Weights such that each asset contributes equally to portfolio volatility.
        Solves: min sum_i (RC_i - 1/n)^2
        """
        initial_weights = np.ones(self.n) / self.n
        bounds = [(0.0, 1.0) for _ in range(self.n)]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        def objective(w):
            p_vol = np.sqrt(w.T @ self.cov @ w)
            # Risk Contribution = w_i * (Cov * w)_i / p_vol
            rc = w * (self.cov @ w) / p_vol
            target_rc = p_vol / self.n
            return np.sum((rc - target_rc)**2)

        res = minimize(objective, initial_weights, bounds=bounds, constraints=cons, tol=1e-10)
        return pd.Series(res.x, index=self.assets)

    # ------------------------------------------------------------------
    # Hierarchical Risk Parity (HRP)
    # ------------------------------------------------------------------

    def hrp_weights(self) -> pd.Series:
        """
        Hierarchical Risk Parity (Lopez de Prado).
        Uses clustering to handle high-correlation assets.
        """
        corr = self.returns.corr()
        # 1. Quasi-Diagonalization
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(dist, 'single')
        sort_ix = self._get_quasi_diag(link)
        sorted_assets = [self.assets[i] for i in sort_ix]

        # 2. Recursive Bisection
        weights = pd.Series(1.0, index=sorted_assets)
        items = [sorted_assets]

        while len(items) > 0:
            items = [i[j:k] for i in items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(items), 2):
                left_cluster = items[i]
                right_cluster = items[i+1]

                left_cov = self.cov.loc[left_cluster, left_cluster]
                right_cov = self.cov.loc[right_cluster, right_cluster]

                left_var = self._get_cluster_var(left_cov)
                right_var = self._get_cluster_var(right_cov)

                alpha = 1 - left_var / (left_var + right_var)
                weights[left_cluster] *= alpha
                weights[right_cluster] *= (1 - alpha)

        return weights.reindex(self.assets)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Helper for HRP: Sorts assets based on hierarchical clustering."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            clusters = sort_ix[sort_ix >= num_items]
            i = clusters.index
            j = clusters.values - num_items
            sort_ix[i] = link[j, 0]
            df = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def _get_cluster_var(self, cluster_cov: pd.DataFrame) -> float:
        """Helper for HRP: Inverse-variance allocation variance."""
        w = 1.0 / np.diag(cluster_cov)
        w /= w.sum()
        return w.T @ cluster_cov @ w


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assets = ["Asset_A", "Asset_B", "Asset_C", "Asset_D"]
    # B and C are highly correlated
    np.random.seed(42)
    base = np.random.normal(0, 0.01, (1000, 1))
    data = pd.DataFrame({
        "Asset_A": np.random.normal(0.0005, 0.01, 1000),
        "Asset_B": base[:, 0] + np.random.normal(0, 0.002, 1000),
        "Asset_C": base[:, 0] + np.random.normal(0, 0.002, 1000),
        "Asset_D": np.random.normal(0.0003, 0.015, 1000)
    })

    engine = RiskParityEngine(data)

    print("Inverse Volatility:")
    print(engine.inverse_volatility_weights().round(4))

    print("
Equal Risk Contribution:")
    print(engine.equal_risk_contribution_weights().round(4))

    print("
Hierarchical Risk Parity:")
    print(engine.hrp_weights().round(4))
