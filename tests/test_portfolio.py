import unittest
import numpy as np
from portfolio.optimizer import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        self.returns = np.array([0.12, 0.15, 0.10, 0.08])
        self.cov_matrix = np.eye(4) * 0.04
        self.optimizer = PortfolioOptimizer(self.tickers, self.returns, self.cov_matrix)

    def test_initialization(self):
        self.assertEqual(len(self.optimizer.tickers), 4)
        self.assertEqual(self.optimizer.returns.shape, (4,))
        self.assertEqual(self.optimizer.cov_matrix.shape, (4, 4))

    def test_mvo_optimization(self):
        weights = self.optimizer.optimize_mvo(target_return=0.10)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0))

    def test_black_litterman(self):
        views = {"AAPL": 0.14, "GOOGL": 0.11}
        bl_returns, bl_cov = self.optimizer.apply_black_litterman(views)
        self.assertEqual(bl_returns.shape, (4,))
        self.assertEqual(bl_cov.shape, (4, 4))

    def test_risk_parity(self):
        weights = self.optimizer.optimize_risk_parity()
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        # In equal variance case, weights should be roughly equal
        for w in weights:
            self.assertAlmostEqual(w, 0.25, places=2)

if __name__ == "__main__":
    unittest.main()
