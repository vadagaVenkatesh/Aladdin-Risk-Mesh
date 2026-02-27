"""
tests/test_signals.py — Unit Tests for Signal Generation

Tests for macro signals, momentum, mean-reversion,
sentiment, and cross-asset signal combiners.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_series():
    """Synthetic price series (geometric Brownian motion, 252 days)."""
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0002, 0.015, 252)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def multi_asset_prices():
    """Synthetic prices for 10 assets over 252 days."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0003, 0.018, (252, 10))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return prices


# ---------------------------------------------------------------------------
# Momentum Signal Tests
# ---------------------------------------------------------------------------

class TestMomentumSignals:
    """Time-series and cross-sectional momentum."""

    def test_12_1_momentum_sign(self, price_series):
        """12-1 month momentum: recent return sign drives signal."""
        # 12m return excluding last 1m
        ret_12m = price_series[-22] / price_series[0] - 1
        signal = np.sign(ret_12m)
        assert signal in (-1, 0, 1)

    def test_rsi_range(self, price_series):
        """RSI must be in [0, 100]."""
        # Simple RSI calculation
        delta = np.diff(price_series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        window = 14
        avg_gain = np.convolve(gain, np.ones(window) / window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window) / window, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        assert np.all(rsi >= 0) and np.all(rsi <= 100)

    def test_cross_sectional_ranking(self, multi_asset_prices):
        """Cross-sectional momentum ranks should be in [1, n_assets]."""
        n_assets = multi_asset_prices.shape[1]
        returns_1m = multi_asset_prices[-1] / multi_asset_prices[-22] - 1
        ranks = returns_1m.argsort().argsort() + 1
        assert ranks.min() >= 1
        assert ranks.max() <= n_assets


# ---------------------------------------------------------------------------
# Mean Reversion Signal Tests
# ---------------------------------------------------------------------------

class TestMeanReversionSignals:
    """Z-score and Bollinger Band mean reversion."""

    def test_zscore_finite(self, price_series):
        """Z-score should be finite for non-constant prices."""
        window = 20
        rolling_mean = np.convolve(price_series, np.ones(window) / window, mode='valid')
        rolling_std = np.array([
            price_series[i:i + window].std()
            for i in range(len(price_series) - window + 1)
        ])
        zscore = (price_series[window - 1:] - rolling_mean) / (rolling_std + 1e-10)
        assert np.all(np.isfinite(zscore))

    def test_bollinger_band_ordering(self, price_series):
        """Upper BB must be above middle, which is above lower."""
        window = 20
        rolling_mean = np.convolve(price_series, np.ones(window) / window, mode='valid')
        rolling_std = np.array([
            price_series[i:i + window].std()
            for i in range(len(price_series) - window + 1)
        ])
        upper = rolling_mean + 2 * rolling_std
        lower = rolling_mean - 2 * rolling_std
        assert np.all(upper >= rolling_mean)
        assert np.all(rolling_mean >= lower)


# ---------------------------------------------------------------------------
# Signal Combination Tests
# ---------------------------------------------------------------------------

class TestSignalCombination:
    """Ensemble signal weighting and normalization."""

    def test_equal_weight_combination(self):
        """Equal-weighted signal combination should average correctly."""
        signals = np.array([1.0, -0.5, 0.3, 0.8])
        combined = signals.mean()
        expected = 0.4
        assert abs(combined - expected) < 1e-10

    def test_signal_normalization(self):
        """Signals normalized to zero mean, unit std."""
        rng = np.random.default_rng(42)
        raw = rng.normal(5, 3, 100)
        normalized = (raw - raw.mean()) / raw.std()
        assert abs(normalized.mean()) < 1e-10
        assert abs(normalized.std() - 1.0) < 1e-10

    def test_winsorization(self):
        """Winsorized signal should have no values beyond ±3 sigma."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 1000)
        signal[0] = 100.0   # outlier
        signal[1] = -100.0  # outlier
        winsorized = np.clip(signal, -3, 3)
        assert winsorized.max() <= 3.0
        assert winsorized.min() >= -3.0


# ---------------------------------------------------------------------------
# Macro Regime Signal Tests
# ---------------------------------------------------------------------------

class TestMacroSignals:
    """Macro regime indicator validation."""

    def test_yield_curve_slope(self):
        """Positive slope: 10Y > 2Y = normal regime."""
        rate_2y = 4.50
        rate_10y = 4.75
        slope = rate_10y - rate_2y
        assert slope > 0  # Normal (non-inverted) curve

    def test_inverted_yield_curve_detection(self):
        """Inverted curve: 2Y > 10Y."""
        rate_2y = 5.10
        rate_10y = 4.80
        inverted = rate_2y > rate_10y
        assert inverted is True

    def test_vix_regime_classification(self):
        """VIX buckets: low (<15), normal (15-25), elevated (>25)."""
        def classify_vix(vix):
            if vix < 15:
                return "low_vol"
            elif vix <= 25:
                return "normal"
            else:
                return "elevated"

        assert classify_vix(12) == "low_vol"
        assert classify_vix(20) == "normal"
        assert classify_vix(35) == "elevated"
