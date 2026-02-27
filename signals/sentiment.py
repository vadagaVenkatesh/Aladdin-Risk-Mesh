"""Sentiment Signal Module for Aladdin-Risk-Mesh.

Provides market sentiment indicators:
- Fear & Greed composite index
- Options market sentiment (put/call ratio, skew, term structure)
- Positioning data (COT reports, short interest)
- Retail vs institutional flow divergence
- News/text sentiment proxy via vader-style scoring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SentimentConfig:
    """Configuration for sentiment signal generation."""
    smoothing_span: int = 10            # EWM span for raw signal smoothing
    z_score_window: int = 252           # Lookback for z-score normalization
    signal_clip: float = 3.0            # Max signal magnitude
    cot_lookback: int = 52              # Weeks lookback for COT normalization
    skew_window: int = 21               # Rolling window for vol skew signal
    pcr_window: int = 21                # Put/call ratio smoothing window


class SentimentSignal:
    """
    Market sentiment signal aggregator.

    Combines options-market, positioning, and flow data into
    contrarian and momentum sentiment scores.

    High sentiment score => crowded long / peak euphoria => potential reversal
    Low sentiment score => extreme fear => potential mean-reversion long entry

    Parameters
    ----------
    config : SentimentConfig
        Engine configuration.
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._signals: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Options Market Sentiment
    # ------------------------------------------------------------------

    def put_call_ratio_signal(self, put_volume: pd.Series, call_volume: pd.Series) -> pd.Series:
        """
        Put/call ratio signal.

        High PCR => bearish fear (contrarian bullish)
        Low PCR  => complacency (contrarian bearish)

        Returns z-scored PCR (positive = elevated fear = contrarian buy).
        """
        cfg = self.config
        pcr = put_volume / call_volume.replace(0, np.nan)
        pcr_smooth = pcr.rolling(cfg.pcr_window, min_periods=5).mean()
        # Invert: high PCR is bullish from contrarian perspective
        signal = self._z_score(pcr_smooth)
        self._signals["pcr"] = signal
        return signal

    def vol_skew_signal(
        self,
        otm_put_vol: pd.Series,
        atm_vol: pd.Series,
        otm_call_vol: pd.Series,
    ) -> pd.Series:
        """
        25-delta risk reversal / skew signal.

        Steep put skew => downside fear premium.
        Formula: (OTM Put IV - OTM Call IV) / ATM IV

        Positive = put premium elevated = fear (contrarian bullish)
        """
        skew = (otm_put_vol - otm_call_vol) / atm_vol.replace(0, np.nan)
        skew_smooth = skew.rolling(self.config.skew_window, min_periods=5).mean()
        signal = self._z_score(skew_smooth)
        self._signals["vol_skew"] = signal
        return signal

    def vix_term_structure_signal(
        self,
        front_vix: pd.Series,
        back_vix: pd.Series,
    ) -> pd.Series:
        """
        VIX term structure slope.

        Contango (back > front): normal, risk-on
        Backwardation (front > back): stress, fear

        Returns z-scored slope; positive backwardation => contrarian long signal.
        """
        slope = front_vix - back_vix
        signal = self._z_score(slope)
        self._signals["vix_term_structure"] = signal
        return signal

    # ------------------------------------------------------------------
    # COT / Positioning
    # ------------------------------------------------------------------

    def cot_positioning_signal(
        self,
        net_speculative_positions: pd.Series,
        open_interest: pd.Series,
    ) -> pd.Series:
        """
        CFTC Commitment of Traders (COT) positioning signal.

        Net speculative long as % of open interest, normalized.
        Extreme longs => crowded, potential reversal risk.

        Returns contrarian signal (negative = crowded long = bearish tilt).
        """
        net_pct = net_speculative_positions / open_interest.replace(0, np.nan)
        # Normalize to historical range
        window = self.config.cot_lookback
        rolling_min = net_pct.rolling(window, min_periods=window // 2).min()
        rolling_max = net_pct.rolling(window, min_periods=window // 2).max()
        # Percentile rank [0,1]
        pct_rank = (net_pct - rolling_min) / (rolling_max - rolling_min + 1e-12)
        # Contrarian: extreme longs (rank > 0.8) => negative; extreme shorts => positive
        contrarian = 0.5 - pct_rank  # center and invert
        signal = self._z_score(contrarian)
        self._signals["cot_positioning"] = signal
        return signal

    def short_interest_signal(
        self,
        short_interest: pd.Series,
        shares_outstanding: pd.Series,
    ) -> pd.Series:
        """
        Short interest ratio signal.

        High short interest => potential short squeeze (contrarian bullish)
        or confirmed downtrend (momentum bearish). Here we use contrarian framing.
        """
        si_ratio = short_interest / shares_outstanding.replace(0, np.nan)
        signal = self._z_score(si_ratio)
        self._signals["short_interest"] = signal
        return signal

    # ------------------------------------------------------------------
    # Flow Signals
    # ------------------------------------------------------------------

    def fund_flow_signal(
        self,
        retail_flows: pd.Series,
        institutional_flows: pd.Series,
    ) -> pd.Series:
        """
        Retail vs institutional flow divergence.

        When retail aggressively buys while institutions sell => bearish (crowding).
        Signal = institutional flow - retail flow (positive = institutions leading)
        """
        divergence = institutional_flows - retail_flows
        divergence_smooth = divergence.ewm(span=self.config.smoothing_span, adjust=False).mean()
        signal = self._z_score(divergence_smooth)
        self._signals["fund_flow"] = signal
        return signal

    def news_sentiment_proxy(
        self,
        positive_news_count: pd.Series,
        negative_news_count: pd.Series,
    ) -> pd.Series:
        """
        Simple news sentiment proxy.

        Ratio-based score: (pos - neg) / (pos + neg + 1)
        Smoothed and z-scored as a trend/sentiment indicator.
        """
        total = positive_news_count + negative_news_count + 1
        raw_score = (positive_news_count - negative_news_count) / total
        smooth = raw_score.ewm(span=self.config.smoothing_span, adjust=False).mean()
        signal = self._z_score(smooth)
        self._signals["news_sentiment"] = signal
        return signal

    # ------------------------------------------------------------------
    # Composite Fear/Greed Score
    # ------------------------------------------------------------------

    def fear_greed_index(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Composite Fear & Greed Index.

        Aggregates all computed sub-signals into a single score:
        - Positive score => Greed / Euphoria (fade / risk-reduce)
        - Negative score => Fear / Panic (accumulate / contrarian long)

        Parameters
        ----------
        weights : dict, optional
            Weight per sub-signal. Defaults to equal weights.

        Returns
        -------
        pd.Series  Fear & Greed score, clipped to [-3, +3]
        """
        if not self._signals:
            raise ValueError("No sub-signals computed. Run signal methods first.")

        keys = list(self._signals.keys())
        if weights is None:
            w = {k: 1.0 / len(keys) for k in keys}
        else:
            total = sum(weights.values())
            w = {k: v / total for k, v in weights.items()}

        df = pd.concat(
            [self._signals[k].rename(k) for k in keys if k in self._signals],
            axis=1,
        ).dropna(how="all")

        fg = sum(df[k] * w.get(k, 0.0) for k in df.columns)
        return fg.clip(-self.config.signal_clip, self.config.signal_clip)

    def sentiment_regime(
        self,
        fg_index: pd.Series,
        fear_threshold: float = -1.0,
        greed_threshold: float = 1.0,
    ) -> pd.Series:
        """
        Classify sentiment into discrete regimes.

        Returns
        -------
        pd.Series  String labels: 'FEAR', 'NEUTRAL', 'GREED'
        """
        regime = pd.Series("NEUTRAL", index=fg_index.index)
        regime[fg_index <= fear_threshold] = "FEAR"
        regime[fg_index >= greed_threshold] = "GREED"
        return regime

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _z_score(self, series: pd.Series) -> pd.Series:
        """Rolling z-score normalization."""
        window = self.config.z_score_window
        mu = series.rolling(window, min_periods=window // 2).mean()
        sigma = series.rolling(window, min_periods=window // 2).std()
        z = (series - mu) / sigma.replace(0, np.nan)
        return z.clip(-self.config.signal_clip, self.config.signal_clip)

    def get_all_signals(self) -> pd.DataFrame:
        """Return DataFrame of all computed signals."""
        if not self._signals:
            return pd.DataFrame()
        return pd.concat(
            [v.rename(k) for k, v in self._signals.items()], axis=1
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    idx = pd.date_range("2018-01-01", periods=1500, freq="B")
    n = len(idx)

    sent = SentimentSignal()

    put_vol  = pd.Series(np.random.randint(5000, 20000, n), index=idx, dtype=float)
    call_vol = pd.Series(np.random.randint(8000, 25000, n), index=idx, dtype=float)
    otm_put  = pd.Series(np.random.normal(0.22, 0.04, n), index=idx)
    atm      = pd.Series(np.random.normal(0.18, 0.03, n), index=idx)
    otm_call = pd.Series(np.random.normal(0.17, 0.03, n), index=idx)
    vix_front = pd.Series(np.random.normal(18, 5, n), index=idx)
    vix_back  = pd.Series(np.random.normal(20, 4, n), index=idx)
    net_spec  = pd.Series(np.random.normal(5000, 2000, n), index=idx)
    oi        = pd.Series(np.full(n, 50000.0), index=idx)
    pos_news  = pd.Series(np.random.randint(10, 100, n), index=idx, dtype=float)
    neg_news  = pd.Series(np.random.randint(5, 80, n), index=idx, dtype=float)

    sent.put_call_ratio_signal(put_vol, call_vol)
    sent.vol_skew_signal(otm_put, atm, otm_call)
    sent.vix_term_structure_signal(vix_front, vix_back)
    sent.cot_positioning_signal(net_spec, oi)
    sent.news_sentiment_proxy(pos_news, neg_news)

    fg = sent.fear_greed_index()
    regime = sent.sentiment_regime(fg)

    print("Fear & Greed Index (last 5):\n", fg.tail())
    print("\nSentiment Regime (last 5):\n", regime.tail())
    print("\nAll signals:\n", sent.get_all_signals().tail(3))
