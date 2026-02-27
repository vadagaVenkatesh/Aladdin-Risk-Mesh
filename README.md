# Aladdin Risk Mesh

> **Global Macro Long/Short Hedge Fund — Full Quantitative Stack for 2026**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Research%20MVP-orange.svg)]()
[![Regime](https://img.shields.io/badge/Regime-Inflationary%20Growth%202026-red.svg)]()

**Authors:** VDG Venkatesh ([@vadagaVenkatesh](https://github.com/vadagaVenkatesh)) & Agentic AI

A production-grade research implementation of a **single-fund global macro / long–short hedge fund** quant stack, purpose-built for the 2026 inflationary growth regime. This project combines robust classical techniques (trend, carry, cross-sectional factors, scenario risk) with carefully-scoped machine learning for macro signal pooling, regime detection, and a thin LLM agentic layer for research intelligence and governance.

---

## Architecture Overview

```
Aladdin-Risk-Mesh/
├── data/                        # Data layer & loaders
│   ├── __init__.py
│   ├── loaders.py               # Multi-asset data ingestion (equity, bonds, FX, commodities, macro)
│   ├── features.py              # Feature engineering & signal preprocessing
│   └── universe.py              # Instrument universe definition
├── signals/                     # Alpha signal generation
│   ├── __init__.py
│   ├── momentum.py              # Time-series momentum (TSMOM)
│   ├── carry.py                 # Carry & roll-down signals
│   ├── value.py                 # Value & macro valuation signals
│   └── cross_asset.py           # Cross-asset relative value spreads
├── ml/                          # Machine learning signal enhancement
│   ├── __init__.py
│   ├── panel_regression.py      # Elastic Net / Ridge panel regression
│   ├── gradient_boosting.py     # XGBoost / LightGBM macro factor models
│   └── learn_to_rank.py         # LambdaMART learn-to-rank for cross-sectional
├── regime/                      # Regime & volatility modeling
│   ├── __init__.py
│   ├── hmm_classifier.py        # Hidden Markov Model regime detection
│   ├── clustering.py            # K-Means / GMM macro regime clustering
│   └── volatility.py            # GARCH / HAR-RV volatility forecasting
├── risk/                        # Risk engine
│   ├── __init__.py
│   ├── factor_model.py          # PCA-based factor risk decomposition
│   ├── var_engine.py            # Historical / Parametric VaR & ES
│   └── scenario_analytics.py    # Historical & hypothetical scenario library
├── portfolio/                   # Portfolio construction
│   ├── __init__.py
│   ├── optimizer.py             # MVO / Black-Litterman constrained optimizer
│   ├── kelly_sizing.py          # Fractional Kelly position sizing
│   └── risk_parity.py           # Risk parity & budgeting across sleeves
├── execution/                   # Execution simulation
│   ├── __init__.py
│   ├── almgren_chriss.py        # Almgren-Chriss optimal execution
│   ├── vwap_twap.py             # VWAP / TWAP schedulers
│   └── impact_model.py          # Market impact & transaction cost model
├── agents/                      # LLM agentic layer
│   ├── __init__.py
│   ├── research_agent.py        # Research Intelligence Agent (RAG over logs)
│   ├── scenario_agent.py        # Scenario & Narrative design agent
│   └── governance_agent.py      # Governance, ops & report drafting agent
├── backtest/                    # Backtesting engine
│   ├── __init__.py
│   ├── engine.py                # Event-driven backtest loop
│   └── performance.py           # Sharpe, drawdown, regime analytics
├── config/
│   ├── universe.yaml            # Instrument universe config
│   └── strategy.yaml            # Strategy parameters
├── notebooks/
│   └── 01_full_pipeline_demo.ipynb
├── main.py                      # Orchestrator entry point
├── requirements.txt
└── README.md
```

---

## Module Descriptions

### 1. Data Layer (`data/`)
Multi-asset daily data ingestion using `yfinance` and `pandas-datareader` for equity indices, bond futures, FX pairs, commodity futures, and macro series (CPI, PMI, rates). Feature engineering pipelines compute returns, z-scores, rolling statistics, and cross-sectional ranks.

### 2. Signal Generation (`signals/`)
- **Momentum (`momentum.py`):** Time-series momentum (TSMOM) with lookback periods of 1, 3, 6, 12 months across equity indices, bond futures, FX pairs, and commodities.
- **Carry (`carry.py`):** Interest rate carry (short vs long maturities), FX carry via interest rate differentials, commodity roll yield (contango/backwardation).
- **Value (`value.py`):** Real yield differentials, FX PPP/BEER deviations, equity CAPE spreads, breakeven inflation.
- **Cross-Asset Spreads (`cross_asset.py`):** Growth/inflation triangles, equity vs commodity proxies, rate-FX-commodity relative value spreads.

### 3. ML Module (`ml/`)
- **Panel Regression (`panel_regression.py`):** Elastic Net with cross-country pooling. Shares information across geographies and instruments with L1/L2 regularization to prevent overfitting on sparse macro data.
- **Gradient Boosting (`gradient_boosting.py`):** LightGBM / XGBoost models capturing non-linear macro-return relationships; regime-conditioned predictions.
- **Learn-to-Rank (`learn_to_rank.py`):** LambdaMART ranking model for cross-sectional macro relative value. Ranks instruments by expected excess return rather than predicting point estimates.

### 4. Regime & Volatility (`regime/`)
- **HMM Classifier (`hmm_classifier.py`):** Gaussian HMM on macro + market features (growth, inflation, VIX, credit spreads, yield curve slope) to detect regimes: Goldilocks, Inflationary Growth, Stagflation, Recession.
- **Clustering (`clustering.py`):** K-Means and GMM clustering for historical analog identification and regime-aware backtesting.
- **Volatility (`volatility.py`):** GARCH(1,1) and HAR-RV models for realized variance forecasting; outputs feed into position sizing and risk targets.

### 5. Risk Engine (`risk/`)
- **Factor Model (`factor_model.py`):** PCA-based factor decomposition across equity, rates, FX, commodity, and style factors (value, momentum, carry). Computes factor exposures and residual idiosyncratic risk.
- **VaR Engine (`var_engine.py`):** Historical simulation VaR/ES for 1-10 day horizons; parametric approximations for intraday monitoring; multiple lookback windows.
- **Scenario Analytics (`scenario_analytics.py`):** Library of historical shocks (2008 GFC, 2013 Taper Tantrum, 2020 COVID, 2022 energy spike) and forward-looking hypothetical scenarios (rapid reflation, stagflation, policy mistake, China shock).

### 6. Portfolio Construction (`portfolio/`)
- **Optimizer (`optimizer.py`):** Constrained mean-variance optimizer and Black-Litterman framework blending model views with market-implied returns. Supports net/gross exposure limits, sector/country caps, liquidity constraints.
- **Kelly Sizing (`kelly_sizing.py`):** Fractional Kelly (0.25-0.5x) at signal/strategy level, calibrated from backtest Sharpe and max drawdown.
- **Risk Parity (`risk_parity.py`):** Equal risk contribution across macro, equity long/short, and commodity-hedge sleeves.

### 7. Execution Simulator (`execution/`)
- **Almgren-Chriss (`almgren_chriss.py`):** Optimal liquidation trajectory minimizing market impact + timing risk, solving the trade-off given order size, volatility, and liquidity.
- **VWAP/TWAP (`vwap_twap.py`):** VWAP and TWAP participation schedulers for large orders.
- **Impact Model (`impact_model.py`):** Square-root market impact model with transaction cost estimation.

### 8. Agents (`agents/`)
- **Research Agent (`research_agent.py`):** RAG-based LLM agent over backtest logs, signal performance history, and markdown research notes. Answers questions like "which signal is driving Sharpe this quarter?"
- **Scenario Agent (`scenario_agent.py`):** Translates macro narratives into structured scenario definitions (factor shocks, curve shifts, commodity moves) and calls the deterministic scenario engine.
- **Governance Agent (`governance_agent.py`):** Monitors data freshness, flags anomalies, and auto-drafts risk reports and model documentation grounded in logs and risk outputs.

---

## Quick Start

```bash
git clone https://github.com/vadagaVenkatesh/Aladdin-Risk-Mesh.git
cd Aladdin-Risk-Mesh
pip install -r requirements.txt
python main.py --mode backtest --start 2015-01-01 --end 2025-12-31
```

---

## Related Repositories

| Repo | Description | Relevance |
|------|-------------|----------|
| [consumer-risk-model-mesh](https://github.com/vadagaVenkatesh/consumer-risk-model-mesh) | Agentic PD/LGD/EAD stress testing | Stress testing patterns, agent architecture |
| [Algorithmic-Trading-Strategy-Backtester](https://github.com/vadagaVenkatesh/Algorithmic-Trading-Strategy-Backtester) | Momentum & stat-arb backtesting | Backtest engine, performance analytics |
| [stochastic-interest-rate-simulator](https://github.com/vadagaVenkatesh/stochastic-interest-rate-simulator) | Short-rate models (EM, Milstein) | Rate factor modeling, bond carry |
| [reg-capital-fairness-agentic-rag](https://github.com/vadagaVenkatesh/reg-capital-fairness-agentic-rag) | Agentic RAG for regulatory compliance | RAG architecture for research agent |

---

## 2026 Macro Regime Context

This stack is designed for the **inflationary growth regime** characterized by:
- Sticky inflation above 2% targets with higher-for-longer nominal rates
- Positive stock-bond correlation reducing traditional 60/40 diversification
- Asynchronous global cycle: US resilient, Europe stabilizing, EM divergent
- Persistent macro and geopolitical shocks driving cross-asset dispersion

**Why this algorithm combination:**
- Trend/carry remain core: inflationary periods with policy volatility favor persistent price moves
- Pooled ML outperforms single-asset models in low-breadth macro environments
- Regime detection provides guardrails for leverage and scenario focus
- Scenario analytics aligned with institutional macro outlooks

---

## Authors & Attribution

| Role | Contributor |
|------|-------------|
| Lead Quant Developer & Architect | **VDG Venkatesh** ([@vadagaVenkatesh](https://github.com/vadagaVenkatesh)) |
| Research Design & Agentic Co-Author | **Agentic AI** (Comet, Perplexity) |

*This repository represents a collaborative research effort combining deep quant finance domain expertise with agentic AI-assisted architecture design, code scaffolding, and documentation.*
