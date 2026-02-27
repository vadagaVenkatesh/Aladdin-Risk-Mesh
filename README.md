# Aladdin Risk Mesh

> **Global Macro Long/Short Hedge Fund — Full Quantitative Stack for 2026**

**Author:** VDG Venkatesh ([@vadagaVenkatesh](https://github.com/vadagaVenkatesh))

A production-grade research implementation of a **single-fund global macro / long–short hedge fund** quant stack, purpose-built for the 2026 inflationary growth regime. This project combines robust classical techniques (trend, carry, cross-sectional factors, scenario risk) with carefully-scoped machine learning for macro signal pooling, regime detection, and an advanced automated layer for research intelligence and governance.

---

## Architecture Overview

```text
Aladdin-Risk-Mesh/
├── data/               # Data layer & loaders
├── signals/            # Alpha signal generation
├── ml/                 # Machine learning signal enhancement
├── regime/             # Regime & volatility modeling
├── risk/               # Risk engine (FRTB-aligned)
├── portfolio/          # Portfolio construction & optimization
├── execution/          # Risk-aware execution simulation
├── agents/             # Automated research & governance layer
├── backtest/           # Backtesting engine & metrics
├── config/             # Basel IV-aligned configurations
├── main.py             # Orchestrator entry point
└── requirements.txt
```

---

## Module Descriptions

### 1. Data Layer (`data/`)
Multi-asset daily data ingestion for equity indices, bond futures, FX pairs, commodity futures, and macro series (CPI, PMI, rates). Feature engineering pipelines compute returns, z-scores, rolling statistics, and cross-sectional ranks.

### 2. Signal Generation (`signals/`)
- **Momentum:** Time-series momentum (TSMOM) across multiple asset classes.
- **Carry:** Interest rate carry, FX carry, and commodity roll yield.
- **Value:** Real yield differentials, FX PPP deviations, and equity valuation spreads.

### 3. ML Module (`ml/`)
- **Panel Regression:** Elastic Net with cross-country pooling.
- **Gradient Boosting:** Non-linear macro factor models.
- **Learn-to-Rank:** LambdaMART for cross-sectional relative value ranking.

### 4. Regime & Volatility (`regime/`)
- **HMM Classifier:** Gaussian HMM for macro regime detection (Growth, Inflation, Stagflation, Recession).
- **Volatility:** GARCH(1,1) and HAR-RV forecasting.

### 5. Risk Engine (`risk/`)
- **Factor Model:** PCA-based risk decomposition.
- **VaR Engine:** Historical and Parametric VaR/ES with multiple lookback windows.
- **Scenario Analytics:** Stress testing library including GFC, COVID, and 2026 inflationary shock analogs.

### 6. Portfolio Construction (`portfolio/`)
- **Optimizer:** Constrained Mean-Variance Optimization and Black-Litterman framework.
- **Kelly Sizing:** Fractional Kelly calibration based on strategy performance.
- **Risk Parity:** Equal risk contribution across macro sleeves.

### 7. Execution Simulator (`execution/`)
- **Optimal Execution:** Intraday impact modeling and slippage estimation.
- **VWAP/TWAP:** Participation-based order scheduling.

### 8. Automated Intelligence (`agents/`)
- **Research Engine:** Intelligence layer for backtest analysis and signal performance monitoring.
- **Scenario Generator:** Translates narratives into structured factor shocks.
- **Governance Monitor:** Model risk monitoring, anomaly detection, and automated reporting.

---

## Proprietary and Confidential

**Intellectual Property Notice**

Copyright (c) 2026 VDG Venkatesh. All rights reserved.

This software and its documentation are the exclusive property of VDG Venkatesh. No part of this project may be used, reproduced, modified, or distributed in any other projects, systems, or platforms without explicit written permission from the owner. 

---

## Authors & Attribution

| Role | Contributor |
| :--- | :--- |
| Lead Quant Developer & Architect | **VDG Venkatesh** ([@vadagaVenkatesh](https://github.com/vadagaVenkatesh)) |

This repository represents a proprietary research effort in quantitative finance architecture and automated model governance.
