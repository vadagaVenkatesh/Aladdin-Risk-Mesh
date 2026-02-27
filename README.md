# Aladdin Risk Mesh

> **Global Macro Long/Short Hedge Fund — Full Quantitative Stack for 2026**

**Author:** VDG Venkatesh ([@vadagaVenkatesh](https://github.com/vadagaVenkatesh))

A production-grade research platform for a single-fund global macro / long-short hedge fund. Built for the 2026 inflationary growth regime — combining robust classical quant techniques (trend, carry, cross-sectional factors, scenario risk) with carefully-scoped machine learning for macro signal pooling, regime detection, and an automated research & governance layer.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Module Descriptions](#3-module-descriptions)
4. [Data Flow](#4-data-flow)
5. [Quick Start](#5-quick-start)
6. [Configuration](#6-configuration)
7. [Running Tests](#7-running-tests)
8. [Deployment (Docker / Azure)](#8-deployment)
9. [Airflow Orchestration](#9-airflow-orchestration)
10. [Legal & Proprietary Notice](#10-legal)

---

## 1. Project Overview

Aladdin Risk Mesh implements a complete institutional-grade quant stack:

| Layer | Purpose |
|---|---|
| **Data** | Bloomberg primary + Yahoo fallback, Parquet caching |
| **Signals** | Trend, carry, cross-sectional, sentiment, VIX term structure |
| **ML** | LSTM regime classifier, conformal prediction, Numba-accelerated |
| **Regime** | Volatility surface modeling, HMM macro state detection |
| **Risk** | FRTB SA-CCR, CVaR, Greeks, stress scenarios (NJIT-optimized) |
| **Portfolio** | Black-Litterman + MVO + Risk Parity (Numba-accelerated) |
| **Execution** | Intraday VaR-aware execution, slippage modeling |
| **Agents** | LLM research agent, governance agent, scenario agent |
| **Backtest** | Vectorized backtesting engine with walk-forward analysis |
| **Spark** | Distributed Monte Carlo, 10k simulation paths on Azure |
| **API** | FastAPI REST service for real-time risk & portfolio queries |
| **Airflow** | Daily 6AM EST pipeline DAG orchestration |

---

## 2. Repository Structure

```
Aladdin-Risk-Mesh/
|
|-- data/                          # Data layer & market loaders
|   |-- loader.py                  # Bloomberg primary + Yahoo fallback
|   |-- parquet_cache.py           # Parquet-based caching layer
|   `-- __init__.py
|
|-- signals/                       # Alpha signal generation
|   |-- alpha.py                   # Trend, carry, cross-sectional signals
|   |-- sentiment.py               # PCR, vol skew, VIX term structure
|   `-- __init__.py
|
|-- ml/                            # Machine learning layer
|   |-- lstm_classifier.py         # LSTM regime classifier (PyTorch)
|   |-- conformal.py               # Conformal prediction wrappers
|   `-- __init__.py
|
|-- regime/                        # Regime & volatility modeling
|   |-- detector.py                # HMM-based macro regime detection
|   |-- volatility.py              # Volatility surface modeling
|   `-- __init__.py
|
|-- risk/                          # Risk engine (FRTB-aligned)
|   |-- engine.py                  # VaR, CVaR, Greeks (NJIT-optimized)
|   |-- greeks.py                  # Options Greeks calculator
|   `-- __init__.py
|
|-- portfolio/                     # Portfolio construction & optimization
|   |-- optimizer.py               # Black-Litterman + MVO + Risk Parity
|   |-- rebalancer.py              # Rebalancing & turnover control
|   `-- __init__.py
|
|-- execution/                     # Trade execution simulation
|   |-- executor.py                # Intraday VaR-aware execution
|   |-- slippage.py                # Slippage & market impact modeling
|   `-- __init__.py
|
|-- agents/                        # AI/LLM research & governance layer
|   |-- llm_agent.py               # Natural language portfolio queries
|   |-- research_agent.py          # Automated research intelligence
|   |-- governance_agent.py        # AI/ML model risk monitoring
|   |-- scenario_agent.py          # Scenario generation & stress testing
|   `-- __init__.py
|
|-- backtest/                      # Backtesting engine
|   |-- engine.py                  # Vectorized walk-forward backtester
|   |-- metrics.py                 # Sharpe, Sortino, max drawdown, etc.
|   `-- __init__.py
|
|-- spark/                         # Distributed computing (Azure Spark)
|   |-- monte_carlo.py             # 10k-path Monte Carlo on Spark
|   `-- __init__.py
|
|-- core/                          # Shared utilities & data bus
|   |-- bus.py                     # Parquet data bus with channel registry
|   |-- utils.py                   # Common helpers & type definitions
|   `-- __init__.py
|
|-- api/                           # FastAPI REST service
|   |-- main.py                    # API entrypoint & route definitions
|   `-- __init__.py
|
|-- config/                        # Configuration files
|   |-- settings.yaml              # Basel IV / SA-CCR aligned config
|   `-- azure_config.yaml          # Azure D16s_v5 deployment config
|
|-- ops/                           # Operational monitoring
|   |-- monitor.py                 # VaR/drawdown/exposure monitoring
|   `-- __init__.py
|
|-- airflow/                       # Airflow DAG orchestration
|   `-- dags/
|       `-- daily_pipeline_dag.py  # Daily 6AM EST pipeline
|
|-- environment/                   # Environment setup
|   |-- azure_env.yaml             # Azure conda environment
|   `-- requirements_spark.txt     # Spark-specific dependencies
|
|-- examples/                      # Demo & integration examples
|   `-- demo_pipeline.py           # End-to-end demo pipeline
|
|-- tests/                         # Test suite
|   |-- test_risk.py               # Risk engine tests
|   |-- test_portfolio.py          # Portfolio optimizer tests
|   |-- test_execution.py          # Execution engine tests
|   |-- test_data.py               # Data loader tests
|   `-- test_signals.py            # Signal generation tests
|
|-- main.py                        # Main orchestrator entry point
|-- requirements.txt               # Python dependencies
|-- Dockerfile                     # Container build spec
|-- docker-compose.yml             # Multi-service orchestration
|-- .gitignore                     # Git ignore rules
|-- LICENSE                        # Proprietary license
`-- README.md                      # This file
```

> **Note:** The `src/` directory was a legacy scaffold folder and has been fully consolidated into the above module structure. All canonical code lives in the named module folders.

---

## 3. Module Descriptions

### `data/` — Market Data Layer
Handles Bloomberg (primary) and Yahoo Finance (fallback) ingestion. All data is cached as Parquet files for fast re-reads. Supports equities, FX, rates, commodities, and crypto.

### `signals/` — Alpha Signal Generation
Implements trend-following (momentum), carry, cross-sectional relative value, sentiment analysis (PCR, vol skew), and VIX term structure signals. All signals are vectorized with NumPy.

### `ml/` — Machine Learning Layer
LSTM-based macro regime classifier trained on rolling windows. Uses conformal prediction for calibrated uncertainty. All compute-intensive paths use Numba `@njit` JIT compilation.

### `regime/` — Regime & Volatility Modeling
Hidden Markov Model (HMM) for macro state detection (risk-on / risk-off / crisis). Volatility surface modeling with SABR and SVI parameterizations.

### `risk/` — Risk Engine (FRTB-Aligned)
Computes VaR, CVaR, Expected Shortfall, Greeks, and stress-test scenarios. Fully aligned with FRTB SA-CCR Basel IV standards. Core loops accelerated with Numba `@njit` and parallel execution.

### `portfolio/` — Portfolio Optimization
Implements Black-Litterman with investor views, Mean-Variance Optimization with shrinkage, and Risk Parity. All matrix operations vectorized. Numba-accelerated for large universes.

### `execution/` — Execution Simulation
Intraday VaR-aware execution with slippage, market impact, and fill modeling. Supports limit orders, TWAP, and VWAP execution strategies.

### `agents/` — AI Research & Governance Layer
LLM-powered natural language interface for portfolio queries (GPT-4 / Llama). Governance agent monitors model drift and risk limit breaches. Scenario agent generates macro stress scenarios.

### `backtest/` — Backtesting Engine
Vectorized walk-forward backtester. Computes Sharpe, Sortino, max drawdown, Calmar, information ratio, and transaction cost-adjusted returns.

### `spark/` — Distributed Monte Carlo (Azure)
Apache Spark-based distributed Monte Carlo with 10,000 simulation paths. Runs on Azure D16s_v5 clusters. Outputs VaR, CVaR, and tail risk distributions.

### `api/` — FastAPI REST Service
REST endpoints for real-time risk queries, portfolio rebalancing triggers, and signal reads. Supports async I/O with full Pydantic validation.

### `airflow/dags/` — Orchestration
Apache Airflow DAG runs daily at 6AM EST: load data → generate signals → compute risk → optimize portfolio → run Monte Carlo → persist state.

---

## 4. Data Flow

```
[Market Data] --> data/loader.py
      |
      v
[Signals] --> signals/alpha.py + signals/sentiment.py
      |
      v
[ML Regime] --> ml/lstm_classifier.py + regime/detector.py
      |
      v
[Risk Engine] --> risk/engine.py (FRTB VaR / CVaR / Greeks)
      |
      v
[Portfolio Optimizer] --> portfolio/optimizer.py (BL + MVO)
      |
      v
[Execution] --> execution/executor.py
      |
      v
[Backtest] --> backtest/engine.py
      |
      v
[Spark MC] --> spark/monte_carlo.py
      |
      v
[Persist] --> core/bus.py (Parquet)
```

---

## 5. Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vadagaVenkatesh/Aladdin-Risk-Mesh.git
cd Aladdin-Risk-Mesh

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Run the API server
uvicorn api.main:app --reload --port 8000

# 5. Run tests
pytest tests/ -v
```

### Docker
```bash
docker-compose up --build
```

---

## 6. Configuration

All configuration lives in `config/`:

- `settings.yaml` — Basel IV SA-CCR aligned parameters, VaR confidence levels, position limits, signal weights
- `azure_config.yaml` — Azure D16s_v5 cluster settings, Spark executor config, storage accounts

Key environment variables:
```bash
BLOOMBERG_API_KEY=...        # Bloomberg API key
OPENAI_API_KEY=...           # LLM agent API key
AZURE_STORAGE_ACCOUNT=...    # Azure Blob storage
AIRFLOW_DAILY_RUN_HOUR_EST=6  # Pipeline schedule
```

---

## 7. Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_risk.py -v
pytest tests/test_portfolio.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 8. Deployment

### Docker
```bash
docker build -t aladdin-risk-mesh .
docker run -p 8000:8000 aladdin-risk-mesh
```

### Azure
See `environment/azure_env.yaml` for full Azure D16s_v5 conda environment spec.

Spark jobs: `spark/monte_carlo.py` submits to Azure HDInsight or Databricks.

---

## 9. Airflow Orchestration

The DAG `airflow/dags/daily_pipeline_dag.py` schedules the full pipeline daily at **6:00 AM EST**:

```
load_market_data
    --> generate_signals
        --> compute_risk
            --> optimize_portfolio
                --> run_monte_carlo
                    --> persist_state
```

Deploy to a managed Airflow environment (MWAA on AWS or Azure Managed Airflow).

---

## 10. Legal

Copyright (c) 2024–2026 VDG Venkatesh. All Rights Reserved.

This software and all associated source code, algorithms, models, configurations, and documentation are the exclusive proprietary intellectual property of VDG Venkatesh.

Unauthorized copying, reproduction, distribution, modification, or use of this software, in whole or in part, in any form or by any means, is strictly prohibited without the express prior written permission of the copyright holder.

This software may not be used in any commercial or non-commercial project, product, or service without explicit written authorization.

---

*Built for institutional-grade macro investing — Aladdin Risk Mesh 2026*
