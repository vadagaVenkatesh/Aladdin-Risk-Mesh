"""
demo_full_pipeline.py

PROPRIETARY AND CONFIDENTIAL
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

This software and associated documentation files are the proprietary 
and confidential information of VDG Venkatesh. Unauthorized copying,
modification, distribution, or use of this software, via any medium,
is strictly prohibited without express written permission.

NO LICENSE IS GRANTED. This code may not be used, reproduced, or 
incorporated into any other projects without explicit authorization.
For licensing inquiries, contact the copyright holder.

End-to-End Demo: Macro Hedge Fund Risk Mesh
- Load macro data
- Generate macro signals (vol skew, PCR, VIX term structure)
- Detect regime with HMM
- Build portfolio with Black-Litterman
- Run execution simulator
- Backtest strategy
- Monitor operational limits
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

# ========================================
# Import all modules
# ========================================
from portfolio.optimizer import BlackLittermanOptimizer
from risk.risk_engine import RiskEngine
from risk.kelly_sizing import KellySizer
from regime.hmm_classifier import HMMRegimeClassifier
from regime.volatility import estimate_realized_vol, estimate_garch_vol
from execution.simulator import ExecutionSimulator
from backtest.engine import BacktestEngine
from ops.monitor import OperationalMonitor


def load_mock_market_data(start_date: str, end_date: str, tickers: list) -> pd.DataFrame:
    """
    Generate mock daily price data for demo.
    Real implementation would load from data/macro_loader.py
    """
    dates = pd.date_range(start_date, end_date, freq='B')
    n = len(dates)
    data = {'date': dates}
    
    np.random.seed(42)
    for ticker in tickers:
        # Simulate GBM price paths
        mu = 0.08 / 252
        sigma = 0.2 / np.sqrt(252)
        returns = np.random.normal(mu, sigma, n)
        prices = 100 * np.exp(np.cumsum(returns))
        data[ticker] = prices
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


def generate_macro_signals(prices: pd.DataFrame) -> dict:
    """
    Generate mock macro signals.
    Real implementation: signals/macro_signals.py
    """
    returns = prices.pct_change().dropna()
    
    # Simple momentum + mean reversion blend
    mom_1m = returns.rolling(21).mean().iloc[-1]
    mom_3m = returns.rolling(63).mean().iloc[-1]
    vol = returns.rolling(21).std().iloc[-1]
    
    signals = {
        'momentum_1m': mom_1m.to_dict(),
        'momentum_3m': mom_3m.to_dict(),
        'volatility': vol.to_dict()
    }
    return signals


def run_full_pipeline():
    """
    Execute end-to-end hedge fund pipeline.
    """
    print("=" * 80)
    print("Aladdin Risk Mesh - Full Pipeline Demo")
    print("=" * 80)
    print()
    
    # ========================================
    # 1. Load Market Data
    # ========================================
    print("[1/9] Loading market data...")
    tickers = ['SPY', 'TLT', 'GLD', 'DXY', 'VIX']  # Equities, Bonds, Gold, Dollar, Vol
    start_date = '2023-01-01'
    end_date = '2025-01-01'
    prices = load_mock_market_data(start_date, end_date, tickers)
    returns = prices.pct_change().dropna()
    print(f"   Loaded {len(prices)} days of data for {len(tickers)} assets")
    print()
    
    # ========================================
    # 2. Generate Macro Signals
    # ========================================
    print("[2/9] Generating macro signals...")
    signals = generate_macro_signals(prices)
    print(f"   Generated signals: {list(signals.keys())}")
    print()
    
    # ========================================
    # 3. Regime Detection
    # ========================================
    print("[3/9] Detecting market regime with HMM...")
    regime_model = HMMRegimeClassifier(n_regimes=3, random_state=42)
    
    # Use VIX as regime indicator (mock)
    vix_series = prices['VIX'].values.reshape(-1, 1)
    regime_model.fit(vix_series)
    current_regime = regime_model.predict(vix_series[-30:])[-1]
    regime_probs = regime_model.predict_proba(vix_series[-30:])
    
    regime_names = {0: 'Low Vol', 1: 'Medium Vol', 2: 'High Vol'}
    print(f"   Current regime: {regime_names.get(current_regime, current_regime)}")
    print(f"   Regime probabilities: {regime_probs[-1]}")
    print()
    
    # ========================================
    # 4. Build Black-Litterman Portfolio
    # ========================================
    print("[4/9] Building Black-Litterman portfolio...")
    optimizer = BlackLittermanOptimizer(
        market_caps={'SPY': 40e12, 'TLT': 30e12, 'GLD': 10e12, 'DXY': 5e12, 'VIX': 1e12},
        risk_free_rate=0.04,
        delta=2.5
    )
    
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Define views: bullish SPY, bearish DXY
    views = pd.DataFrame({
        'SPY': [1.0, 0.0],
        'TLT': [0.0, 0.0],
        'GLD': [0.0, 0.0],
        'DXY': [0.0, -1.0],
        'VIX': [0.0, 0.0]
    })
    expected_returns_views = np.array([0.12, -0.08])  # 12% SPY, -8% DXY
    view_confidences = np.array([0.6, 0.5])
    
    weights = optimizer.optimize(
        cov_matrix=cov_matrix,
        views=views,
        expected_returns_views=expected_returns_views,
        view_confidences=view_confidences
    )
    
    print("   Optimal weights:")
    for ticker, w in weights.items():
        print(f"      {ticker}: {w*100:.2f}%")
    print()
    
    # ========================================
    # 5. Risk Sizing with Kelly Criterion
    # ========================================
    print("[5/9] Applying Kelly criterion for position sizing...")
    sizer = KellySizer(max_leverage=2.0, half_kelly=True)
    
    # Mock expected returns and covariance
    exp_returns = returns.mean() * 252
    kelly_weights = sizer.calculate_kelly_weights(
        expected_returns=exp_returns,
        cov_matrix=cov_matrix
    )
    
    print("   Kelly weights:")
    for ticker, w in kelly_weights.items():
        print(f"      {ticker}: {w*100:.2f}%")
    print()
    
    # ========================================
    # 6. Risk Engine Analysis
    # ========================================
    print("[6/9] Running risk engine...")
    risk_engine = RiskEngine()
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    
    var_95 = risk_engine.calculate_var(portfolio_returns, confidence_level=0.95)
    cvar_95 = risk_engine.calculate_cvar(portfolio_returns, confidence_level=0.95)
    
    print(f"   Portfolio VaR (95%): {var_95*100:.2f}%")
    print(f"   Portfolio CVaR (95%): {cvar_95*100:.2f}%")
    print()
    
    # ========================================
    # 7. Execution Simulation
    # ========================================
    print("[7/9] Simulating execution with market impact...")
    simulator = ExecutionSimulator(
        initial_capital=10_000_000,
        commission_rate=0.001,
        slippage_bps=5
    )
    
    trades = [
        {'ticker': ticker, 'quantity': int(weights[ticker] * 10_000_000 / prices[ticker].iloc[-1])}
        for ticker in tickers if abs(weights.get(ticker, 0)) > 0.01
    ]
    
    executed_trades = []
    for trade in trades:
        result = simulator.execute_trade(
            ticker=trade['ticker'],
            quantity=trade['quantity'],
            price=prices[trade['ticker']].iloc[-1]
        )
        executed_trades.append(result)
    
    print(f"   Executed {len(executed_trades)} trades")
    print(f"   Final capital: ${simulator.get_portfolio_value():,.2f}")
    print()
    
    # ========================================
    # 8. Backtest Engine
    # ========================================
    print("[8/9] Running backtest...")
    backtest = BacktestEngine(
        initial_capital=10_000_000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Simple static weight backtest
    for idx in range(100, len(returns)):
        daily_ret = (returns.iloc[idx] * pd.Series(weights)).sum()
        backtest.update(daily_ret)
    
    performance = backtest.get_performance_metrics()
    print("   Backtest results:")
    print(f"      Total Return: {performance['total_return']*100:.2f}%")
    print(f"      Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"      Max Drawdown: {performance['max_drawdown']*100:.2f}%")
    print()
    
    # ========================================
    # 9. Operational Monitoring
    # ========================================
    print("[9/9] Checking operational limits...")
    monitor = OperationalMonitor(log_dir="/tmp/aladdin_logs")
    
    # Simulate VaR breach
    monitor.check_var_limit(var_value=0.028, var_limit=0.020, strategy="global_macro")
    
    # Simulate drawdown check
    monitor.check_drawdown_limit(drawdown=-0.12, drawdown_limit=0.10, strategy="carry")
    
    # Check exposure limits
    monitor.check_exposure_limits(
        gross=3.8, net=0.85, gross_limit=3.0, net_limit=0.5, strategy="ls_equity"
    )
    
    # Daily summary
    summary = monitor.daily_summary()
    print("   Operational summary:")
    print(f"      Total incidents: {len(monitor.incidents)}")
    print()
    
    print("=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
  
