from src.data_loader import DataLoader
from src.alpha_research import AlphaResearch
from src.risk_engine import RiskEngine
from src.backtester import Backtester
import datetime

def run_mvp():
    # 1. Initialize Components
    loader = DataLoader(config_path='config/settings.yaml')
    alpha = AlphaResearch()
    risk = RiskEngine(confidence_level=0.99)
    tester = Backtester(initial_capital=100000.0)
    
    # 2. Fetch Data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    start_date = '2023-01-01'
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    price_data = loader.fetch_equity_data(tickers, start_date, end_date)
    
    # For MVP, we use SPY for signal generation
    spy_data = price_data['SPY']
    
    # 3. Generate Signals
    signals = alpha.generate_macro_signal(spy_data)
    
    # 4. Run Backtest
    results = tester.run_backtest(spy_data, signals)
    
    # 5. Risk Assessment
    var_99 = risk.calculate_var(results['returns'].dropna())
    
    print(f"--- {loader.config['fund_name']} Report ---")
    print(f"Final Portfolio Value: ${results['total'].iloc[-1]:.2f}")
    print(f"Total Return: {(results['total'].iloc[-1]/100000.0 - 1)*100:.2f}%")
    print(f"99% Value at Risk (Daily): {var_99*100:.2f}%")

if __name__ == "__main__":
    run_mvp()
