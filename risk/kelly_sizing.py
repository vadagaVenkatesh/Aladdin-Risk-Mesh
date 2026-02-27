import numpy as np
import pandas as pd
from typing import Dict

class KellySizer:
    \"\"\"
    Implementation of the Kelly Criterion for optimal position sizing.
    Ownership: Copyright (c) 2026 VDG Venkatesh. All Rights Reserved.
    PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED USE PROHIBITED.
    \"\"\"
    
    def __init__(self, fraction: float = 0.5):
        \"\"\"
        fraction: Kelly fraction (e.g., 0.5 for Half-Kelly).
        \"\"\"
        self.fraction = fraction

    def calculate_kelly_fraction(self, win_prob: float, win_loss_ratio: float) -> float:
        \"\"\"
        k = (p*b - q) / b
        where p = win_prob, b = win_loss_ratio, q = 1-p
        \"\"\"
        if win_loss_ratio <= 0:
            return 0.0
        
        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
        return max(0.0, kelly * self.fraction)

    def size_portfolio(self, signals: pd.Series, strategy_stats: Dict) -> pd.Series:
        \"\"\"
        Apply Kelly sizing to a set of signals.
        \"\"\"
        win_prob = strategy_stats.get('win_rate', 0.5)
        win_loss_ratio = strategy_stats.get('win_loss_ratio', 1.0)
        
        k_fraction = self.calculate_kelly_fraction(win_prob, win_loss_ratio)
        return signals * k_fraction

if __name__ == \"__main__\":
    # Demo logic
    sizer = KellySizer(fraction=0.5)
    stats = {'win_rate': 0.55, 'win_loss_ratio': 1.2}
    signals = pd.Series([1.0, -1.0, 1.0], index=['AAPL', 'MSFT', 'GOOGL'])
    
    sized_positions = sizer.size_portfolio(signals, stats)
    print(\"Kelly Sized Positions (Half-Kelly):\")
    print(sized_positions)
