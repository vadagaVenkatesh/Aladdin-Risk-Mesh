import numpy as np
import pandas as pd
from typing import Tuple

class VolatilityRegimeFilter:
    \"\"\"
    Volatility-based regime filter using trailing standard deviation.
    Ownership: Copyright (c) 2026 VDG Venkatesh. All Rights Reserved.
    PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED USE PROHIBITED.
    \"\"\"
    
    def __init__(self, window: int = 21, threshold_z: float = 1.5):
        self.window = window
        self.threshold_z = threshold_z
        self.mean_vol = None
        self.std_vol = None

    def calculate_volatility(self, returns: pd.Series) -> pd.Series:
        \"\"\"Calculate trailing annualized volatility.\"\"\"
        return returns.rolling(window=self.window).std() * np.sqrt(252)

    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        \"\"\"
        Detect regimes based on volatility spikes.
        0: Low Volatility (Normal)
        1: High Volatility (Stress)
        \"\"\"
        vol = self.calculate_volatility(returns)
        
        if self.mean_vol is None:
            self.mean_vol = vol.mean()
            self.std_vol = vol.std()
            
        threshold = self.mean_vol + (self.threshold_z * self.std_vol)
        regimes = (vol > threshold).astype(int)
        
        return regimes

    def get_summary(self, returns: pd.Series) -> pd.DataFrame:
        vol = self.calculate_volatility(returns)
        regimes = self.detect_regimes(returns)
        
        df = pd.DataFrame({'vol': vol, 'regime': regimes}, index=returns.index)
        return df.groupby('regime')['vol'].agg(['mean', 'std', 'count'])

if __name__ == \"__main__\":
    # Demo logic
    dates = pd.date_range('2020-01-01', periods=500)
    low_vol = np.random.normal(0, 0.01, 400)
    high_vol = np.random.normal(0, 0.05, 100)
    returns = pd.Series(np.concatenate([low_vol, high_vol]), index=dates)
    
    filter = VolatilityRegimeFilter()
    summary = filter.get_summary(returns)
    print(\"Volatility Regime Summary:\")
    print(summary)
