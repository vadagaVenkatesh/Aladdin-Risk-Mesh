import pandas as pd
import numpy as np

class AlphaResearch:
    def __init__(self):
        pass
        
    def generate_macro_signal(self, data):
        """Generates a signal based on moving average crossover."""
        short_window = 20
        long_window = 50
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        signals['short_mavg'] = data.rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data.rolling(window=long_window, min_periods=1, center=False).mean()
        
        # Signal generation
        signals.iloc[short_window:, signals.columns.get_loc('signal')] = np.where(
            signals['short_mavg'].iloc[short_window:] > signals['long_mavg'].iloc[short_window:], 1.0, 0.0
        )
        
        signals['positions'] = signals['signal'].diff()
        
        return signals

if __name__ == "__main__":
    alpha = AlphaResearch()
    print("AlphaResearch initialized.")
