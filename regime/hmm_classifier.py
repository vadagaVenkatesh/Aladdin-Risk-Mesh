import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Tuple, List

class RegimeClassifierHMM:
    """
    Gaussian Hidden Markov Model for detecting market regimes.
    """
    def __init__(self, n_regimes: int = 4, covariance_type: str = "full"):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type=covariance_type, 
            n_iter=1000,
            random_state=42
        )
        self.regime_map = {
            0: "Goldilocks",
            1: "Inflationary Growth",
            2: "Stagflation",
            3: "Recession"
        }

    def fit(self, features: pd.DataFrame):
        """
        Train the HMM on historical macro and market features.
        """
        self.model.fit(features.values)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict the current regime.
        """
        regimes = self.model.predict(features.values)
        return pd.Series(regimes, index=features.index, name="regime")

    def get_regime_characteristics(self, returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
        """
        Analyze return characteristics per regime.
        """
        df = pd.DataFrame({'returns': returns, 'regime': regimes})
        stats = df.groupby('regime')['returns'].agg(['mean', 'std', 'count'])
        stats['sharpe'] = stats['mean'] / stats['std'] * np.sqrt(252)
        return stats

if __name__ == "__main__":
    # Demo logic
    dates = pd.date_range('2010-01-01', periods=1000)
    # Mock features: Volatility, Yield Curve Slope, Inflation
    mock_features = pd.DataFrame(
        np.random.normal(0, 1, (1000, 3)),
        index=dates,
        columns=['vol', 'slope', 'cpi']
    )
    
    classifier = RegimeClassifierHMM()
    classifier.fit(mock_features)
    pred_regimes = classifier.predict(mock_features)
    
    print("Detected Regime Counts:")
    print(pred_regimes.value_counts())
