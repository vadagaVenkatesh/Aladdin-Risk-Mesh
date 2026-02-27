"""
lstm_regime.py

PROPRIETARY AND CONFIDENTIAL
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

This software and associated documentation files are the proprietary 
and confidential information of VDG Venkatesh. Unauthorized copying,
modification, distribution, or use of this software, via any medium,
is strictly prohibited without express written permission.

NO LICENSE IS GRANTED. This code may not be used, reproduced, or 
incorporated into any other projects without explicit authorization.
For licensing inquiries, contact the copyright holder.

LSTM-Based Market Regime Detection
- Time-series classification for regime shifts
- Combines macro indicators, vol, and rates
- Supervised learning with historical labels
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LSTMRegimeClassifier:
    """
    LSTM-based classifier for market regime detection.
    Predicts regimes: Low Vol, Medium Vol, High Vol (or custom).
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        sequence_length: int = 30,
        n_features: int = 10,
        lstm_units: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM regime classifier.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regime classes
        sequence_length : int
            Lookback window for LSTM
        n_features : int
            Number of input features
        lstm_units : int
            LSTM hidden units
        dropout : float
            Dropout rate
        """
        self.n_regimes = n_regimes
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def build_model(self):
        """
        Build LSTM architecture.
        """
        model = keras.Sequential([
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            ),
            layers.Dropout(self.dropout),
            
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(self.dropout),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout / 2),
            
            layers.Dense(self.n_regimes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM training/inference.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray, optional
            Labels (n_samples,)
        
        Returns:
        --------
        tuple : (X_sequences, y_sequences)
        """
        n_samples = X.shape[0]
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, n_samples):
            X_seq.append(X[i - self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0
    ):
        """
        Train LSTM classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Regime labels (n_samples,)
        epochs : int
        batch_size : int
        validation_split : float
        verbose : int
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Train
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        self.is_fitted = True
        return history
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict regime labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray : Predicted regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self.prepare_sequences(X_scaled)
        
        # Predict
        probs = self.model.predict(X_seq, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        # Pad predictions to match input length
        padded_preds = np.full(X.shape[0], preds[0])
        padded_preds[self.sequence_length:] = preds
        
        return padded_preds
    
    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict regime probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray : Probabilities (n_samples, n_regimes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled)
        
        probs = self.model.predict(X_seq, verbose=0)
        
        # Pad
        padded_probs = np.zeros((X.shape[0], self.n_regimes))
        padded_probs[:self.sequence_length] = probs[0]
        padded_probs[self.sequence_length:] = probs
        
        return padded_probs
    
    def save_model(self, filepath: str):
        """Save model weights."""
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model weights."""
        self.model = keras.models.load_model(filepath)
        self.is_fitted = True


def create_regime_labels_from_volatility(
    volatility: pd.Series,
    low_threshold: float = 0.15,
    high_threshold: float = 0.30
) -> np.ndarray:
    """
    Create regime labels based on volatility thresholds.
    
    Parameters:
    -----------
    volatility : pd.Series
        Historical volatility
    low_threshold : float
        Threshold for Low Vol regime
    high_threshold : float
        Threshold for High Vol regime
    
    Returns:
    --------
    np.ndarray : Labels (0: Low Vol, 1: Medium Vol, 2: High Vol)
    """
    labels = np.zeros(len(volatility), dtype=int)
    labels[volatility < low_threshold] = 0  # Low Vol
    labels[(volatility >= low_threshold) & (volatility < high_threshold)] = 1  # Medium
    labels[volatility >= high_threshold] = 2  # High Vol
    
    return labels


# ========================================
# Demo
# ========================================
if __name__ == "__main__":
    print("=" * 80)
    print("LSTM Regime Classifier Demo")
    print("=" * 80)
    print()
    
    # Generate mock data
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    # Features: returns, vol, VIX, rates, momentum
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic regimes based on first feature (vol proxy)
    vol_proxy = np.abs(X[:, 1])
    y = create_regime_labels_from_volatility(
        pd.Series(vol_proxy),
        low_threshold=0.5,
        high_threshold=1.5
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Regime distribution: {np.bincount(y)}")
    print()
    
    # Train LSTM
    classifier = LSTMRegimeClassifier(
        n_regimes=3,
        sequence_length=30,
        n_features=n_features,
        lstm_units=32
    )
    
    print("Training LSTM classifier...")
    history = classifier.fit(
        X, y,
        epochs=20,
        batch_size=16,
        verbose=0
    )
    
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
    print()
    
    # Predict
    predictions = classifier.predict(X)
    probs = classifier.predict_proba(X)
    
    print("Prediction sample (last 10):")
    print(f"  True labels: {y[-10:]}")
    print(f"  Predictions: {predictions[-10:]}")
    print(f"  Probabilities (last): {probs[-1]}")
    
    print("\n" + "=" * 80)
  
