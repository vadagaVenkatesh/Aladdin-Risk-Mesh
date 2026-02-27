"""ml — Machine Learning Layer

LSTM-based macro regime classifier trained on rolling windows.
Uses conformal prediction for calibrated uncertainty.
Numba @njit JIT compilation on all compute-intensive paths.
"""
