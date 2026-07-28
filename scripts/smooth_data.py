#!/usr/bin/env python3
"""
Data Smoothing Implementation Module

Provides smoothing and filtering methods for economic time series.
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, Dict, Optional


def henderson_filter(data: np.ndarray, length: int = 13) -> np.ndarray:
    """
    Apply Henderson filter for trend extraction.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series
    length : int, default=13
        Filter length (5, 9, 13, or 23)
        
    Returns:
    --------
    smoothed : np.ndarray
        Henderson-filtered series
    """
    if length not in [5, 9, 13, 23]:
        raise ValueError("Henderson filter length must be 5, 9, 13, or 23")
    
    weights = _get_henderson_weights(length)
    half_length = length // 2
    
    # Apply filter to interior points
    smoothed = np.full_like(data, np.nan, dtype=float)
    for i in range(half_length, len(data) - half_length):
        smoothed[i] = np.sum(weights * data[i-half_length:i+half_length+1])
    
    # Handle endpoints with asymmetric filters
    for i in range(half_length):
        # Left endpoint: use rightmost n_available weights (missing data to the left)
        n_available = i + half_length + 1
        if n_available >= 3:
            w = weights[half_length - i:]  # Rightmost n_available weights
            w = w / np.sum(w)  # Renormalize
            smoothed[i] = np.sum(w * data[:n_available])
        
        # Right endpoint: use leftmost n_available weights (missing data to the right)
        j = len(data) - 1 - i
        n_available = half_length + i + 1
        if n_available >= 3:
            w = weights[:n_available]  # Leftmost n_available weights
            w = w / np.sum(w)
            smoothed[j] = np.sum(w * data[j - half_length:])
    
    return smoothed


def _get_henderson_weights(length: int) -> np.ndarray:
    """Get Henderson filter weights."""
    weights_dict = {
        5: np.array([-0.073, 0.294, 0.558, 0.294, -0.073]),
        9: np.array([-0.041, -0.010, 0.119, 0.267, 0.330, 0.267, 0.119, -0.010, -0.041]),
        13: np.array([-0.019, -0.028, 0.000, 0.066, 0.147, 0.214, 0.240, 0.214, 0.147, 0.066, 0.000, -0.028, -0.019]),
        23: np.array([-0.014, -0.017, -0.016, -0.011, -0.004, 0.007, 0.018, 0.031, 0.045,
                     0.058, 0.068, 0.075, 0.068, 0.058, 0.045, 0.031, 0.018, 0.007, -0.004,
                     -0.011, -0.016, -0.017, -0.014])
    }
    return weights_dict[length]


def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply simple exponential smoothing.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series
    alpha : float, default=0.3
        Smoothing parameter (0 < alpha < 1)
        Higher = more responsive
        
    Returns:
    --------
    smoothed : np.ndarray
        Exponentially smoothed series
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    
    return smoothed


def optimal_exponential_alpha(data: np.ndarray) -> float:
    """
    Find optimal alpha by minimizing one-step-ahead forecast error.
    
    Returns:
    --------
    alpha : float
        Optimal smoothing parameter
    """
    from scipy.optimize import minimize_scalar
    
    def mse(alpha):
        smoothed = exponential_smoothing(data, alpha)
        errors = data[1:] - smoothed[:-1]
        return np.mean(errors**2)
    
    result = minimize_scalar(mse, bounds=(0.01, 0.99), method='bounded')
    return result.x


def double_exponential_smoothing(data: np.ndarray, 
                                 alpha: float = 0.3, 
                                 beta: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply double exponential smoothing (Holt's method).
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series
    alpha : float, default=0.3
        Level smoothing parameter
    beta : float, default=0.1
        Trend smoothing parameter
        
    Returns:
    --------
    smoothed : np.array
        Smoothed series
    trend : np.array
        Trend component
    """
    n = len(data)
    level = np.zeros(n)
    trend = np.zeros(n)
    smoothed = np.zeros(n)
    
    # Initialize
    level[0] = data[0]
    trend[0] = data[1] - data[0] if n > 1 else 0
    smoothed[0] = level[0]
    
    # Apply recursively
    for t in range(1, n):
        level[t] = alpha * data[t] + (1 - alpha) * (level[t-1] + trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
        smoothed[t] = level[t]
    
    return smoothed, trend


def simple_moving_average(data: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Calculate simple moving average.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series
    window : int, default=12
        Window size
        
    Returns:
    --------
    ma : np.ndarray
        Moving average
    """
    ma = np.full_like(data, np.nan, dtype=float)
    
    for i in range(window - 1, len(data)):
        ma[i] = np.mean(data[i-window+1:i+1])
    
    return ma


def centered_moving_average(data: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Calculate centered moving average.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series
    window : int, default=12
        Window size
        
    Returns:
    --------
    ma : np.ndarray
        Centered moving average
    """
    if window % 2 == 0:
        # Even window: two-step averaging
        ma1 = simple_moving_average(data, window)
        ma2 = np.full_like(ma1, np.nan)
        for i in range(1, len(ma1)):
            if not np.isnan(ma1[i-1]) and not np.isnan(ma1[i]):
                ma2[i] = (ma1[i-1] + ma1[i]) / 2
        return ma2
    else:
        # Odd window: direct centering
        half = window // 2
        ma = np.full_like(data, np.nan, dtype=float)
        for i in range(half, len(data) - half):
            ma[i] = np.mean(data[i-half:i+half+1])
        return ma


def weighted_moving_average(data: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Calculate weighted moving average with linear weights.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series
    window : int, default=12
        Window size
        
    Returns:
    --------
    wma : np.ndarray
        Weighted moving average
    """
    weights = np.arange(1, window + 1)
    weights = weights / np.sum(weights)
    
    wma = np.full_like(data, np.nan, dtype=float)
    
    for i in range(window - 1, len(data)):
        wma[i] = np.sum(weights * data[i-window+1:i+1])
    
    return wma


def loess_smoothing(data: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """
    Apply LOESS (locally weighted scatterplot smoothing).
    
    Parameters:
    ------------
    data : np.ndarray
        Input time series
    frac : float, default=0.1
        Fraction of data for smoothing (0 < frac ≤ 1)
        
    Returns:
    --------
    smoothed : np.ndarray
        LOESS-smoothed series
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    x = np.arange(len(data))
    smoothed = lowess(data, x, frac=frac, it=3, return_sorted=False)
    
    return smoothed


def savitzky_golay_filter(data: np.ndarray, 
                          window_length: int = 11, 
                          polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter.
    
    Parameters:
    ------------
    data : np.ndarray
        Input time series
    window_length : int, default=11
        Filter window (must be odd, ≥ polyorder+2)
    polyorder : int, default=3
        Polynomial order (1-5)
        
    Returns:
    --------
    smoothed : np.ndarray
        Savitzky-Golay filtered series
    """
    if window_length % 2 == 0:
        window_length += 1
    
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    
    smoothed = signal.savgol_filter(data, window_length, polyorder, mode='nearest')
    
    return smoothed


def smooth_auto(data: np.ndarray, 
                is_time_series: bool = True,
                assessment: Optional[Dict] = None) -> Tuple[np.ndarray, str, Dict]:
    """
    Automatically select and apply smoothing method.
    
    Parameters:
    ------------
    data : np.ndarray
        Input data
    is_time_series : bool, default=True
        Whether data is time series
    assessment : dict, optional
        Pre-computed data assessment
        
    Returns:
    ---------
    smoothed : np.ndarray
        Smoothed data
    method_used : str
        Name of method applied
    details : dict
        Method parameters
    """
    n_obs = len(data)
    
    if assessment is None:
        from analyze_data import assess_data, recommend_smoothing_method
        assessment = assess_data(data)
        method, reason = recommend_smoothing_method(assessment, is_time_series)
    else:
        from analyze_data import recommend_smoothing_method
        method, reason = recommend_smoothing_method(assessment, is_time_series)
    
    details = {'reason': reason, 'n_obs': n_obs}
    
    # Apply recommended method
    if method == 'henderson':
        # Choose length based on data size
        if n_obs >= 100:
            length = 13
        elif n_obs >= 50:
            length = 9
        else:
            length = 5
        smoothed = henderson_filter(data, length=length)
        details['length'] = length
    
    elif method == 'exponential-smoothing':
        alpha = optimal_exponential_alpha(data)
        smoothed = exponential_smoothing(data, alpha=alpha)
        details['alpha'] = alpha
    
    elif method == 'moving-average':
        # Choose window based on frequency
        if n_obs >= 52:
            window = 12  # Annual smoothing for monthly
        elif n_obs >= 24:
            window = 4   # Quarterly smoothing
        else:
            window = 3   # Short-term
        smoothed = simple_moving_average(data, window=window)
        details['window'] = window
    
    elif method == 'loess':
        frac = min(0.2, 10 / n_obs)  # Adaptive bandwidth
        smoothed = loess_smoothing(data, frac=frac)
        details['frac'] = frac
    
    elif method == 'savitzky-golay':
        window = min(11, n_obs // 5)
        if window % 2 == 0:
            window += 1
        smoothed = savitzky_golay_filter(data, window_length=window, polyorder=3)
        details['window_length'] = window
        details['polyorder'] = 3
    
    else:
        # Default to simple MA
        window = min(12, n_obs // 3)
        smoothed = simple_moving_average(data, window=window)
        details['window'] = window
        method = 'moving-average'
    
    return smoothed, method, details


if __name__ == "__main__":
    print("Smoothing Module - Test Suite")
    print("=" * 50)
    
    # Generate test series with noise
    np.random.seed(42)
    n = 100
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 5, n)
    test_data = trend + noise
    
    print(f"\nTest Data: {n} observations")
    print(f"  Trend range: [100, 120]")
    print(f"  Noise std: 5")
    print(f"  Observed std: {np.std(test_data):.2f}")
    
    # Test each method
    print("\n1. Henderson Filter (13-term):")
    smooth_henderson = henderson_filter(test_data, length=13)
    valid = smooth_henderson[~np.isnan(smooth_henderson)]
    print(f"   Valid points: {len(valid)}/{n}")
    print(f"   Smoothed std: {np.std(valid):.2f}")
    
    print("\n2. Exponential Smoothing:")
    alpha_opt = optimal_exponential_alpha(test_data)
    smooth_exp = exponential_smoothing(test_data, alpha=alpha_opt)
    print(f"  Optimal alpha: {alpha_opt:.3f}")
    print(f"   Smoothed std: {np.std(smooth_exp):.2f}")
    
    print("\n3. Simple Moving Average (12-period):")
    smooth_ma = simple_moving_average(test_data, window=12)
    valid = smooth_ma[~np.isnan(smooth_ma)]
    print(f"   Valid points: {len(valid)}/{n}")
    print(f"   Smoothed std: {np.std(valid):.2f}")
    
    print("\n4. LOESS (frac=0.1):")
    smooth_loess = loess_smoothing(test_data, frac=0.1)
    print(f"   Smoothed std: {np.std(smooth_loess):.2f}")
    
    print("\n5. Savitzky-Golay (11-term, order 3):")
    smooth_sg = savitzky_golay_filter(test_data, window_length=11, polyorder=3)
    print(f"   Smoothed std: {np.std(smooth_sg):.2f}")
    
    print("\n6. Automatic Selection:")
    smooth_auto_result, method, details = smooth_auto(test_data)
    print(f"   Method selected: {method}")
    print(f"   Reason: {details['reason']}")