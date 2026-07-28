#!/usr/bin/env python3
"""
Data Normalization Implementation Module

Provides production-grade normalization and scaling methods for economic data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Tuple, Dict, Optional


def min_max_scale(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, object]:
    """
    Apply Min-Max scaling.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    feature_range : tuple, default=(0, 1)
        Desired output range
        
    Returns:
    --------
    scaled_data : np.ndarray
        Min-max scaled data
    scaler : MinMaxScaler
        Fitted scaler for inverse transform
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Handle 1D data
    data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
    
    scaled_data = scaler.fit_transform(data_reshaped)
    
    # Return in original shape
    if data.ndim == 1:
        scaled_data = scaled_data.ravel()
    
    return scaled_data, scaler


def z_score_standardize(data: np.ndarray) -> Tuple[np.ndarray, object]:
    """
    Apply Z-score standardization (mean=0, std=1).
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
        
    Returns:
    --------
    standardized_data : np.ndarray
        Standardized data
    scaler : StandardScaler
        Fitted scaler for inverse transform
    """
    scaler = StandardScaler()
    
    data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
    standardized_data = scaler.fit_transform(data_reshaped)
    
    if data.ndim == 1:
        standardized_data = standardized_data.ravel()
    
    return standardized_data, scaler


def robust_scale(data: np.ndarray, quantile_range: Tuple[float, float] = (25.0, 75.0)) -> Tuple[np.ndarray, object]:
    """
    Apply robust scaling using median and IQR.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    quantile_range : tuple, default=(25.0, 75.0)
        Quantile range for IQR calculation
        
    Returns:
    --------
    scaled_data : np.ndarray
        Robustly scaled data
    scaler : RobustScaler
        Fitted scaler for inverse transform
    """
    scaler = RobustScaler(quantile_range=quantile_range)
    
    data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
    scaled_data = scaler.fit_transform(data_reshaped)
    
    if data.ndim == 1:
        scaled_data = scaled_data.ravel()
    
    return scaled_data, scaler


def log_transform(data: np.ndarray, 
                  method: str = 'natural', 
                  handle_nonpositive: str = 'add_one') -> Tuple[np.ndarray, Dict]:
    """
    Apply log transformation.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    method : str, default='natural'
        Log type: 'natural', 'log10', 'log2'
    handle_nonpositive : str, default='add_one'
        How to handle zeros/negatives:
        - 'add_one': Add 1 before transform
        - 'add_constant': Add |min| + 1
        - 'error': Raise error if non-positive
        
    Returns:
    ---------
    transformed_data : np.ndarray
        Log-transformed data
    params : dict
        Transform parameters for inverse
    """
    # Check for non-positive values
    if np.any(data <= 0):
        if handle_nonpositive == 'error':
            raise ValueError("Data contains non-positive values. Cannot apply log.")
        elif handle_nonpositive == 'add_one':
            shift = 1
        elif handle_nonpositive == 'add_constant':
            shift = abs(np.min(data)) + 1
        else:
            shift = 0
    else:
        shift = 0
    
    data_shifted = data + shift
    
    # Apply transformation
    if method == 'natural':
        transformed = np.log(data_shifted)
    elif method == 'log10':
        transformed = np.log10(data_shifted)
    elif method == 'log2':
        transformed = np.log2(data_shifted)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    params = {
        'method': method,
        'shift':  shift,
        'data_min': np.min(data),
        'data_max': np.max(data)
    }
    
    return transformed, params


def inverse_log_transform(transformed_data: np.ndarray, params: Dict) -> np.ndarray:
    """
    Reverse log transformation.
    
    Parameters:
    ------------
    transformed_data : np.ndarray
        Log-transformed data
    params : dict
        Parameters from log_transform
        
    Returns:
    ---------
    original_data : np.ndarray
        Data in original scale
    """
    method = params['method']
    shift = params['shift']
    
    if method == 'natural':
        original = np.exp(transformed_data)
    elif method == 'log10':
        original = np.power(10, transformed_data)
    elif method == 'log2':
        original = np.power(2, transformed_data)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return original - shift


def sqrt_transform(data: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply square root transformation.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
        
    Returns:
    ---------
    transformed_data : np.ndarray
        Square-root transformed data
    shift : float
        Value added before transformation
    """
    if np.any(data < 0):
        shift = abs(np.min(data))
    else:
        shift = 0
    
    transformed = np.sqrt(data + shift)
    
    return transformed, shift


def inverse_sqrt_transform(transformed_data: np.ndarray, shift: float) -> np.ndarray:
    """Reverse square root transformation."""
    return np.power(transformed_data, 2) - shift


def boxcox_transform(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Applx Box-Cox transformation.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data (must become positive)
        
    Returns:
    --------
    transformed_data : np.ndarray
        Box-Cox transformed data
    params : dict
        Lambda and shift parameters
    """
    # Ensure positive data
    if np.any(data <= 0):
        shift = abs(np.min(data)) + 1
        data_shifted = data + shift
    else:
        shift = 0
        data_shifted = data
    
    # Applx Box-Cox
    transformed, lambda_param = stats.boxcox(data_shifted)
    
    params = {
        'lambda': lambda_param,
        'shift': shift
    }
    
    return transformed, params


def inverse_boxcox_transform(transformed_data: np.ndarray, params: Dict) -> np.ndarray:
    """Reverse Box-Cox transformation."""
    lambda_param = params['lambda']
    shift = params['shift']
    
    if lambda_param == 0:
        original = np.exp(transformed_data)
    else:
        original = np.power(lambda_param * transformed_data + 1, 1 / lambda_param)
    
    return original - shift


def rank_transform(data: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Apply rank transformation.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    method : str, default='average'
        Tie handling: 'average', 'min', 'max', 'dense', 'ordinal'
        
    Returns:
    ---------
    ranks : np.ndarray
        Rank-transformed data
    """
    from scipy.stats import rankdata
    ranks = rankdata(data, method=method)
    return ranks


def normalize_auto(data: np.ndarray, 
                   purpose: str = 'comparison',
                   assessment: Optional[Dict] = None) -> Tuple[np.ndarray, str, Dict]:
    """
    Automatically select and apply best normalization method.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    purpose : str, default='comparison'
        Purpose: 'comparison', 'ml', 'visualization'
    assessment : dict, optional
        Pre-computed assessment
        
    Returns:
    --------
    normalized_data : np.ndarray
        Normalized data
    method_used : str
        Name of method applied
    details : dict
        Transform parameters and objects
    """
    # Run assessment if needed
    if assessment is None:
        from analyze_data import assess_data, recommend_normalization_method
        assessment = assess_data(data)
        method, reason = recommend_normalization_method(assessment, purpose)
    else:
        from analyze_data import recommend_normalization_method
        method, reason = recommend_normalization_method(assessment, purpose)
    
    details = {'reason': reason}
    
    # Apply recommended method
    if method == 'standard-scaling':
        normalized, scaler = z_score_standardize(data)
        details['scaler'] = scaler
        details['mean'] = scaler.mean_[0] if hasattr(scaler, 'mean_') else np.mean(data)
        details['std'] = scaler.scale_[0] if hasattr(scaler, 'scale_') else np.std(data)
    
    elif method == 'min-max-scaling':
        normalized, scaler = min_max_scale(data)
        details['scaler'] = scaler
        details['min'] = scaler.data_min_[0] if hasattr(scaler, 'data_min_') else np.min(data)
        details['max'] = scaler.data_max_[0] if hasattr(scaler, 'data_max_') else np.max(data)
    
    elif method == 'robust-scaling':
        normalized, scaler = robust_scale(data)
        details['scaler'] = scaler
        details['median'] = scaler.center_[0] if hasattr(scaler, 'center_') else np.median(data)
        details['iqr'] = scaler.scale_[0] if hasattr(scaler, 'scale_') else (np.percentile(data, 75) - np.percentile(data, 25))
    
    elif method == 'log-transform':
        normalized, params = log_transform(data)
        details.update(params)
        details['can_inverse'] = True
    
    elif method == 'log-transform-then-standard':
        # Two-step: log then standardize
        log_data, log_params = log_transform(data)
        normalized, scaler = z_score_standardize(log_data)
        details['log_params'] = log_params
        details['scaler'] = scaler
        details['two_step'] = True
    
    elif method == 'boxcox':
        normalized, params = boxcox_transform(data)
        details.update(params)
        details['can_inverse'] = True
    
    else:
        # Default to standard scaling
        normalized, scaler = z_score_standardize(data)
        details['scaler'] = scaler
        method = 'standard-scaling'
    
    return normalized, method, details


if __name__ == "__main__":
    print("Normalization Module - Test Suite")
    print("=" * 50)
    
    # Test data (right-skewed income-like data)
    np.random.seed(42)
    test_data = np.random.lognormal(mean=10, sigma=1, size=100)
    
    print(f"\nOriginal Data:")
    print(f"  Mean: {np.mean(test_data):.2f}")
    print(f"  Std: {np.std(test_data):.2f}")
    print(f"  Skewness: {stats.skew(test_data):.3f}")
    print(f"  Range: [{np.min(test_data):.2f}, {np.max(test_data):.2f}]")
    
    # Test each method
    print("\n1. Min-Max Scaling:")
    scaled, scaler = min_max_scale(test_data)
    print(f"   Range: [{np.min(scaled):.2f}, {np.max(scaled):.2f}]")
    
    print("\n2. Z-Score Standardization:")
    standardized, scaler = z_score_standardize(test_data)
    print(f"   Mean: {np.mean(standardized):.6f}")
    print(f"   Std: {np.std(standardized, ddof=0):.6f}")
    
    print("\n3. Robust Scaling:")
    robust, scaler = robust_scale(test_data)
    print(f"   Median: {np.median(robust):")
    
    print("\n4. Log Transformation:")
    log_data, params = log_transform(test_data)
    print(f"   Skewness reduced: {stats.skew(test_data):")
    
    # Test inverse
    recovered = inverse_log_transform(log_data, params)
    print(f"  Inverse accuracy: {np.allclose(test_data, recovered)}")
    
    print("\n5. Automatic Selection:")
    normalized, method, details = normalize_auto(test_data, purpose='ml')
    print(f"   Method selected: {method}")
    print(f"   Reason: {details['reason']}")