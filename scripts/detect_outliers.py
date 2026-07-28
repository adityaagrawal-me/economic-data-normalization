#!/usr/bin/env python3
"""
Outlier Detection Implementation Module

Provides production-grade outlier detection methods for economic data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using Z-score method.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    threshold : float, default=3.0
        Z-score threshold (typically 2.5-3.0)
        
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean array (True = outlier)
    z_scores : np.ndarray
        Z-scores for all points
    """
    # Remove NaN for calculation
    valid_mask = ~np.isnan(data)
    z_scores = np.full_like(data, np.nan, dtype=float)
    
    if np.sum(valid_mask) > 0:
        z_scores[valid_mask] = np.abs(stats.zscore(data[valid_mask]))
    
    outlier_mask = z_scores > threshold
    
    return outlier_mask, z_scores


def detect_outliers_iqr(data: np.ndarray, k: float = 1.5) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    k : float, default=1.5
        IQR multiplier (1.5=Tukey, 2.0=conservative, 3.0=very conservative)
        
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean array (True = outlier)
    bounds : tuple
        (lower_bound, upper_bound)
    """
    valid_data = data[~np.isnan(data)]
    
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    return outlier_mask, (lower_bound, upper_bound)


def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float, float, float]:
    """
    Perform Grubbs' test for single outlier detection.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data (should be approximately normal)
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    is_outlier : bool
        True if outlier detected
    extreme_value : float
        Most extreme value
    G_stat : float
        Grubbs test statistic
    p_value : float
        P-value of test
    """
    n = len(data)
    
    if n < 3:
        return False, np.nan, np.nan, 1.0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return False, np.nan, np.nan, 1.0
    
    # Find most extreme value
    abs_diff = np.abs(data - mean)
    max_idx = np.argmax(abs_diff)
    extreme_value = data[max_idx]
    G_stat = abs_diff[max_idx] / std
    
    # Critical value from t-distribution
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_critical = ((n - 1) * np.sqrt(t_dist**2)) / \
                 (np.sqrt(n) * np.sqrt(n - 2 + t_dist**2))
    
    is_outlier = G_stat > G_critical
    
    # Approximate p-value
    t_stat = G_stat * np.sqrt(n * (n - 2)) / (n - 1)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    return is_outlier, extreme_value, G_stat, p_value


def iterative_grubbs(data: np.ndarray, alpha: float = 0.05, max_outliers: int = 10) -> Tuple[np.ndarray, List[float]]:
    """
    Iteratively apply Grubbs' test to detect multiple outliers.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    alpha : float, default=0.05
        Significance level
    max_outliers : int, default=10
        Maximum outliers to detect
        
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean array indicating all detected outliers
    detected_values : list
        Values identified as outliers
    """
    working_data = data.copy()
    outlier_mask = np.zeros(len(data), dtype=bool)
    detected_values = []
    
    for _ in range(max_outliers):
        is_outlier, extreme_val, G_stat, p_val = grubbs_test(working_data, alpha)
        
        if not is_outlier:
            break
        
        # Mark outlier in original data
        outlier_idx = np.where(data == extreme_val)[0]
        if len(outlier_idx) > 0:
            outlier_mask[outlier_idx] = True
            detected_values.append(extreme_val)
        
        # Remove from working data
        working_data = working_data[working_data != extreme_val]
        
        if len(working_data) < 3:
            break
    
    return outlier_mask, detected_values


def detect_outliers_dbscan(data: np.ndarray, 
                           eps: Optional[float] = None, 
                           min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using DBSCAN clustering.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data (can be multidimensional)
    eps : float, optional
        Maximum distance between points
        If None, automatically determined
    min_samples : int, default=5
        Minimum points to form dense region
        
    Returns:
    ---------
    outlier_mask : np.ndarray
        Boolean array (True = outlier)
    labels : np.ndarray
        Cluster labels (-1 = noise/outlier)
    """
    # Reshape if 1D
    if data.ndim == 1:
        data_reshaped = data.reshape(-1, 1)
    else:
        data_reshaped = data
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    # Auto-determine epsilon if not provided
    if eps is None:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(data_scaled)
        distances, _ = nbrss.kneighbors(data_scaled)
        eps = np.percentile(distances[:, -1], 90)
    
    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
    labels = clustering.labels_
    
    # Outliers have label -1
    outlier_mask = labels == -1
    
    return outlier_mask, labels


def detect_outliers_modified_zscore(data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using Modified Z-score (MAD-based).
    
    More robust than standard Z-score. Uses median and MAD.
    
    Parameters:
    ------------
    data : np.ndarray
        Input data
    threshold : float, default=3.5
        Modified Z-score threshold (typically 3.5)
        
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean array (True = outlier)
    modified_z_scores : np.ndarray
        Modified Z-scores
    """
    valid_data = data[~np.isnan(data)]
    median = np.median(valid_data)
    mad = np.median(np.abs(valid_data - median))
    
    # Handle zero MAD
    if mad == 0:
        mad = np.mean(np.abs(valid_data - median))
        if mad == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
    
    # Calculate modified Z-scores
    modified_zscores = 0.6745 * (data - median) / mad
    outlier_mask = np.abs(modified_z_scores) > threshold
    
    return outlier_mask, modified_zscores


def treat_outliers(data: np.ndarray, 
                  outlier_mask: np.ndarray, 
                   method: str = 'remove') -> np.ndarray:
    """
    Treat detected outliers.
    
    Parameters:
    -----------
    data : np.ndarray
        Original data
    outlier_mask : np.ndarray
        Boolean mask indicating outliers
    method : str
        Treatment method:
        - 'remove': Delete outliers (NaN)
        - 'winsorize': Cap at threshold
        - 'median': Replace with median
        - 'mean': Replace with mean
        
    Returns:
    --------
    treated_data : np.ndarray
        Data with outliers treated
    """
    treated = data.copy()
    
    if method == 'remove':
        treated[outlier_mask] = np.nan
    
    elif method == 'winsorize':
        valid_data = data[~outlier_mask]
        lower = np.percentile(valid_data, 5)
        upper = np.percentile(valid_data, 95)
        treated = np.clip(data, lower, upper)
    
    elif method == 'median':
        median_val = np.median(data[~outlier_mask])
        treated[outlier_mask] = median_val
    
    elif method == 'mean':
        mean_val = np.mean(data[~outlier_mask])
        treated[outlier_mask] = mean_val
    
    return treated


def detect_outliers_auto(data: np.ndarray, 
                         assessment: Optional[Dict] = None) -> Tuple[np.ndarray, str, Dict]:
    """
    Automatically select and apply best outlier detection method.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    assessment : dict, optional
        Pre-computed data assessment
        
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean mask of outliers
    method_used : str
        Name of method applied
    details : dict
        Method-specific details
    """
    # Run assessment if not provided
    if assessment is None:
        from analyze_data import assess_data, recommend_outlier_method
        assessment = assess_data(data)
        method, reason = recommend_outlier_method(assessment)
    else:
        from analyze_data import recommend_outlier_method
        method, reason = recommend_outlier_method(assessment)
    
    # Apply recommended method
    if method == 'z-score':
        outlier_mask, zscores = detect_outliers_zscore(data)
        details = {'z_scores': z_scores, 'threshold': 3.0}
    
    elif method == 'iqr':
        outlier_mask, bounds = detect_outliers_iqr(data)
        details = {'bounds': bounds, 'k': 1.5}
    
    elif method == 'grubbs':
        outlier_mask, detected_values = iterative_grubbs(data)
        details = {'detected_values': detected_values, 'alpha': 0.05}
    
    elif method == 'dbscan':
        outlier_mask, labels = detect_outliers_dbscan(data)
        details = {'labels': labels}
    
    else:
        # Default to IQR
        outlier_mask, bounds = detect_outliers_iqr(data)
        details = {'bounds': bounds, 'k': 1.5}
        method = 'iqr'
    
    details['reason'] = reason
    details['n_outliers'] = np.sum(outlier_mask)
    details['outlier_pct'] = np.sum(outlier_mask) / len(data) * 100
    
    return outlier_mask, method, details detect outliers_modified_zscore(data)
    mask_mod, mod_z = detect_outliers_modified_zscore(test_data)
    print(f"   Outliers detected: {np.sum(mask_mod)}")
    
    print("\n5. Automatic Selection:")
    mask_auto, method, details = detect_outliers_auto(test_data)
    print(f"   Method selected: {method}")
    print(f"   Reason: {details['reason']}")
    print(f"   Outliers detected: {details['n_outliers']}")