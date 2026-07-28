#!/usr/bin/env python3
"""
Data Assessment and Method Recommendation Module

Analyzes economic data characteristics and recommends appropriate
cleaning, normalization, and smoothing methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def assess_data(data: np.ndarray, data_name: str = "data") -> Dict:
    """
    Comprehensive data assessment for method selection.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    data_name : str, default="data"
        Name of the dataset for reporting
        
    Returns:
    --------
    assessment : dict
        Complete data characteristics report
    """
    # Remove NaN values for analysis
    clean_data = data[~np.isnan(data)]
    
    assessment = {
        'name': data_name,
        'n_obs': len(data),
        'n_valid': len(clean_data),
        'n_missing': len(data) - len(clean_data),
        'missing_pct': (len(data) - len(clean_data)) / len(data) * 100,
    }
    
    # Descriptive statistics
    assessment.update({
        'mean': np.mean(clean_data),
        'median': np.median(clean_data),
        'std': np.std(clean_data, ddof=1),
        'min': np.min(clean_data),
        'max': np.max(clean_data),
        'range': np.max(clean_data) - np.min(clean_data),
        'q1': np.percentile(clean_data, 25),
        'q3': np.percentile(clean_data, 75),
        'iqr': np.percentile(clean_data, 75) - np.percentile(clean_data, 25)
    })
    
    # Distribution characteristics
    assessment['skewness'] = stats.skew(clean_data)
    assessment['kurtosis'] = stats.kurtosis(clean_data)
    
    # Normality test (Shapiro-Wilk for n < 5000)
    if len(clean_data) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(clean_data)
        assessment['normality_test'] = 'shapiro-wilk'
        assessment['normality_stat'] = shapiro_stat
        assessment['normality_p'] = shapiro_p
        assessment['is_normal'] = shapiro_p > 0.05
    else:
        # Use Kolmogorov-Smirnov for large samples
        ks_stat, ks_p = stats.kstest(clean_data, 'norm', 
                                     args=(assessment['mean'], assessment['std']))
        assessment['normality_test'] = 'kolmogorov-smirnov'
        assessment['normality_stat'] = ks_stat
        assessment['normality_p'] = ks_p
        assessment['is_normal'] = ks_p > 0.05
    
    # Distribution classification
    if abs(assessment['skewness']) < 0.5:
        assessment['distribution'] = 'symmetric'
    elif assessment['skewness'] > 0.5:
        assessment['distribution'] = 'right-skewed'
    else:
        assessment['distribution'] = 'left-skewed'
    
    if assessment['kurtosis'] > 3:
        assessment['tail_behavior'] = 'heavy-tailed'
    elif assessment['kurtosis'] < -1:
        assessment['tail_behavior'] = 'light-tailed'
    else:
        assessment['tail_behavior'] = 'normal-tailed'
    
    # Outlier presence (quick Z-score check)
    z_scores = np.abs(stats.zscore(clean_data))
    assessment['n_outliers_z3'] = np.sum(z_scores > 3)
    assessment['outlier_pct_z3'] = assessment['n_outliers_z3'] / len(clean_data) * 100
    
    # Check for zeros and negatives
    assessment['n_zeros'] = np.sum(clean_data == 0)
    assessment['n_negative'] = np.sum(clean_data < 0)
    assessment['all_positive'] = assessment['n_negative'] == 0 and assessment['n_zeros'] == 0
    
    return assessment


def recommend_outlier_method(assessment: Dict) -> Tuple[str, str]:
    """
    Recommend outlier detection method based on data assessment.
    
    Returns:
    --------
    method : str
        Recommended method name
    reason : str
        Explanation for recommendation
    """
    if assessment['n_obs'] < 20:
        return 'iqr', "Small sample size (<20) - IQR most robust"
    
    # Many outliers takes priority over distribution shape
    if assessment['outlier_pct_z3'] > 10:
        return 'dbscan', "Many outliers detected - DBSCAN can identify outlier clusters"
    
    # IQR is the default: robust, distribution-free, and well-suited to
    # economic data which is rarely perfectly normal.
    if assessment['distribution'] in ['right-skewed', 'left-skewed']:
        return 'iqr', "Skewed distribution - IQR method robust and distribution-free"
    
    # Grubbs is an opt-in for the narrow case of approximately normal data
    # with very few outliers (<2%), where a formal statistical test adds value.
    if assessment['is_normal'] and assessment['outlier_pct_z3'] < 2:
        return 'grubbs', "Data approximately normal with very few outliers (<2%) - Grubbs test provides formal statistical test"
    
    if assessment['is_normal'] and assessment['outlier_pct_z3'] < 10:
        return 'z-score', "Data approximately normal - Z-score method fast and effective"
    
    # Default
    return 'iqr', "General robust method suitable for most economic data"


def recommend_normalization_method(assessment: Dict, purpose: str = 'comparison') -> Tuple[str, str]:
    """
    Recommend normalization method based on data assessment and purpose.
    
    Parameters:
    -----------
    assessment : dict
        Data assessment results
    purpose : str
        Purpose: 'comparison', 'ml', 'visualization'
        
    Returns:
    --------
    method : str
        Recommended method name
    reason : str
        Explanation for recommendation
    """
    # Check for heavy outliers
    has_heavy_outliers = assessment['outlier_pct_z3'] > 5
    
    # Check for skewness
    is_skewed = abs(assessment['skewness']) > 1.0
    
    # Machine learning purpose
    if purpose == 'ml':
        if has_heavy_outliers:
            return 'robust-scaling', "Heavy outliers present - Robust scaling protects ML models"
        elif assessment['all_positive'] and is_skewed:
            return 'log-transform-then-standard', "Right-skewed positive data - Log transform then standardize"
        else:
            return 'standard-scaling', "Standard z-score scaling for ML models"
    
    # Comparison purpose
    elif purpose == 'comparison':
        if is_skewed and assessment['all_positive']:
            return 'log-transform-then-standard', "Skewed data - Log transform for comparability"
        elif has_heavy_outliers:
            return 'robust-scaling', "Outliers present - Robust scaling for fair comparison"
        else:
            return 'standard-scaling', "Z-score standardization for comparing variables"
    
    # Visualization purpose
    elif purpose == 'visualization':
        if is_skewed and assessment['all_positive']:
            return 'log-transform', "Skewed data - Log transform for better visualization"
        else:
            return 'min-max-scaling', "Min-max scaling for bounded visualization range"
    
    # Default
    return 'standard-scaling', "General purpose standardization"


def recommend_smoothing_method(assessment: Dict, is_time_series: bool = True) -> Tuple[str, str]:
    """
    Recommend smoothing method based on data assessment.
    
    Parameters:
    -----------
    assessment : dict
        Data assessment results
    is_time_series : bool, default=True
        Whether data is time series
        
    Returns:
    --------
    method : str
        Recommended method name
    reason : str
        Explanation for recommendation
    """
    if not is_time_series:
        return 'loess', "Non-time-series data - LOESS provides flexible smoothing"
    
    # Check data length
    if assessment['n_obs'] < 50:
        return 'moving-average', "Short series - Simple moving average appropriate"
    
    # Check volatility (coefficient of variation)
    cv = assessment['std'] / abs(assessment['mean']) if assessment['mean'] != 0 else 0
    
    if cv > 0.3:  # High volatility
        return 'exponential-smoothing', "High volatility - Exponential smoothing adapts to recent changes"
    
    if assessment['n_obs'] >= 100:
        return 'henderson', "Sufficient length - Henderson filter preserves economic trends"
    
    # Default
    return 'moving-average', "Standard moving average for general smoothing"


def check_seasonality(data: pd.Series, max_period: int = 52) -> Dict:
    """
    Detect potential seasonal patterns in time series.
    
    Parameters:
    -----------
    data : pd.Series
        Time series with datetime index
    max_period : int, default=52
        Maximum period to test (52 weeks, 12 months, etc.)
        
    Returns:
    --------
    seasonality_info : dict
        Detected seasonal patterns and strengths
    """
    from scipy.signal import find_peaks
    from scipy.fft import fft, fftfreq
    
    # Autocorrelation function
    acf_result = pd.Series(data).autocorr
    
    # FFT to detect dominant frequencies
    n = len(data)
    values = data.values - np.mean(data.values)  # Remove mean
    fft_vals = np.abs(fft(values))[:n//2]
    freqs = fftfreq(n, d=1)[:n//2]
    
    # Find peaks in FFT (potential seasonal periods)
    # Use a higher threshold (10% of max) to reduce false positives
    max_fft = np.max(fft_vals) if len(fft_vals) > 0 else 0
    peaks, properties = find_peaks(fft_vals, height=max_fft * 0.10)
    
    # Convert frequencies to periods
    periods = [1/freqs[p] for p in peaks if freqs[p] > 0]
    periods = [int(round(p)) for p in periods if 2 <= p <= max_period]
    
    # Filter to only strong peaks: keep periods whose FFT power is at least
    # 25% of the strongest peak (reduces harmonic/leakage false positives)
    if len(peaks) > 0 and max_fft > 0:
        peak_powers = fft_vals[peaks]
        strong_mask = peak_powers >= max_fft * 0.25
        strong_peaks = peaks[strong_mask]
        periods = [1/freqs[p] for p in strong_peaks if freqs[p] > 0]
        periods = [int(round(p)) for p in periods if 2 <= p <= max_period]
    
    # Check common economic periods
    common_periods = {
        4: 'quarterly',
        12: 'monthly',
        52: 'weekly',
        7: 'daily-weekly'
    }
    
    detected = []
    for period, name in common_periods.items():
        # Only report a common period if it's an exact match (not within ±1)
        if period in periods:
            detected.append({'period': period, 'name': name})
    
    return {
        'has_seasonality': len(detected) > 0,
        'detected_periods': detected,
        'all_periods': periods[:5],  # Top 5
        'strength': max_fft / np.mean(fft_vals) if len(fft_vals) > 0 and np.mean(fft_vals) > 0 else 0
    }


def recommend_seasonal_method(seasonality_info: Dict, n_obs: int) -> Tuple[Optional[str], str]:
    """
    Recommend seasonal adjustment method.
    
    Returns:
    --------
    method : str or None
        Recommended method (None if no seasonality)
    reason : str
        Explanation
    """
    if not seasonality_info['has_seasonality']:
        return None, "No significant seasonality detected"
    
    detected = seasonality_info['detected_periods']
    
    # Check for monthly or quarterly
    has_monthly = any(d['period'] == 12 for d in detected)
    has_quarterly = any(d['period'] == 4 for d in detected)
    
    if (has_monthly or has_quarterly) and n_obs >= 36:
        return 'x13-arima-seats', f"Standard economic frequency detected ({detected[0]['name']}) - X-13ARIMA-SEATS provides official method"
    
    if n_obs >= 24:
        return 'stl', f"Seasonal pattern detected - STL decomposition flexible and robust"
    
    return 'classical', "Short series with seasonality - Classical decomposition appropriate"


def generate_report(assessment: Dict, 
                   outlier_rec: Tuple[str, str],
                   norm_rec: Tuple[str, str],
                   smooth_rec: Tuple[str, str],
                   seasonal_rec: Tuple[Optional[str], str]) -> str:
    """
    Generate comprehensive assessment and recommendation report.
    
    Returns:
    --------
    report : str
        Markdown-formatted report
    """
    report = f"""# Data Assessment Report: {assessment['name']}

## Data Overview

- **Total Observations**: {assessment['n_obs']}
- **Valid Observations**: {assessment['n_valid']}
- **Missing Values**: {assessment['n_missing']} ({assessment['missing_pct']:.1f}%)
- **Range**: [{assessment['min']:.2f}, {assessment['max']:.2f}]

## Descriptive Statistics

| Statistic | Value |
|-----------|-------|
| Mean | {assessment['mean']:.4f} |
| Median | {assessment['median']:.4f} |
| Std Dev | {assessment['std']:.4f} |
| Q1 | {assessment['q1']:.4f} |
| Q3 | {assessment['q3']:.4f} |
| IQR | {assessment['iqr']:.4f} |

## Distribution Characteristics

- **Shape**: {assessment['distribution']}
- **Skewness**: {assessment['skewness']:.3f}
- **Kurtosis**: {assessment['kurtosis']:.3f}
- **Tail Behavior**: {assessment['tail_behavior']}
- **Normality Test**: {assessment['normality_test']} (p={assessment['normality_p']:.4f})
- **Approximately Normal**: {'Yes' if assessment['is_normal'] else 'No'}

## Outlier Assessment

- **Outliers (|Z| > 3)**: {assessment['n_outliers_z3']} ({assessment['outlier_pct_z3']:.1f}%)
- **Zeros**: {assessment['n_zeros']}
- **Negative Values**: {assessment['n_negative']}
- **All Positive**: {'Yes' if assessment['all_positive'] else 'No'}

---

## Recommendations

### 1. Outlier Detection
**Recommended Method**: `{outlier_rec[0]}`  
**Reason**: {outlier_rec[1]}

### 2. Normalization
**Recommended Method**: `{norm_rec[0]}`  
**Reason**: {norm_rec[1]}

### 3. Smoothing
**Recommended Method**: `{smooth_rec[0]}`  
**Reason**: {smooth_rec[1]}

### 4. Seasonal Adjustment
**Recommended Method**: `{seasonal_rec[0] if seasonal_rec[0] else 'None'}`  
**Reason**: {seasonal_rec[1]}

---

## Next Steps

1. Address missing values if present (>5%)
2. Apply recommended outlier detection
3. Consider transformation if heavily skewed
4. Apply normalization as needed
5. Seasonally adjust if time series
6. Smooth to reveal trend if needed
"""
    
    return report


def quick_assess(data: np.ndarray) -> Dict[str, any]:
    """
    Quick assessment for interactive use.
    
    Returns:
    --------
    quick_summary : dict
        Essential characteristics only
    """
    clean_data = data[~np.isnan(data)]
    
    return {
        'n': len(clean_data),
        'mean': round(np.mean(clean_data), 2),
        'std': round(np.std(clean_data, ddof=1), 2),
        'skew': round(stats.skew(clean_data), 2),
        'normal': stats.shapiro(clean_data[:min(5000, len(clean_data))])[1] > 0.05,
        'outliers_pct': round(np.sum(np.abs(stats.zscore(clean_data)) > 3) / len(clean_data) * 100, 1)
    }


if __name__ == "__main__":
    # Example usage
    print("Data Assessment Module")
    print("=" * 50)
    
    # Test with simulated data
    np.random.seed(42)
    test_data = np.random.normal(100, 15, 200)
    test_data[50] = 200  # Add outlier
    
    # Run assessment
    assessment = assess_data(test_data, "Test Economic Series")
    outlier_rec = recommend_outlier_method(assessment)
    norm_rec = recommend_normalization_method(assessment, purpose='comparison')
    smooth_rec = recommend_smoothing_method(assessment)
    seasonal_rec = (None, "Not time series - no seasonal adjustment")
    
    # Generate report
    report = generate_report(assessment, outlier_rec, norm_rec, smooth_rec, seasonal_rec)
    print(report)