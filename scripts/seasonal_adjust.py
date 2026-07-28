#!/usr/bin/env python3
"""
Seasonal Adjustment Implementation Module

Provides seasonal adjustment methods for economic time series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


def stl_seasonal_adjustment(data: pd.Series, 
                            period: int = 12, 
                            seasonal: int = 7,
                            trend: Optional[int] = None,
                            robust: bool = True) -> Tuple[pd.Series, object]:
    """
    Apply STL decomposition for seasonal adjustment.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data with datetime index
    period : int, default=12
        Seasonal period (12=monthly, 4=quarterly, 52=weekly)
    seasonal : int, default=7
        Length of seasonal smoother (must be odd)
    trend : int, optional
        Length of trend smoother (auto if None)
    robust : bool, default=True
        Use robust fitting
        
    Returns:
    --------
    seasonally_adjusted : pd.Series
        Seasonally adjusted series
    result : STLResult
        Full decomposition result
    """
    from statsmodels.tsa.seasonal import STL
    
    # Ensure seasonal is odd
    if seasonal % 2 == 0:
        seasonal += 1
    
    # Run STL
    stl = STL(data, period=period, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()
    
    # Seasonally adjusted = Observed - Seasonal
    seasonally_adjusted = result.observed - result.seasonal
    
    return seasonally_adjusted, result


def classical_seasonal_adjustment(data: pd.Series,
                                 model: str = 'multiplicative',
                                 period: int = 12) -> Tuple[pd.Series, object]:
    """
    Apply classical seasonal decomposition.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    model : str, default='multiplicative'
        Model type: 'additive' or 'multiplicative'
    period : int, default=12
        Seasonal period
        
    Returns:
    --------
    seasonally_adjusted : pd.Series
        Seasonally adjusted series
    result : DecomposeResult
        Full decomposition result
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    result = seasonal_decompose(data, model=model, period=period, extrapolate_trend='freq')
    
    # Calculate seasonally adjusted
    if model == 'multiplicative':
        seasonally_adjusted = result.observed / result.seasonal
    else:
        seasonally_adjusted = result.observed - result.seasonal
    
    return seassonally_adjusted, result


def moving_average_seasonal_adjustment(data: pd.Series, 
                                      period: int = 12,
                                      model: str = 'multiplicative') -> Tuple[pd.Series, pd.Series]:
    """
    Apply moving average seasonal adjustment.
    
    Parameters:
    ------------
    data : pd.Series
        Time series with datetime index
    period : int, default=12
        Seasonal period
    model : str, default='multiplicative'
        Model type: 'additive' or 'multiplicative'
        
    Returns:
    ---------
    seasonally_adjusted : pd.Series
        Seasonally adjusted series
    seasonal_component : pd.Series
        Extracted seasonal component
    """
    # Calculate centered moving average (trend estimate)
    if period % 2 == 0:
        # Even period: two-step centered MA
        ma1 = data.rolling(window=period, center=True).mean()
        trend = ma1.rolling(window=2, center=True).mean()
    else:
        # Odd period: single centered MA
        trend = data.rolling(window=period, center=True).mean()
    
    # Detrend
    if model == 'multiplicative':
        detrended = data / trend
    else:
        detrended = data - trend
    
    # Calculate seasonal indices by period
    if isinstance(data.index, pd.DatetimeIndex):
        if period == 12:
            period_key = detrended.index.month
        elif period == 4:
            period_key = detrended.index.quarter
        else:
            # Generic: use position mod period
            period_key = np.arange(len(detrended)) % period
    else:
        period_key = np.arange(len(detrended)) % period
    
    seasonal_avg = detrended.groupby(period_key).mean()
    
    # Normalize seasonal component
    if model == 'multiplicative':
        seasonal_avg = seasonal_avg / seasonal_avg.mean()
    else:
        seasonal_avg = seasonal_avg - seasonal_avg.mean()
    
    # Create seasonal component series
    seasonal_component = pd.Series(index=data.index, dtype=float)
    for idx, val in enumerate(data.index):
        if isinstance(data.index, pd.DatetimeIndex):
            if period == 12:
                key = val.month
            elif period == 4:
                key = val.quarter
            else:
                key = idx % period
        else:
            key = idx % period
        seasonal_component.iloc[idx] = seasonal_avg[key]
    
    # Seasonally adjust
    if model == 'multiplicative':
        seasonally_adjusted = data / seasonal_component
    else:
        seasonally_adjusted = data - seasonal_component
    
    return seasonally_adjusted, seasonal_component


def x13_seasonal_adjustment(data: pd.Series,
                           freq: int = 12,
                           x12path: Optional[str] = None) -> Tuple[pd.Series, object]:
    """
    Apply X-13ARIMA-SEATS seasonal adjustment.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    freq : int, default=12
        Frequency (12=monthly, 4=quarterly)
    x12path : str, optional
        Path to X-13 executable (auto-detect if None)
        
    Returns:
    --------
    seasonally_adjusted : pd.Series
        Seasonally adjusted series
    results : X13ArimaAnalysisResult
        Full X-13 results object
    """
    import statsmodels.api as sm
    
    # Check if X-13 available
    if x12path is None:
        import shutil
        x12path = shutil.which('x13as') or shutil.which('x13as.exe')
        
        if x12path is None:
            raise EnvironmentError(
                "X-13ARIMA-SEATS not found. Install from: "
                "https://www.census.gov/data/software/x13as.html "
                "or use STL decomposition as alternative."
            )
    
    # Run X-13
    results = sm.tsa.x13_arima_analysis(
        data,
        x12path=x12path,
        freq=freq,
        trading=True,        # Trading day adjustment
        outlier=True,        # Automatic outlier detection
        automdl=True,        # Automatic ARIMA model
        forecast_periods=0   # No forecasting
    )
    
    return results.seasadj, results


def seasonal_adjust_auto(data: pd.Series,
                        period: Optional[int] = None) -> Tuple[pd.Series, str, Dict]:
    """
    Automatically select and apply seasonal adjustment method.
    
    Parameters:
    -----------
    data : pd.Series
        Time series with datetime or period index
    period : int, optional
        Seasonal period (auto-detect if None)
        
    Returns:
    --------
    seasonally_adjusted : pd.Series
        Seasonally adjusted series
    method_used : str
        Name of method applied
    details : dict
        Decomposition details
    """
    n_obs = len(data)
    
    # Auto-detect period if not provided
    if period is None:
        if isinstance(data.index, pd.DatetimeIndex):
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq:
                if 'M' in inferred_freq:  # Monthly
                    period = 12
                elif 'Q' in inferred_freq:  # Quarterly
                    period = 4
                elif 'W' in inferred_freq:  # Weekly
                    period = 52
                elif 'D' in inferred_freq:  # Daily
                    period = 7
            else:
                period = 12  # Default to monthly
        else:
            period = 12  # Default
    
    details = {'period': period, 'n_obs': n_obs}
    
    # Select method based on characteristics
    if period in [4, 12] and n_obs >= 36:
        # Try X-13 first
        try:
            sa_series, result = x13_seasonal_adjustment(data, freq=period)
            method = 'x13-arima-seats'
            details['arima_model'] = str(result.mdl) if hasattr(result, 'mdl') else 'N/A'
            details['outliers'] = result.outlier if hasattr(result, 'outlier') else []
            return sa_series, method, details
        except (EnvironmentError, Exception) as e:
            # Fall back to STL
            pass
    
    # Try STL as reliable fallback (requires statsmodels)
    if n_obs >= 24:
        try:
            sa_series, result = stl_seasonal_adjustment(data, period=period, robust=True)
            method = 'stl-decomposition'
            details['trend_strength'] = 1 - (np.var(result.resid) / np.var(result.trend + result.resid))
            details['seasonal_strength'] = 1 - (np.var(result.resid) / np.var(result.seasonal + result.resid))
            return sa_series, method, details
        except (ImportError, Exception):
            # statsmodels not available, fall through to moving average
            pass
    
    # Try classical decomposition (requires statsmodels)
    try:
        sa_series, result = classical_seasonal_adjustment(data, period=period)
        method = 'classical-decomposition'
        details['model'] = 'multiplicative'
        return sa_series, method, details
    except (ImportError, Exception):
        # statsmodels not available, fall through to moving average
        pass
    
    # Final fallback: moving-average seasonal adjustment (pure pandas, no statsmodels)
    sa_series, seasonal_component = moving_average_seasonal_adjustment(data, period=period)
    method = 'moving-average'
    details['model'] = 'multiplicative'
    details['seasonal_component'] = seasonal_component
    
    return sa_series, method, details


if __name__ == "__main__":
    print("Seasonal Adjustment Module - Test Suite")
    print("=" * 50)
    
    # Generate test series with seasonality
    np.random.seed(42)
    n = 60  # 5 years monthly
    trend = np.linspace(100, 120, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 2, n)
    test_series = pd.Series(
        trend + seasonal + noise,
        index=pd.date_range('2021-01-01', periods=n, freq='MS')
    )
    
    print(f"\nTest Series: {n} observations")
    print(f"  Mean: {test_series.mean():.2f}")
    print(f"  Std: {test_series.std():.2f}")
    
    # Test STL
    print("\n1. STL Decomposition:")
    sa_stl, result_stl = stl_seasonal_adjustment(test_series, period=12)
    print(f"   Seasonal component variance: {result_stl.seasonal.var():.2f}")
    print(f"   Residual variance: {result_stl.resid.var():.2f}")
    
    # Test Classical
    print("\n2. Classical Decomposition:")
    sa_classical, result_classical = classical_seasonal_adjustment(test_series, period=12)
    print(f"   Seasonal component range: [{result_classical.seasonal.min():.2f}, {result_classical.seasonal.max():.2f}]")
    
    # Test Auto
    print("\n3. Automatic Selection:")
    sa_auto, method, details = seasonal_adjust_auto(test_series)
    print(f"   Method selected: {method}")
    print(f"   Period: {details['period']}")