# Seasonal Adjustment Methods - Technical Reference

## Overview

This reference covers seasonal adjustment techniques for removing recurring calendar-related patterns from economic time series.

## 1. X-13ARIMA-SEATS

### Theory
Official seasonal adjustment program developed by U.S. Census Bureau. Combines ARIMA modeling with SEATS (Signal Extraction in ARIMA Time Series) decomposition.

### Components
```
Original = Trend-Cycle × Seasonal × Irregular  (multiplicative)
or
Original = Trend-Cycle + Seasonal + Irregular  (additive)
```

### Key Features
- Automatic ARIMA model selection
- Trading day and holiday adjustments
- Outlier detection (AO, LS, TC, RP)
- Sliding spans diagnostics
- Revision history analysis

### Implementation
```python
import statsmodels.api as sm

def x13_seasonal_adjustment(data, freq=12, x12path=None):
    """
    Apply X-13ARIMA-SEATS seasonal adjustment.
    
    Parameters:
    -----------
    data : array-like or pd.Series
        Time series data
<br/>    freq : int, default=12
        Frequency (12=monthly, 4=quarterly)
    x12path : str, optional
        Path to X-13 executable
        
    Returns:
    --------
    results : X13ArimaAnalysisResult
        Contains:
        - seasadj: seasonally adjusted series
        - trend: trend-cycle component
        - seasonal: seasonal component
        - irregular: irregular component
        - diagnostics: quality metrics
    """
    results = sm.tsa.x13_arima_analysis(
        data,
        x12path=x12path,
        freq=freq,
        trading=True,       # Trading day adjustment
        outlier=True,       # Automatic outlier detection
        automdl=True        # Automatic ARIMA model selection
    )
    
    return results
```

### Supported Frequencies
- Monthly (freq=12) ✓
- Quarterly (freq=4) ✓
- Daily, Weekly, Annual ✗ (not supported)

### Adjustment Methods
1. **X-11**: Filter-based method (traditional)
2. **SEATS**: Signal extraction via ARIMA decomposition (modern)

### Automatic Features

#### Outlier Types
- **AO** (Additive Outlier): Single spike
- **LS** (Level Shift): Permanent change
- **TC** (Temporary Change): Gradual effect
- **RP** (Ramp): Linear transition

#### Trading Day Adjustment
Accounts for varying number of weekdays per month.

#### Holiday Effects
- Easter (movable)
- Labor Day
- Thanksgiving
- Custom holidays

### Diagnostics

#### Quality Measures
- **M statistics**: Overall quality (M1-M11)
- **Q statistic**: Combined quality measure
- **F statistics**: Seasonal stability tests

#### Acceptable Ranges
- M7 < 1: Good seasonal adjustment
- Q < 1: Acceptable quality
- F-tests: p > 0.01 for stability

### Use Cases
- Official government statistics
- Central bank data
- Academic research requiring standard methods
- Publishing seasonally adjusted figures

### Advantages
- Industry standard
- Comprehensive diagnostics
- Automatic model selection
- Well-documented and tested

### Limitations
- Monthly/quarterly only
- Requires external binary
- Setup complexity
- Steeper learning curve

---

## 2. STL Decomposition

### Theory
Seasonal-Trend decomposition using Loess (Locally Estimated Scatterplot Smoothing). Robust, flexible method supporting any seasonal period.

### Algorithm Steps
```
1. Inner loop (iterative):
   a. Detrending
   b. Cycle-subseries smoothing
   c. Low-pass filtering of smoothed cycle-subseries
   d. Detrending of smoothed cycle-subseries
   e. Deseasonalizing
   f. Trend smoothing
   
2. Outer loop (robustness):
   Repeat inner loop with robustness weights
```

### Implementation
```python
from statsmodels.tsa.seasonal import STL

def stl_seasonal_adjustment(data, period, seasonal=7, trend=None, robust=True):
    """
    Apply STL decomposition for seasonal adjustment.
    
    Parameters:
    -----------
    data : array-like or pd.Series
        Time series data
    period : int
        Seasonal period (12=monthly, 4=quarterly, 7=weekly)
    seasonal : int, default=7
        Length of seasonal smoother (must be odd)
    trend : int, optional
        Length of trend smoother (if None, calculated automatically)
    robust : bool, default=True
        Use robust fitting (resistant to outliers)
        
    Returns:
    --------
    result : STLResult
        Contains:
        - seasonal: seasonal component
        - trend: trend component
        - resid: residual (irregular) component
        - observed: original series
    """
    stl = STL(data, period=period, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()
    
    # Seasonally adjusted = Observed - Seasonal
    seasonally_adjusted = result.observed - result.seasonal
    
    return result, seasonally_adjusted
```

### Parameters

#### seasonal (must be odd)
- Small values: Follow seasonal pattern closely
- Large values: Smooth seasonal pattern
- **Rule of thumb**: 7 for monthly, 11 for quarterly
- Must be odd and ≥ 3

#### trend (optional, calculated if None)
- Controls trend smoothness
- **Default**: Calculated as smallest odd integer ≥ (1.5 × period / (1 - 1.5/seasonal)) + 1
- Larger = smoother trend

#### robust (boolean)
- True: Down-weights outliers in fitting
- False: Standard fitting
- **Recommendation**: Use True for most economic data

### Seasonal Periods

Common economic periods:
```python
monthly_data:    period=12
quarterly_data:  period=4
weekly_data:     period=52
daily_data:      period=7 (day-of-week) or 365 (annual)
hourly_data:     period=24 (daily) or 168 (weekly)
```

### Use Cases
- Any seasonal frequency (not just monthly/quarterly)
- Multiple seasonal patterns (e.g., daily + weekly)
- Robust to outliers
- Quick exploratory analysis
- When X-13 not appropriate

### Advantages
- Works with any period
- Robust to outliers
- Fast computation
- No external dependencies
- Intuitive parameters

### Limitations
- Less comprehensive diagnostics than X-13
- Not official standard
- May over-smooth with wrong parameters
- No automatic outlier adjustment

---

## 3. Classical Seasonal Decomposition

### Theory
Traditional method using moving averages. Simple but effective for stable seasonal patterns.

### Additive Model
```
Y_t = T_t + S_t + R_t

where:
  Y_t = Observed value
  T_t = Trend component
  S_t = Seasonal component
  R_t = Random (irregular) component
```

### Multiplicative Model
```
Y_t = T_t × S_t × R_t
```

### Algorithm

#### Step 1: Estimate Trend
Apply centered moving average of length equal to seasonal period.

For monthly data (period=12):
```
T_t = (1/12) × (0.5×Y_{t-6} + Y_{t-5} + ... + Y_{t+5} + 0.5×Y_{t+6})
```

#### Step 2: Remove Trend
```
Additive:       D_t = Y_t - T_t
Multiplicative: D_t = Y_t / T_t
```

#### Step 3: Estimate Seasonal Component
Average detrended values by season:
```
S_m = average of all D_t where t is in month m
```

Normalize so seasonal components sum to 0 (additive) or average to 1 (multiplicative).

#### Step 4: Calculate Irregular Component
```
Additive:       R_t = Y_t - T_t - S_t
Multiplicative: R_t = Y_t / (T_t × S_t)
```

### Implementation
```python
From statsmodels.tsa.seasonal import seasonal_decompose

def classical_seasonal_adjustment(data, model='multiplicative', period=12):
    """
    Apply classical seasonal decomposition.
    
    Parameters:
    -----------
    data : array-like or pd.Series
        Time series data
    model : {'additive', 'multiplicative'}
        Decomposition model
    period : int
        Seasonal period
        
    Returns:
    --------
    result : DecomposeResult
        Contains:
        - trend: trend component
        - seasonal: seasonal component
        - resid: residual component
        - observed: original data
    """
    result = seasonal_decompose(data, model=model, period=period)
    
    # Seasonally adjusted
    if model == 'multiplicative':
        seasonally_adjusted = result.observed / result.seasonal
    else:
        seasonally_adjusted = result.observed - result.seasonal
    
    return result, seasonally_adjusted
```

### Choosing Model Type

**Use Multiplicative when:**
- Seasonal variation increases with level
- Percentage changes matter
- Most economic growth series

**Use Additive when:**
- Seasonal variation constant over time
- Absolute changes matter
- Linear relationships

### Use Cases
- Quick exploratory analysis
- Educational purposes
- Stable seasonal patterns
- When simplicity preferred

### Advantages
- Very simple conceptually
- Fast computation
- No parameters to tune
- Easy to implement

### Limitations
- Not robust to outliers
- Fixed seasonal pattern
- Loses observations at ends
- No outlier detection
- Less sophisticated than X-13 or STL

---

## 4. Moving Average Seasonal Adjustment

### Theory
Rolling window average to estimate and remove seasonal patterns.

### Simple Moving Average
```
MA_t = (1/k) × Σ_{i=-m}^{m} Y_{t+i}

where k = 2m + 1 (window width)
```

### Centered Moving Average (for even periods)
```
For period=12:
MA_t = (1/24) × (Y_{t-6} + 2×Y_{t-5} + ... + 2×Y_{t+5} + Y_{t+6})
```

### Implementation
```python
def moving_average_seasonal_adjustment(data, period=12):
    """
    Apply moving average seasonal adjustment.
    
    Parameters:
    -----------
    data : pd.Series
        Time series with datetime index
    period : int
        Seasonal period
        
    Returns:
    --------
    seasonally_adjusted : pd.Series
        Seasonally adjusted series
    seasonal_component : pd.Series
        Extracted seasonal component
    """
    # Calculate centered moving average (trend)
    if period % 2 == 0:
        # Even period: two-step centered MA
        ma1 = data.rolling(window=period, center=True).mean()
        trend = ma1.rolling(window=2, center=True).mean()
    else:
        # Odd period: single centered MA
        trend = data.rolling(window=period, center=True).mean()
    
    # Detrend
    detrended = data / trend  # or data - trend for additive
    
    # Calculate seasonal indices
    seasonal_avg = detrended.groupby(detrended.index.month).mean()
    
    # Normalize (multiplicative: average to 1)
    seasonal_avg = seasonal_avg / seasonal_avg.mean()
    
    # Create seasonal component series
    seasonal_component = data.index.map(lambda x: seasonal_avg[x.month])
    
    # Seasonally adjust
    seasonally_adjusted = data / seasonal_component
    
    return seasonally_adjusted, seasonal_component
```

### Use Cases
- Quick seasonal removal
- Simple trend extraction
- When advanced methods unavailable
- Educational demonstrations

### Advantages
- Extremely simple
- No external dependencies
- Fast
- Easy to understand

### Limitations
- Crude approximation
- Loses data at ends
- Assumes stable seasonality
- No diagnostics

---

## Method Selection Guide

### Decision Tree

```
START
  │
  ├─ Publishing official statistics?
  │   └─ YES → X-13ARIMA-SEATS
  │
  ├─ Monthly or Quarterly data?
  │   ├─ YES → X-13ARIMA-SEATS or STL
  │   └─ NO (other frequency) → STL
  │
  ├─ Need robustness to outliers?
  │   └─ YES → STL (robust=True)
  │
  ├─ Quick exploratory analysis?
  │   └─ YES → Classical Decomposition or STL
  │
  └─ Very simple method needed?
      └─ YES → Moving Average or Classical
```

### Quick Comparison

| Method | Frequencies | Outlier Robust | Diagnostics | Complexity | Speed |
|--------|------------|----------------|-------------|------------|-------|
| X-13ARIMA-SEATS | 12, 4 | Yes (auto) | Comprehensive | High | Medium |
| STL | Any | Yes (optional) | Basic | Low | Fast |
| Classical | Any | No | None | Very Low | Fast |
| Moving Avg | Any | No | None | Very Low | Very Fast |

---

## Best Practices

### 1. Minimum Data Requirements
- **Monthly**: At least 3 years (36 obs)
- **Quarterly**: At least 5 years (20 obs)
- **Weekly**: At least 2 years (104 obs)

### 2. Data Preparation
- Check for missing values (interpolate if needed)
- Transform if necessary (log for multiplicative)
- Remove known outliers before adjustment

### 3. Model Selection
- Start with STL for exploration
- Use X-13 for official/publication purposes
- Check multiplicative vs additive assumption

### 4. Validation
- Plot original vs adjusted
- Check residuals for patterns
- Verify seasonal component stable
- Run diagnostics (if available)

### 5. Reporting
- Always report adjustment method
- Include diagnostic statistics
- Show before/after comparisons
- Document parameter choices

---

## Common Issues & Solutions

### Issue 1: Unstable Seasonal Pattern
**Problem**: Seasonal pattern changes over time  
**Solution**: Use STL (more flexible) or revisable X-13

### Issue 2: Extreme Outliers
**Problem**: Large spikes affecting adjustment  
**Solution**: X-13 with outlier detection or STL with robust=True

### Issue 3: Multiple Seasonal Patterns
**Problem**: Daily + weekly seasonality  
**Solution**: Nested STL or MSTL (multiple STL)

### Issue 4: Irregular Frequency
**Problem**: Daily data with weekends/holidays  
**Solution**: STL with appropriate period after removing missing days

---

## Validation Metrics

### 1. Residual Analysis
Check if residuals (irregular component) are:
- White noise (no autocorrelation)
- Normally distributed
- Constant variance

```python
From statsmodels.stats.diagnostic import acorr_ljungbox

def validate_residuals(residuals):
    """Check if residuals are white noise."""
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=20, return_df=True)
    
    # Should have p-values > 0.05 (no autocorrelation)
    return lb_test
```

### 2. Seasonal Stability
Check if seasonal pattern consistent year-over-year.

### 3. Revision Analysis
For X-13: Check how estimates change with new data.

---

## References

1. U.S. Census Bureau (2025). "X-13ARIMA-SEATS Reference Manual". U.S. Census Bureau.

2. Cleveland, R.B., Cleveland, W.S., McRae, J.E., Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess". Journal of Official Statistics. 6(1): 3–73.

3. Findley, D.F., Monsell, B.C., Bell, W.R., Otto, M.C., Chen, B.C. (1998). "New Capabilities and Methods of the X-12-ARIMA Seasonal-Adjustment Program". Journal of Business & Economic Statistics. 16(2): 127–152.

4. Hyndman, R.J., Athanasopoulos, G. (2021). "Forecasting: Principles and Practice" (3rd ed.). OTexts.