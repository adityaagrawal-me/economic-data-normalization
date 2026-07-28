# Smoothing Techniques - Technical Reference

## Overview

This reference covers smoothing and filtering methods for extracting trends and reducing noise in economic time series.

## 1. Henderson Filters

### Theory
Symmetric moving average filters designed to reproduce polynomial trends without distortion while minimizing variance of residuals.

### Key Property
Henderson filters preserve polynomials up to degree 3 (cubic trends) exactly.

### Common Filter Lengths
- 5-term: Very short-term smoothing
- 9-term: Short-term trends
- 13-term: Medium-term trends
- 23-term: Long-term trends

### Weight Formula
```
For n-term Henderson filter (n odd):
w_j = [315(n²-1 - j²)²(n²-9 - j²)(3n⁴ - 16n² + 19)] / 
      [8(n-2)(n²-4)(n²-1)(n²-9)(n²-16)]

where j = 0, ±1, ±2, ..., ±(n-1)/2
```

### Implementation
```python
import numpy as np

def henderson_filter(data, length=13):
    """
    Apply Henderson filter for trend extraction.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    length : int, default=13
        Filter length (must be odd: 5, 9, 13, 23)
        
    Returns:
    --------
    smoothed : ndarray
        Henderson-filtered series
    weights : ndarray
        Filter weights used
    """
    if length not in [5, 9, 13, 23]:
        raise ValueError("Length must be 5, 9, 13, or 23")
    
    # Pre-computed Henderson weights
    weights = get_henderson_weights(length)
    
    # Apply symmetric filter
    half_length = length // 2
    smoothed = np.full_like(data, np.nan, dtype=float)
    
    for i in range(half_length, len(data) - half_length):
        smoothed[i] = np.sum(weights * data[i-half_length:i+half_length+1])
    
    # Handle endpoints with asymmetric filters
    smoothed = handle_endpoints(data, smoothed, weights, half_length)
    
    return smoothed, weights

def get_henderson_weights(length):
    """Get pre-computed Henderson filter weights."""
    # 13-term Henderson (most common)
    if length == 13:
        return np.array([
            -0.019, -0.028, 0.000, 0.066, 0.147,
            0.214, 0.240, 0.214, 0.147,
            0.066, 0.000, -0.028, -0.019
        ])
    
    # 5-term Henderson
    elif length == 5:
        return np.array([
            -0.073, 0.294, 0.558, 0.294, -0.073
        ])
    
    # 9-term Henderson
    elif length == 9:
        return np.array([
            -0.041, -0.010, 0.119, 0.267,
            0.330, 0.267, 0.119, -0.010, -0.041
        ])
    
    # 23-term Henderson
    elif length == 23:
        return np.array([
            -0.007, -0.014, -0.017, -0.016, -0.011,
            -0.004, 0.007, 0.018, 0.031, 0.045,
            0.058, 0.068, 0.075, 0.068, 0.058,
            0.045, 0.031, 0.018, 0.007, -0.004,
            -0.011, -0.016, -0.017, -0.014, -0.007
        ])
```

### Endpoint Treatment
Use asymmetric filters at series endpoints:
- Shorter filter lengths near boundaries
- Weights adjusted to preserve polynomial property
- May introduce small endpoint bias

### Use Cases
- Official economic trend extraction (BLS, Census Bureau)
- Business cycle analysis
- Removing high-frequency noise while preserving cycles
- When polynomial trends expected

### Advantages
- Preserves polynomial trends exactly
- Well-studied properties
- Standard in official statistics
- Minimal phase shift

### Limitations
- Loses observations at endpoints
- Fixed weights (not adaptive)
- Requires sufficient data length
- May over-smooth sharp changes

---

## 2. Exponential Smoothing

### Theory
Weighted average giving exponentially decreasing weights to past observations.

### Simple Exponential Smoothing
```
S_t = α × Y_t + (1 - α) × S_{t-1}

where:
  S_t = smoothed value at time t
  Y_t = actual value at time t
  α = smoothing parameter (0 < α < 1)
```

### Equivalent Form
```
S_t = S_{t-1} + α × e_{t-1}

where e_{t-1} = Y_{t-1} - S_{t-1} (forecast error)
```

### Weight Pattern
Observation k periods ago receives weight: α(1-α)^k

### Implementation
```python
def exponential_smoothing(data, alpha=0.3):
    """
    Apply simple exponential smoothing.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    alpha : float, default=0.3
        Smoothing parameter (0 < alpha < 1)
        Higher = more responsive to recent data
        
    Returns:
    --------
    smoothed : ndarray
        Exponentially smoothed series
    """
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    
    return smoothed
```

### Choosing Alpha

| Alpha | Behavior | Use when |
|-------|----------|----------|
| 0.1 - 0.2 | Heavy smoothing, slow response | Noisy data, stable trends |
| 0.3 - 0.5 | Moderate smoothing | Typical economic data |
| 0.6 - 0.9 | Light smoothing, fast response | Recent data critical, volatile |

### Automatic Alpha Selection
Minimize one-step-ahead forecast error:
```python
From scipy.optimize import minimize_scalar

def optimal_alpha(data):
    """Find optimal alpha by minimizing MSE."""
    def mse(alpha):
        smoothed = exponential_smoothing(data, alpha)
        errors = data[1:] - smoothed[:-1]
        return np.mean(errors**2)
    
    result = minimize_scalar(mse, bounds=(0.01, 0.99), method='bounded')
    return result.x
```

### Double Exponential Smoothing (Holt's Method)
Accounts for trend:
```
Level:  L_t = α × Y_t + (1 - α) × (L_{t-1} + T_{t-1})
Trend:  T_t = β × (L_t - L_{t-1}) + (1 - β) × T_{t-1}
```

### Triple Exponential Smoothing (Holt-Winters)
Accounts for seasonality:
```
Level:    L_t = α × (Y_t / S_{t-s}) + (1 - α) × (L_{t-1} + T_{t-1})
Trend:    T_t = β × (L_t - L_{t-1}) + (1 - β) × T_{t-1}
Seasonal: S_t = γ × (Y_t / L_t) + (1 - γ) × S_{t-s}
```

### Use Cases
- Forecasting
- Real-time data smoothing
- When recent data more important
- Inventory management, sales forecasting

### Advantages
- Simple and intuitive
- Computationally efficient
- Works well for forecasting
- Naturally handles new data

### Limitations
- Lags behind actual data
- Fixed decay rate
- Sensitive to initial value- May over-smooth sharp changes

---

## 3. Moving Averages

### Simple Moving Average (SMA)
Equal weights to all observations in window.

```
SMA_t = (1/k) × Σ_{i=-n}^{k-1} Y_{t-i}

where k = window size
```

### Implementation
```python
def simple_moving_average(data, window=12):
    """
    Calculate simple moving average.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    window : int, default=12
        Window size (number of periods)
        
    Returns:
    --------
    ma : ndarray
        Moving average series
    """
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Pad with NaNs to match original length
    pad_length = len(data) - len(ma)
    ma = np.concatenate([np.full(pad_length, np.nan), ma])
    
    return ma
```

### Centered Moving Average
Centers window around current observation:
```python
def centered_moving_average(data, window=12):
    """Calculate centered moving average."""
    if window % 2 == 0:
        # Even window: two-step averaging
        ma1 = simple_moving_average(data, window)
        ma2 = simple_moving_average(ma1, 2)
        return ma2
    else:
        # Odd window: direct centering
        half = window // 2
        ma = np.full_like(data, np.nan, dtype=float)
        for i in range(half, len(data) - half):
            ma[i] = np.mean(data[i-half:i+half+1])
        return ma
```

### Weighted Moving Average (WMA)
Linear decreasing weights:
```
WMA_t = Σ_{i=0}^{k-1} [(k-i)/Σ_weights] × Y_{t-i}

where weights = k + (k-1) + ... + 1 = k(k+1)/2
```

### Window Size Selection

For monthly data:
- 3-month: Short-term smoothing
- 6-month: Semi-annual trends
- 12-month: Annual smoothing (removes seasonality)

For quarterly data:
- 4-quarter: Annual smoothing

Rule of thumb: Window = Seasonal period to remove seasonality

### Use Cases
- Quick trend visualization
- Removing noise
- Technical analysis (finance)
- Simple forecasting

### Advantages
- Very simple
- Intuitive interpretation
- Fast computation
- No parameters (except window)

### Limitations
- Lags behind data
- Equal weights may not be optimal
- Loses observations at endpoints
- Can create artificial patterns

---

## 4. LOESS (Locally Weighted Scatterplot Smoothing)

### Theory
Non-parametric regression using locally weighted polynomial fits.

### Algorithm
For each point x_i:
1. Select neighborhood (nearest k points or within bandwidth)
2. Weight points by distance (tri-cube weight function)
3. Fit weighted polynomial (usually degree 1 or 2)
4. Smoothed value = fitted value at x_i

### Tri-Cube Weight Function
```
w(d) = (1 - (d/d_max)³)³  for d < d_max
w(d) = 0                   for d ≥ d_max

where:
  d = distance from x_i
  d_max = distance to k-th nearest neighbor
```

### Implementation
```python
from statsmodels.nonparametric.smoothers_lowest import lowess

def loess_smoothing(data, frac=0.1, it=3, delta=0.0):
    """
    Apply LOESS smoothing.
    
    Parameters:
    -----------
    data : array-like or pd.Series
        Input time series
    frac : float, default=0.1
        Fraction of data used for smoothing (bandwidth)
        Between 0 and 1
    it : int, default=3
        Number of robustifying iterations
    delta : float, default=0.0
        Distance within which to use linear interpolation
        
    Returns:
    --------
    smoothed : ndarray
        LOESS-smoothed series
    """
    # Create x values (time indices)
    x = np.arange(len(data))
    
    # Apply LOWESS
    smoothed = lowess(data, x, frac=frac, it=it, delta=delta, return_sorted=False)
    
    return smoothed
```

### Parameters

#### frac (Bandwidth)
- Small (0.05-0.1): Follows data closely, less smooth
- Medium (0.2-0.3): Balanced smoothing
- Large (0.4-0.5): Heavy smoothing
- **Rule of thumb**: frac = k/n where k = neighborhood size, n = data length

#### it (Robustifying Iterations)
- 0: No robustification (sensitive to outliers)
- 2-3: Standard (recommended)
- 5+: Very robust (may over-smooth)

### Robust LOESS
Down-weights outliers using bisquare function:
```
w_robust = w × [1 - (|residual| / 6MAD)²]²

where MAD = median absolute deviation
```

### Use Cases
- Exploratory data analysis
- Flexible trend extraction
- When functional form unknown
- Scatterplot smoothing
- Non-linear relationships

### Advantages
- Very flexible
- No functional form assumed
- Robust version handles outliers
- Adaptive to local data structure

### Limitations
- Computationally intensive
- Difficult to select frac parameter
- No closed-form equation
- Can over-fit with small frac
- Not suitable for forecasting

---

## 5. Savitzky-Golay Filter

### Theory
Fits polynomial to sliding window using least squares. Smooths while preserving peak shapes.

### Implementation
```python
from scipy.signal import savgol_filter

def savitzky_golay_smoothing(data, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay smoothing filter.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    window_length : int, default=11
        Length of filter window (must be odd, ≥ polyorder+2)
    polyorder : int, default=3
        Order of polynomial (1=linear, 2=quadratic, 3=cubic)
        
    Returns:
    --------
    smoothed : ndarray
        Savitzky-Golay filtered series
    """
    smoothed = savgol_filter(data, window_length, polyorder, mode='nearest')
    return smoothed
```

### Parameter Selection

#### window_length
- Larger: More smoothing
- Smaller: Preserves details
- Must be odd and > polyorder
- **Typical**: 5-25 for economic data

#### polyorder
- 1: Linear (simple)
- 2: Quadratic (recommended for most)
- 3: Cubic (maximum detail preservivation)
- Higher orders risk over-fitting

### Use Cases
- Spectroscopy, chemical data
- Signal processing
- Preserving peak shapes
- When derivative needed

### Advantages
- Preserves features (peaks, valleyes)
- Can compute derivatives
- Handles evenly spaced data well
- Fast computation

### Limitations
- Requires evenly spaced data
- Sensitive to outliers
- Not robust
- Edge effects

---

## Method Selection Guide

### Decision Tree

```
START
  │
  ├─ Need to preserve polynomial trends?
  │   └─ YES → Henderson Filter
  │
  ├─ Need for forecasting?
  │   └─ YES → Exponential Smoothing
  │
  ├─ Non-parametric, flexible smoothing?
  │   └─ YES → LOESS
  │
  ├─ Simple, quick visualization?
  │   └─ YES → Moving Average
  │
  ├─ Preserve peak shapes?
  │   └─ YES → Savitzky-Golay
  │
  └─ Official statistics?
      └─ YES → Henderson Filter
```

### Quick Comparison

| Method | Parametric | Preserves Trends | Robust | Forecasting | Complexity | Speed |
|-------|------------|------------------|--------|-------------|------------|-------|
| Henderson | Yes | Polynomial | No | No | Low | Fast |
| Exponential | Yes | Linear | No | Yes | Low | Very Fast |
| Moving Avg | Yes | None | No | No | Very Low | Very Fast |
| LOESS | No | Any | Yes | No | High | Slow |
| Savitzky-Golay | Yes | Polynomial | No | No | Medium | Fast |

---

## Choosing Smoothing Strength

### Under-Smoothing
**Signs:**
- Still see noise in output
- No clear trend visible
- High-frequency fluctuations remain

**Fix:** Increase smoothing (larger window, lower alpha, higher frac)

### Over-Smoothing
**Signs:**
- Important features disappeared
- Trend too smooth (unrealistic)
- Lags far behind actual data

**Fix:** Decrease smoothing (smaller window, higher alpha, lower frac)

### Optimal Smoothing
**Goldilocks principle:**
- Trend visible and clear
- Important features preserved
- Noise substantially reduced
- Appropriate for analysis purpose

---

## Validation

### Visual Checks
- Plot original vs smoothed
- Check if trend makes economic sense
- Verify features preserved/removed appropriately

### Statistical Checks
```python
def validate_smoothing(original, smoothed):
    """Calculate smoothing diagnostics."""
    residuals = original - smoothed
    
    metrics = {
        'mae': np.mean(np.abs(residuals)),
        'rmse': np.sqrt(np.mean(residuals**2)),
        'smoothness': np.mean(np.abs(np.diff(smoothed, 2))),
        'var_reduction': 1 - np.var(residuals) / np.var(original)
    }
    
    return metrics
```

---

## Best Practices

1. **Match method to purpose**
   - Exploration: LOESS or Moving Average
   - Official stats: Henderson
   - Forecasting: Exponential Smoothing

2. **Consider data frequency**
   - High frequency → More smoothing needed
   - Low frequency → Less smoothing needed

3. **Check endpoint behavior**
   - Be aware of endpoint effects
   - Consider asymmetric filters for recent data

4. **Combine with other methods**
   - Seasonal adjustment + Smoothing
   - Transformation + Smoothing

5. **Document parameters**
   - Always report smoothing parameters used
   - Justify parameter choices

---

## References

1. Henderson, R. (1916). "Note on graduation by adjusted average". Transactions of the Actuarial Society of America. 17: 43–48.

2. Holt, C.C. (1957). "Forecasting seasonals and trends by exponentially weighted averages". O.N.R. Memorandum 52. Carnegie Institute of Technology.

3. Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association. 74 (368): 829–836.

4. Savitzky, A., Golay, M.J.E. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures". Analytical Chemistry. 36 (8): 1627–1639.