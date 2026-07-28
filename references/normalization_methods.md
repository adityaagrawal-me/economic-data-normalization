# Data Normalization Methods - Technical Reference

## Overview

This reference covers scaling and transformation techniques for economic data normalization.

## 1. Min-Max Scaling

### Theory
Linear transformation that scales data to a fixed range, typically [0, 1].

### Formula
```
X_scaled = (X - X_min) / (X_max - X_min) × (new_max - new_min) + new_min

Standard [0, 1] scaling:
X_scaled = (X - X_min) / (X_max - X_min)
```

### Implementation
```python
from sklearn.preprocessing import MinMaxScaler

def min_max_scale(data, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to data.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data
    feature_range : tuple, default=(0, 1)
        Desired range (min, max)
        
    Returns:
    --------
    scaled_data : ndarray
        Scaled data
    scaler : MinMaxScaler
        Fitted scaler object for inverse transform
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(
        data.reshape(-1, 1) if data.ndim == 1 else data
    )
    return scaled_data, scaler
```

### Properties
- **Preserves shape** of original distribution
- **Bounded output** in specified range
- **Preserves zero** if range includes zero
- **Sensitive to outliers** (they compress normal range)

### Use Cases
- Neural networks (bounded inputs 0-1)
- Image processing (pixel values)
- Algorithms requiring bounded inputs (e.g., sigmoid activation)
- When preserving ratios important

### Advantages
- Simple and intuitive- All values in specified range
- Fast computation
- Invertible

### Limitations
- Very sensitive to outliers- Doesn't center data
- New data outside training range can exceed bounds
- Not robust

---

## 2. Z-Score Standardization

### Theory
Centers data at zero and scales to unit variance (standard normal distribution).

### Formula
```
X_std = (X - μ) / σ  

where:
  μ = mean of X
  σ = standard deviation of X
```

### Implementation
```python
from sklearn.preprocessing import StandardScaler

def z_score_standardize(data):
    """
    Apply Z-score standardization.
    
    Parameters:
    -----------
    data : array-like
        Input data
        
    Returns:
    --------
    standardized_data : ndarray
        Standardized data (mean=0, std=1)
    scaler : StandardScaler
        Fitted scaler for inverse transform
    """
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(
        data.reshape(-1, 1) if data.ndim == 1 else data
    )
    return standardized_data, scaler
```

### Properties
- **Mean = 0**
- **Standard deviation = 1**
- **No bounded range** (can be any value)
- **Assumes normal distribution** for interpretation

### Use Cases
- Machine learning algorithms (SVM, PCA, clustering)
- Comparing variables with different units
- When centering important
- Regression with standardized coefficients

### Advantages
- Centers at zero (mean comparison)
- Unit variance (fair comparison)
- Widely used and understood
- Works well with normal data

### Limitations
- Sensitive to outliers (use robust scaling if outliers present)
- Not bounded (can't use where range required)
- Assumes data somewhat normally distributed

---

## 3. Robust Scaling

### Theory
Uses median and IQR instead of mean and standard deviation for outlier resistance.

### Formula
```
X_robust = (X - median(X)) / IQR(X)

where:
  IQR = Q3 - Q1 (interquartile range)
  Q1 = 25th percentile
  Q3 = 75th percentile
```

### Implementation
```python
from sklearn.preprocessing import RobustScaler

def robust_scale(data, quantile_range=(25.0, 75.0)):
    """
    Apply robust scaling using median and IQR.
    
    Parameters:
    -----------
    data : array-like
        Input data
    quantile_range : tuple, default=(25.0, 75.0)
        Quantile range for scaling
        
    Returns:
    --------
    scaled_data : ndarray
        Robustly scaled data
    scaler : RobustScaler
        Fitted scaler
    """
    scaler = RobustScaler(quantile_range=quantile_range)
    scaled_data = scaler.fit_transform(
        data.reshape(-1, 1) if data.ndim == 1 else data
    )
    return scaled_data, scaler
```

### Properties
- **Median-centered** (robust to outliers)
- **IQR-scaled** (robust scale measure)
- **Not sensitive** to extreme values
- **No bounded range**

### Use Cases
- Heavy outliers present
- Skewed distributions
- When mean/std unreliable
- Robust preprocessing pipeline

### Advantages
- Outlier-resistant
- No distribution assumptions
- Works with skewed data
- Protects against extreme values

### Limitations
- Not as intuitive as z-score
- IQR can be zero (rare)
- Less standardized interpretation

---

## 4. Log Transformation

### Theory
Applies logarithm to compress large values and expand small values.

### Common Forms
```
Natural log:     X_log = ln(X)
Log base 10:     X_log = log10(X)
Log plus one:    X_log = ln(X + 1)  [if zeros present]
Shifted log:     X_log = ln(X + c)  [if negative values]
```

### Implementation
```pythol
import numpy as np

def log_transform(data, method='natural', handle_zeros='add_one'):
    """
    Apply log transformation.
    
    Parameters:
    -----------
    data : array-like
        Input data (must be positive if handle_zeros=None)
    method : {'natural', 'log10', 'log2'}
        Logarithm base
    handle_zeros : {'add_one', 'add_constant', None}
        How to handle zeros and negative values
        
    Returns:
    --------
    transformed_data : ndarray
        Log-transformed data
    params : dict
        Transformation parameters for inverse
    """
    if handle_zeros == 'add_one':
        shift = 1
    elif handle_zeros == 'add_constant':
        shift = abs(data.min()) + 1 if data.min() <= 0 else 0
    else:
        shift = 0
    
    data_shifted = data + shift
    
    if method == 'natural':
        transformed = np.log(data_shifted)
    elif method == 'log10':
        transformed = np.log10(data_shifted)
    elif method == 'log2':
        transformed = np.log2(data_shifted)
    
    params = {'method': method, 'shift': shift}
    return transformed, params
```

### Inverse Transformation
```python
def inverse_log_transform(transformed_data, params):
    """Reverse log transform."""
    method = params['method']
    shift = params['lhift']
    
    if method == 'natural':
        original = np.exp(transformed_data)
    elif method == 'log10':
        original = np.power(10, transformed_data)
    elif method == 'log2':
        original = np.power(2, transformed_data)
    
    return original - shift
```

### Properties
- **Reduces right skewness**
- **Compresses large values**
- **Expands small values**
- **Stabilizes variance**
- **Makes multiplicative relationships additive**

### Use Cases
- Right-skewed income/wealth data
- Exponential growth patterns (GDP, population)
- Multiplicative economic relationships
- Heteroscedastic data (variance increases with level)

### Advantages
- Reduces skewness
- Stabilizes variance
- Linearizes exponential relationships
- Interpretable (log-returns in finance)

### Limitations
- Requires positive values (or shifting)
- Changes interpretation (log-units)
- Complicates back-transform
- Can over-compress large values

---

## 5. Box-Cox Transformation

### Theory
Power transformation that finds optimal λ parameter to normalize data.

### Formula
```
For λ ∠:  y(î�I = (Xî� - 1) / λ
For î = 0:  y(λ)I = ln(X)

where λ is chosen to maximize normality
```

### Implementation
```python
From scipy import stats

def boxcox_transform(data):
    """
    Apply Box-Cox transformation.
    
    Parameters:
    -----------
    data : array-like
        Input data (must be positive)
        
    Returns:
    ---------
    transformed_data : ndarray
        Box-Cox transformed data
    lambda_param : float
        Optimal lambda parameter
    """
    # Requires positive data
    if np.any(data <= 0):
        shift = abs(data.min()) + 1
        data = data + shift
    else:
        shift = 0
    
    transformed_data, lambda_param = stats.boxcox(data)
    
    return transformed_data, {'lambda': lambda_param, 'shift': shift}
```

### Common λ Values
- λ = 1: No transformation
- λ = 0.5: Square root transformation
- λ = 0: Log transformation
- λ = -1: Reciprocal transformation

### Use Cases
- Achieving normality for hypothesis tests
- Variance stabilization
- When optimal transformation needed
- Regression diagnostics

### Advantages
- Data-driven optimization
- Often achieves normality
- Flexible
- Well-studied method

### Limitations
- Requires positive data
- Computationally more expensive
- Interpretation complex for non-standard λ
- Overfitting risk on small samples

---

## 6. Square Root Transformation

### Theory
Simple power transformation that reduces right skewness.

### Formula
```
X_sqrt = √X or X_sqrt = √(X + c) if zeros/negatives present
```

### Implementation
```python
def sqrt_transform(data):
    """
    Apply square root transformation.
    
    Parameters:
    -----------
    data : array-like
        Input data
        
    Returns:
    --------
    transformed_data : ndarray
        Square root transformed data
    shift : float
        Value added before transformation
    """
    if np.any(data < 0):
        shift = abs(data.min())
        data = data + shift
    else:
        shift = 0
    
    transformed_data = np.sqrt(data)
    return transformed_data, shift
```

### Properties
- **Milder than log** transformation
- **Reduces skewness**
- **Stabilizes variance** (Poisson data)
- **Simple interpretation**

### Use Cases
- Count data (Poisson-distributed)
- Moderately skewed data
- When log too strong
- Variance proportional to mean

### Advantages
- Simple and intuitive
- Less extreme than log
- Works for zeros (with shift)
- Variance stabilization for count data

### Limitations
- Only works for non-negative data
- Limited flexibility- May not fully normalize

---

## 7. Rank Transformation

### Theory
Replaces values with their ranks (ordinal positions).

### Implementation
```python
From scipy.stats import rankdata

def rank_transform(data, method='average'):
    """
    Apply rank transformation.
    
    Parameters:
    -----------
    data : array-like
        Input data
    method : {'average', 'min', 'max', 'dense', 'ordinal'}
        Tie-handling method
        
    Returns:
    ---------
    ranks : ndarray
        Rank-transformed data
    """
    ranks = rankdata(data, method=method)
    return ranks
```

### Tie-Handling Methods
- **average**: Average rank of tied values (default)
- **min**: Minimum rank of tied values
- **max**: Maximum rank of tied values
- **dense**: Like min but with no gaps
- **ordinal**: Each value gets unique rank

### Use Cases
- Non-parametric tests (Spearman correlation)
- Ordinal data
- When scale doesn't matter, only order
- Heavy outliers (ranks unaffected)

### Advantages
- Completely outlier-resistant
- No distribution assumptions
- Simple
- Works with ordinal data

### Limitations
- Loses absolute scale information
- Cannot reverse transformation
- Ties need handling
- Changes interpretation dramatically

---

## Method Selection Guide

### Decision Tree

```
START
  │
  ├─ Need bounded output [0,1]?
  │   └─ YES → Min-Max Scaling
  │
  ├─ Heavy outliers present?
  │   └─ YES → Robust Scaling or Rank Transform
  │
  ├─ Data right-skewed?
  │   ├─ Heavily → Log Transformation
  │   └─ Moderately → Square Root or Box-Cox
  │
  ├─ Need mean=0, std=1?
  │   └─ YES → Z-Score Standardization
  │
  └─ Achieve normality?
      └─ YES → Box-Cox Transformation
```

### Quick Reference Table

| Method | Output Range | Outlier Robust | Skewness Fix | Use Case |
|--------|-------------|----------------|--------------|----------|
| Min-Max | [0,1] | No | No | Bounded inputs needed |
| Z-Score | Any | No | No | ML, comparison |
| Robust | Any | Yes | No | Heavy outliers |
| Log | Any | Moderate | Yes | Right-skewed |
| Box-Cox | Any | No | Yes | Achieve normality |
| Square Root | Any | Moderate | Yes | Count data |
| Rank | [1, n] | Yes | Yes | Ordinal/non-parametric |

---

## Combining Methods

### Common Pipelines

#### Pipeline 1: Robust Preprocessing
```
1. Log transform (if heavily skewed)
2. Robust scaling (center by median, scale by IQR)
```

#### Pipeline 2: ML Preprocessing
```
1. Handle outliers (remove or winsorize)
2. Box-Cox transform (if skewed)
3. Standard scaling (z-score)
```

#### Pipeline 3: Comparison
```
1. Log transform (if scales very different)
2. Z-score standardization
```

---

## Validation

### Check Transformation Success
```python
def validate_transformation(original, transformed):
    """
    Validate transformation effectiveness.
    
    Returns:
    ---------
    metrics : dict
        Skewness\nimprovement, normality tests
    """
    from scipy import stats
    
    metrics = {
        'orihinal_skew': stats.skew(original),
        'transformed_skew': stats.skew(transformed),
        'original_shapiro_p': stats.shapiro(original if len(original) < 5000) else original[:5000][1],
        'transformed_shapiro_p': stats.shapiro(transformed if len(transformed) < 5000) else transformed[:5000][1]
    }
    
    return metrics
```

### Visual Checks
- **Histogram**: Compare distributions before/after
- **Q-Q plot**: Check normality
- **Box plot**: Compare outlier presence- **Scatter plot**: Check variance stability

---

## Economic Data Examples

### Example 1: Income Data (Right-Skewed)
```
Problem: Income very right-skewed (median << mean)
Solution: Log transformation
Why: Reduces influence of billionaires, interpretable as log-income
```

### Example 2: GDP Time Series
```
Problem: Exponential growth pattern
Solution: Log transformation
Why: Makes growth rates (log-returns) linear and interpretable
```

### Example 3: Unemployment Rate
```
Problem: Bounded [0, 100], compare with other indicators
Solution: Z-score standardization (if no outliers) or Robust scaling
Why: Compares unemployment with GDP growth, inflation on same scale
```

### Example 4: Financial Returns
```
Problem: Heavy tails, outliers
Solution: Rank transformation or Robust scaling
Why: Outlier-resistant, preserves order for analysis
```

---

## Inverse Transformations

Always store transformation parameters for inverse operation:
```python
# Forward
transformed, params = transform_function(data)

# Inverse
original = inverse_transform_function(transformed, params)
```

### When Inverse Needed
- Interpreting model predictions
- Reporting in original units
- Backtesting
- Communication with stakeholders

---

## References

1. Box, G.E.P., Cox, D.R. (1964). "An Analysis of Transformations". Journal of the Royal Statistical Society, Series B. 26 (2): 211–252.

2. Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python". Journal of Machine Learning Research. 12: 2825–2830.

3. Osborne, J.W. (2010). "Improving your data transformations: Applying the Box-Cox transformation". Practical Assessment, Research & Evaluation. 15(12): 1-9.

4. Yeo, I.-K., Johnson, R.A. (2000). "A new family of power transformations to improve normality or symmetry". Biometrika. 87 (4): 954–959.