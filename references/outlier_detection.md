# Outlier Detection Methods - Technical Reference

## Overview

This reference provides detailed technical specifications for outlier detection methods used in economic data analysis.

## 1. Z-Score Method

### Theory
The Z-score measures the number of standard deviations a data point is from the mean.

### Formula
```
Z = (X - μ) / σ

where:
  X = individual data point
  μ = population or sample mean
  σ = population or sample standard deviation
```

### Decision Rule
- |Z| > 3: Outlier (captures 99.73% of normal data)
- |Z| > 2: Potential outlier (captures 95.45% of normal data)
- |Z| > 2.5: Conservative threshold

### Implementation
```python
from scipy import stats
import numpy as np

def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers using Z-score method.
    
    Parameters:
    -----------
    data : array-like
        Input data
    threshold : float, default=3
        Z-score threshold for outlier detection
        
    Returns:
    --------
    outlier_indices : ndarray
        Boolean array indicating outliers
    z_scores : ndarray
        Z-scores for all data points
    """
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    outlier_indices = z_scores > threshold
    return outlier_indices, z_scores
```

### Assumptions
- Data approximately normally distributed
- Mean and standard deviation meaningful
- Outliers don't heavily influence statistics

### Use Cases
- Quick screening of large datasets
- When normality assumption reasonable
- Initial exploratory analysis

### Limitations
- Sensitive to extreme outliers (they inflate σ)
- Not robust for small samples (< 20 observations)
- Assumes independence of observations
- Poor performance with heavy-tailed distributions

---

## 2. IQR (Interquartile Range) Method

### Theory
Non-parametric method based on quartile positions, robust to outliers.

### Formula
```
IQR = Q3 - Q1

Lower bound = Q1 - k × IQR
Upper bound = Q3 + k × IQR

where:
  Q1 = 25th percentile (first quartile)
  Q3 = 75th percentile (third quartile)
  k = multiplier, typically 1.5 (Tukey) or 2.0 (more conservative)
```

### Decision Rule
Data points outside [Lower bound, Upper bound] are outliers.

### Implementation
```python
def detect_outliers_iqr(data, k=1.5):
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    data : array-like
        Input data
    k : float, default=1.5
        IQR multiplier (1.5 = Tukey, 2.0 = conservative)
        
    Returns:
    --------
    outlier_indices : ndarray
        Boolean array indicating outliers
    bounds : tuple
        (lower_bound, upper_bound)
    """
    Q1 = np.percentile(data, 25, method='midpoint')
    Q3 = np.percentile(data, 75, method='midpoint')
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    outlier_indices = (data < lower_bound) | (data > upper_bound)
    return outlier_indices, (lower_bound, upper_bound)
```

### Assumptions
- None (distribution-free method)
- Independent observations

### Use Cases
- Skewed distributions
- When robustness is priority
- Box plot visualizations
- Small to medium sample sizes

### Advantages
- Robust to extreme values
- No distribution assumptions
- Easy to interpret and visualize

### Limitations
- Less sensitive than parametric methods
- Fixed thresholds may miss subtle outliers
- Can be too conservative or liberal depending on k

---

## 3. Grubbs' Test

### Theory
Statistical hypothesis test for detecting single outliers in normally distributed data.

### Formula
```
G = |X_extreme - X̄| / s

where:
  X_extreme = most extreme value (max or min)
  X̄ = sample mean
  s = sample standard deviation
```

### Hypothesis Test
```
H0: No outliers present
Ha: Exactly one outlier present

Critical value from t-distribution:
G_critical = ((n-1) / sqrt(n)) * sqrt(t²_(α/(2n), n-2) / (n - 2 + t²_(α/(2n), n-2)))

where:
  n = sample size
  α = significance level (typically 0.05)
  t = t-distribution value
```

### Decision Rule
If G > G_critical, reject H0 (outlier present at significance level α).

### Implementation
```python
from scipy import stats

def grubbs_test(data, alpha=0.05):
    """
    Perform Grubbs' test for outliers.
    
    Parameters:
    -----------
    data : array-like
        Input data (must be approximately normal)
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    is_outlier : bool
        True if outlier detected
    extreme_value : float
        Most extreme value tested
    G_stat : float
        Grubbs test statistic
    p_value : float
        P-value of the test
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Calculate G for both min and max
    abs_diff = np.abs(data - mean)
    max_idx = np.argmax(abs_diff)
    extreme_value = data[max_idx]
    G_stat = np.max(abs_diff) / std
    
    # Critical value
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_critical = ((n - 1) * np.sqrt(np.power(t_dist, 2))) / \
                 (np.sqrt(n) * np.sqrt(n - 2 + np.power(t_dist, 2)))
    
    is_outlier = G_stat > G_critical
    
    # Calculate p-value
    p_value = 1 - stats.t.cdf(G_stat * np.sqrt(n) / np.sqrt(n - 1), n - 2)
    
    return is_outlier, extreme_value, G_stat, p_value
```

### Assumptions
- Data normally distributed
- Single outlier at a time
- Independent observations

### Use Cases
- Formal statistical testing required
- Normal or near-normal data
- Need p-value for reporting
- Regulatory or compliance contexts

### Iterative Application
For multiple outliers, apply iteratively:
1. Perform Grubbs' test
2. If outlier detected, remove it
3. Repeat on remaining data
4. Stop when no outlier detected or maximum iterations reached

### Limitations
- Requires normality (test Shapiro-Wilk first)
- One outlier per test (less efficient for multiple outliers)
- Sensitive to sample size
- Cannot detect masking (multiple outliers hiding each other)

---

## 4. DBSCAN (Density-Based Spatial Clustering)

### Theory
Density-based clustering algorithm that identifies outliers as noise points not belonging to any cluster.

### Parameters
```
ε (epsilon): Maximum distance between points in same neighborhood
minPts: Minimum points required to form dense region (core point)
```

### Classification
- **Core point**: Has ≥ minPts points within ε distance
- **Border point**: Within ε of core point, but < minPts neighbors
- **Noise point**: Neither core nor border → **OUTLIER**

### Implementation
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def detect_outliers_dbscan(data, eps='auto', min_samples=5):
    """
    Detect outliers using DBSCAN clustering.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data (can be multidimensional)
    eps : float or 'auto'
        Maximum distance between points
        If 'auto', uses k-distance plot heuristic
    min_samples : int, default=5
        Minimum points to form dense region
        
    Returns:
    --------
    outlier_indices : ndarray
        Boolean array indicating outliers
    labels : ndarray
        Cluster labels (-1 = outlier)
    """
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1) 
                                       if data.ndim == 1 
                                       else data)
    
    # Auto-determine epsilon if needed
    if eps == 'auto':
        from sklearn.neighbors import NearestNeighbors
        nbbrs = NearestNeighbors(n_neighbors=min_samples).fit(data_scaled)
        distances, _ = nbrrs.knearbors(data_scaled)
        eps = np.percentile(distances[:, -1], 90)
    
    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
    labels = clustering.labels_
    
    # Outliers have label = -1
    outlier_indices = labels == -1
    
    return outlier_indices, labels
```

### Choosing Parameters

#### epsilon (õ)
- **Too small**: Many points become noise
- **Too large**: All points in one cluster
- **Heuristic**: k-distance plot (plot sorted k-nearest-neighbor distances)

#### minPts
- **Rule of thumb**: minPts ≥ dimensions + 1
- **Small datasets**: minPts = 4 or 5
- **Large datasets**: minPts = 10-20
- **Economic data**: Often 5-10 works well

### Use Cases
- Multidimensional outlier detection
- Unknown number of outliers
- Clusters of varying shapes and sizes
- Non-Gaussian distributions

### Advantages
- No distribution assumptions
- Handles arbitrary cluster shapes
- Finds outliers automatically
- Works in multiple dimensions

### Limitations
- Requires parameter tuning
- Struggles with varying densities
- Not suitable for high dimensions (curse of dimensionality)
- Computationally expensive for large datasets

---

## 5. Modified Z-Score (MAD-based)

### Theory
Robust alternative to Z-score using Median Absolute Deviation (MAD).

### Formula
```
Modified Z-score = 0.6745 × (X - median(X)) / MED

where:
  MAD = median(|X - median(X|))
  0.6745 = constant to make MAD consistent with σ for normal data
```

### Decision Rule
|Modified Z-score| > 3.5 indicates outlier (more conservative than regular Z-score).

### Implementation
```python
def detect_outliers_modified_zscore(data, threshold=3.5):
    """
    Detect outliers using Modified Z-score (MAD-based).
    
    Parameters:
    -----------
    data : array-like
        Input data
    threshold : float, default=3.5
        Modified Z-score threshold
        
    Returns:
    --------
    outlier_indices : ndarray
        Boolean array indicating outliers
    modified_z_scores : ndarray
        Modified Z-scores for all points
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Avoid division by zero
    if mad == 0:
        mad = np.mean(np.abs(data - median))
    
    modified_z_scores = 0.6745 * (data - median) / mad
    outlier_indices = np.abs(modified_z_scores) > threshold
    
    return outlier_indices, modified_z_scores
```

### Advantages
- Robust to outliers (uses median, not mean)
- No distribution assumptions
- Better than regular Z-score for skewed data

### Use Cases
- When regular Z-score too sensitive
- Skewed distributions
- Presence of extreme values

---

## Method Selection Guide

### Decision Tree

```
START
  │
  ├─ Multiple dimensions?
  │   └─ YES → Use DBSCAN
  │
  ├─ Formal statistical test needed?
  │   └─ YES → Grubbs' Test (if normal)
  │
  ├─ Data normal or near-normal?
  │   ├─ YES → Z-score method
  │   └─ NO → Is data heavily skewed?
  │      ├─ YES → IQR or Modified Z-score
  │     └─ NN → IQR method
  │
  └─ Very small sample (n < 20)?
      └─ YES → IQR method (most robust)
```

### Quick Reference Table

| Method | Distribution | Sample Size | Outliers | Speed | Robustness |
|--------|-------------|-------------|---------|-------|------------|
| Z-score | Normal | Medium-Large | Few | Fast | Low |
| IQR | Any | Small-Large | Any | Fast | High |
| Grubbs | Normal | Small-Medium | Single | Medium | Medium |
| DBSCAN | Any | Medium-Large | Multiple | Slow | High |
| Modified Z | Any | Small-Large | Any | Fast | High |

---

## Treatment Options After Detection

### 1. Removal
Simply delete outlier observations.
- **Pro**: Clean dataset
- **Con**: Loss of information, biased if outliers meaningful

### 2. Winsorization
Cap outliers at threshold (e.g., 5th and 95th percentiles).
```python
def winsorize(data, limits=(0.05, 0.05)):
    from scipy.stats.mstats import winsorize as sp_winsorize
    return sp_winsorize(data, limits=limits)
```

### 3. Transformation
Apply transformation to reduce influence (log, square-root, Box-Cox).

### 4. Imputation
Replace outliers with median, mean, or model-based prediction.

### 5. Flag and Keep
Mark outliers but retain in dataset for separate analysis.

---

## Validation and Diagnostics

### Visual Inspection
- **Box plots**: Show IQR method outliers clearly
- **Scatter plots**: Reveal multivariate outliers
- **Q-Q plots**: Check normality assumption
- **Residual plots**: Identify regression outliers

### Statistical Tests
- **Shapiro-Wilk**: Test normality before Grubbs
- **Levene's test**: Check homoscedasticity
- **Kolmogorov-Smirnov**: Compare distributions

### Domain Knowledge
Always consider:
- Are outliers data errors or valid extreme values?
- Do outliers have economic meaning (e.g., crisis periods)?
- Will removal bias analysis?

---

## References

1. Grubbs, F.E. (1950). "Sample Criteria for Testing Outlying Observations". Annals of Mathematical Statistics. 21 (1): 27–58.

2. Tukey, J.W. (1977). Exploratory Data Analysis. Addison-Wesley.

3. Ester, M., Kriegel, H.P., Sander, J., Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise". Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96).

4. Boris Iglewicz and David Hoaglin (1993). "Volume 16: How to Detect and Handle Outliers", The ASQC Basic References in Quality Control: Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.