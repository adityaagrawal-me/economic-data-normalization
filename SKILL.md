---
description: Comprehensive data cleaning, smoothing, and normalization toolkit for
  economic time series and cross-sectional data. Automatically detects and handles
  outliers, applies appropriate transformations (z-score, min-max, log, robust scaling),
  performs seasonal adjustment (X-13ARIMA-SEATS, moving averages, STL decomposition),
  and smooths data using Henderson filters, exponential smoothing, or LOESS. Intelligently
  selects methods based on data characteristics (distribution, seasonality, outliers,
  skewness).
metadata:
  default_prompt: Normalize this GDP time series data and remove seasonal patterns
  display_name: Economic Data Normalization
  icon_path: assets/icon.png
  short_description: Clean, normalize, and smooth economic data with automatic method
    selection
name: economic-data-normalization
---

# Economic Data Normalization

This skill provides production-grade data cleaning, normalization, and smoothing for economic time series and cross-sectional datasets. It intelligently selects and applies appropriate techniques based on data characteristics.

## What This Skill Does

- **Outlier Detection & Treatment**: Identifies anomalies using Z-score, IQR, Grubbs' test, or DBSCAN
- **Data Normalization**: Applies min-max scaling, z-score standardization, log transformation, or robust scaling
- **Economic-Specific Normalizations**: GDP-based ratios, per capita, YoY growth, real terms, PPP adjustments, HICP standardization
- **Seasonal Adjustment**: Removes seasonal patterns using X-13ARIMA-SEATS, STL decomposition, or moving averages
- **Smoothing & Filtering**: Applies Henderson filters, exponential smoothing, moving averages, or LOESS
- **Automatic Method Selection**: Analyzes data properties and recommends optimal techniques

## When to Use This Skill

Use this skill when you need to:
- Clean economic datasets before analysis or modeling
- Remove seasonal effects from monthly/quarterly data
- Normalize variables for comparison or machine learning
- Apply economic-specific normalizations (debt-to-GDP, per capita, YoY growth, real terms)
- Convert between nominal and real values using price indices
- Calculate standard economic ratios and growth rates
- Detect and handle outliers in financial or economic data
- Smooth volatile time series to reveal underlying trends
- Prepare data for econometric models or forecasting

## Invocation

```
/economic-data-normalization <task description>
```

**Examples:**
- `/economic-data-normalization clean this unemployment data and remove outliers`
- `/economic-data-normalization seasonally adjust monthly retail sales`
- `/economic-data-normalization normalize GDP, inflation, and unemployment for comparison`
- `/economic-data-normalization calculate debt-to-GDP ratio`
- `/economic-data-normalization convert to per capita and real terms`
- `/economic-data-normalization calculate year-over-year growth rates`
- `/economic-data-normalization smooth this volatile stock price series`
- `/economic-data-normalization detect outliers in this cross-sectional dataset`

## How It Works

### 1. Data Assessment

The skill first analyzes your data:
- Distribution (normal, skewed, heavy-tailed)
- Presence of outliers (extreme values)
- Seasonality patterns (monthly, quarterly, weekly)
- Time series properties (stationarity, trend)
- Scale and range characteristics

### 2. Method Selection

Based on assessment, the skill recommends:

**For Outlier Detection:**
- Z-score (3ς rule) → Normal distributions with few outliers
- IQR method → Skewed data or robust detection needed
- Grubbs' test → Formal hypothesis test for single outliers
- DBSCAN → Complex multi-dimensional data with clusters

**For Normalization:**
- Min-Max scaling → Bounded outputs needed (0-1 or custom range)
- Z-score standardization → Comparing variables on same scale
- Log transformation → Reducing skewness or multiplicative relationships
- Robust scaling → Heavy outlier presence, use median/IQR

**For Seasonal Adjustment:**
- X-13ARIMA-SEATS → Official statistics, monthly/quarterly only
- STL decomposition → Flexible, any seasonal period, robust to outliers
- Moving averages → Quick smoothing, simple seasonal removal

**For Smoothing:**
- Henderson filters → Economic trends without distorting cycles
- Exponential smoothing → Recent data more important, forecasting
- Moving averages → Simple trend extraction, noise reduction
- LOESS → Non-parametric, flexible local regression

### 3. Execution

The skill applies selected methods in appropriate order:
1. Outlier detection/treatment (if requested)
2. Transformation (if needed for normalization)
3. Seasonal adjustment (for time series)
4. Smoothing/filtering (if requested)

### 4. Output

Returns:
- Cleaned/normalized data in same format as input
- Summary of methods applied and why
- Diagnostic plots (before/after comparisons)
- Statistical summaries of transformations
- Recommendations for further analysis

## Detailed Method Descriptions

### Outlier Detection Methods

#### Z-Score Method
**Formula:** Z = (X - μ) / σ  
**Threshold:** |Z| > 3 (captures ~99.7% of normal data)  
**Use when:**
- Data approximately normal
- Few outliers expected
- Quick screening needed

**Treatment options:**
- Remove outliers
- Winsorize (cap at threshold)
- Replace with median/mean

#### IQR (Interquartile Range) Method
**Formula:** Outliers outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]  
**Use when:**
- Data skewed or non-normal
- Robust method needed
- Box plot visualization desired

**Advantages:**
- Not affected by extreme values
- No distribution assumptions
- Works with small samples

#### Grubbs' Test
**Statistical test:** Tests if most extreme value is outlier  
**Null hypothesis:** No outliers present  
**Use when:**
- Need formal statistical test
- Single outlier at a time
- Normally distributed data

**Process:**
1. Calculate G = |X_extreme - mean| / std_dev
2. Compare to critical value from t-distribution
3. If p < 0.05, reject null (outlier present)
4. Remove and repeat if needed

#### DBSCAN Clustering
**Parameters:** ε (neighborhood radius), minPts (minimum points)  
**Use when:**
- Multi-dimensional data
- Clusters of varying shapes
- Non-Gaussian distributions

**Advantages:**
- Discovers arbitrary cluster shapes
- Handles noise automatically
- No need to specify number of clusters

### Normalization Methods

#### Min-Max Scaling
**Formula:** X_scaled = (X - X_min) / (X_max - X_min) × (new_max - new_min) + new_min  
**Default range:** [0, 1]  
**Use when:**
- Bounded output required
- Preserving zero values important
- Neural networks or algorithms sensitive to scale

**Properties:**
- Preserves relationships between values
- Sensitive to outliers
- All values in specified range

#### Z-Score Standardization
**Formula:** X_std = (X - μ) / σ  
**Result:** Mean = 0, Std Dev = 1  
**Use when:**
- Comparing variables with different units
- Machine learning requires standardization
- Gaussian-distributed data

**Properties:**
- Centers data at zero
- No bounded range
- Assumes normal distribution

#### Log Transformation
**Formula:** X_log = log(X) or log(X + 1) if zeros present  
**Use when:**
- Right-skewed distributions
- Multiplicative relationships
- Reducing large value influence

**Properties:**
- Reduces skewness
- Stabilizes variance
- Makes exponential growth linear

#### Robust Scaling
**Formula:** X_robust = (X - median) / IQR  
**Use when:**
- Heavy outliers present
- Median-based centering preferred
- Standard scaling too sensitive

**Properties:**
- Uses median instead of mean
- Uses IQR instead of std dev
- Outlier-resistant

### Economic-Specific Normalizations

#### GDP-Based Normalizations
**Debt-to-GDP Ratio:**
```
Debt-to-GDP (%) = (Total Debt / GDP) × 100
```
**Use cases:** Sovereign debt analysis, fiscal sustainability, cross-country comparisons

**Trade-to-GDP Ratio:**
```
Trade Openness (%) = ((Exports + Imports) / GDP) × 100
```
**Use cases:** Measuring economic openness, trade dependency analysis

**Other ratios:** Deficit-to-GDP, government spending-to-GDP, tax revenue-to-GDP

#### Per Capita Normalization
**Formula:** Value per capita = Total Value / Population  
**Use cases:**
- GDP per capita (living standards)
- Income per capita (wealth distribution)
- Debt per capita (per-person burden)
- Emissions per capita (environmental impact)

#### Year-over-Year Growth
**Formula:** YoY Growth (%) = ((Value_t - Value_{t-n}) / Value_{t-n}) × 100  
where n = periods per year (12 for monthly, 4 for quarterly)

**Use cases:**
- Inflation measurement (CPI year-over-year)
- GDP growth reporting
- Wage growth tracking

**Variants:**
- Month-over-month (MoM)
- Quarter-over-quarter annualized (QoQ SAAR)
- Cumulative growth rates

#### Real Terms (Inflation Adjustment)
**Formula:** Real Value = (Nominal Value / Price Index) × Base Year Index

**Common price indices:**
- **CPI**: Consumer Price Index
- **PCE**: Personal Consumption Expenditures
- **GDP Deflator**: All goods/services in GDP
- **HICP**: Harmonized Index (EU standard)

**Use cases:**
- Real wage calculation
- Real GDP measurement
- Historical comparisons
- Living standards analysis

#### Index Standardization
**Base Period = 100:**
```
Index_t = (Value_t / Value_base) × 100
```

**HICP Standardization (EU):**
- Standard base: 2015 = 100
- Used for cross-country EU comparisons
- ECB inflation target reference

#### Purchasing Power Parity (PPP)
**Formula:** PPP Value = Local Currency Value × (PPP Factor / Exchange Rate)

**Use cases:**
- Cross-country GDP comparisons
- International poverty measurement
- Wage comparisons across countries

#### Component Contribution Analysis
**Formula:** Contribution = (ΔComponent / Component_{t-1}) × (Component_{t-1} / Total_{t-1}) × 100

**Use cases:**
- GDP growth decomposition (C, I, G, NX contributions)
- Inflation breakdown (core vs energy)
- Employment growth by sector

### Seasonal Adjustment Methods

#### X-13ARIMA-SEATS
**Official method:** U.S. Census Bureau standard  
**Supports:** Monthly or quarterly data only  
**Use when:**
- Publishing official statistics
- Need SEATS signal extraction
- Automatic ARIMA model selection desired

**Features:**
- Automatic outlier detection
- Trading day/holiday adjustments
- Trend-cycle decomposition
- Diagnostic statistics

**Limitations:**
- Requires X-13 binary installation
- Monthly/quarterly only
- More complex setup

#### STL Decomposition
**Method:** Seasonal-Trend decomposition using LOESS  
**Supports:** Any seasonal period  
**Use when:**
- Flexible seasonal period needed
- Multiple seasonal patterns
- Robust to outliers desired

**Components:**
- Trend (long-term movement)
- Seasonal (repeating pattern)
- Remainder (irregular)

**Advantages:**
- Works with any frequency
- Robust to outliers
- Updates easily with new data

#### Moving Average Seasonal Adjustment
**Method:** Ratio-to-moving-average or difference-from-moving-average  
**Use when:**
- Quick seasonal removal needed
- Simple method preferred
- Educational/exploratory analysis

**Process:**
1. Calculate centered moving average (trend)
2. Detrend data (ratio or difference)
3. Average seasonal indices by period
4. Divide or subtract seasonal component

### Smoothing Methods

#### Henderson Filters
**Purpose:** Smooth without distorting polynomial trends  
**Common lengths:** 5, 9, 13, 23 terms  
**Use when:**
- Economic trend extraction
- Cycles should be preserved
- Official statistics smoothing

**Properties:**
- Symmetric weights
- Preserves polynomials up to degree 3
- Minimizes variance of smoothed series

#### Exponential Smoothing
**Formula:** S_t = α × X_t + (1 - α) × S_{t-1}  
**Parameter:** α (smoothing constant, 0 < α < 1)  
**Use when:**
- Recent data more important
- Forecasting needed
- Weighted average desired

**Variants:**
- Simple (level only)
- Double (level + trend)
- Triple (level + trend + seasonality)

#### Moving Averages
**Types:** Simple, weighted, exponential  
**Window:** Choose based on data frequency  
**Use when:**
- Simple smoothing needed
- Noise reduction
- Quick trend visualization

**Common windows:**
- 3-month: Short-term smoothing
- 12-month: Annual smoothing
- Custom: Based on data characteristics

#### LOESS (Local Regression)
**Method:** Local polynomial regression  
**Parameters:** Bandwidth (span), polynomial degree  
**Use when:**
- Non-parametric smoothing
- Flexible, data-driven approach
- Complex non-linear trends

**Advantages:**
- No functional form assumed
- Adapts to local data structure
- Robust to outliers (optional)

## Workflow Examples

### Example 1: Cleaning Quarterly GDP Data

**Input:**
```
/economic-data-normalization clean quarterly GDP data, remove outliers, apply seasonal adjustment
```

**Process:**
1. Load GDP data
2. Check for normality (Shapiro-Wilk test)
3. Detect outliers using Grubbs' test (assuming normal)
4. Apply X-13ARIMA-SEATS seasonal adjustment
5. Plot before/after comparison

**Output:**
- Seasonally adjusted GDP series
- Outlier report (dates and values removed)
- Diagnostic plots
- Specification file used

### Example 2: Normalizing Cross-Sectional Variables

**Input:**
```
/economic-data-normalization normalize income, education, and age variables for regression model
```

**Process:**
1. Assess distributions (income likely skewed)
2. Apply log transformatio} to income
3. Apply z-score standardization to all variables
4. Check for outliers post-normalization
5. Report transformations

**Output:**
- Normalized variables (mean=0, std=1)
- Transformation summary
- Correlation matrix before/after
- Recommendations for model

### Example 3: Smoothing Volatile Stock Prices

**Input:**
```
/economic-data-normalization smooth daily stock prices to reveal trend
```

**Process:**
1. Assess volatility
2. Apply 21-day (1-month) moving average
3. Optionally apply exponential smoothing (α=0.1)
4. Plot original vs smoothed

**Output:**
- Smoothed price series
- Trend identification
- Volatility metrics
- Trading signals (if requested)

## Best Practices

### 1. Always Inspect Data First
- Plot raw data
- Check summary statistics
- Identify obvious issues

### 2. Choose Methods Based on Purpose
- **For comparison:** Standardize (z-score)
- **For ML models:** Scale appropriately (often min-max or z-score)
- **For visualization:** Smooth but don't over-smooth
- **For forecasting:** Seasonal adjust but preserve information

### 3. Document All Transformations
- Keep record of methods applied
- Note parameters used
- Save original data
- Report any data removed

### 4. Validate Results
- Plot before/after
- Check if issues resolved
- Verify statistical properties
- Test downstream analysis

### 5. Consider Data Frequency
- **Daily:** Use short moving averages, check for day-of-week effects
- **Monthly:** X-13ARIMA-SEATS, 12-month seasonality
- **Quarterly:** X-13ARIMA-SEATS, 4-quarter seasonality
- **Annual:** Focus on trend, detrending if needed

## Constraints & Limitations

### Data Requirements
- Minimum 3 years for reliable seasonal adjustment
- At least 20 observations for outlier detection methods
- Regular frequency for time series methods

### Method Limitations
- X-13ARIMA-SEATS: Monthly/quarterly only, requires external binary
- Z-score: Assumes normality, sensitive to outliers
- Grubbs' test: One outlier at a time, normality assumption
- Log transformation: Requires positive values

### When NOT to Use Certain Methods
- **Don't use min-max scaling** if outliers present (will compress normal range)
- **Don't use z-score outlier detection** on heavy-tailed distributions
- **Don't over-smooth** if high-frequency information important
- **Don't blindly remove outliers** without understanding their cause

## Error Handling

The skill will:
- Warn if data too short for method
- Suggest alternative if method unavailable
- Report if assumptions violated
- Provide fallback options

**Common issues:**
- "Data not normally distributed" → Suggests IQR instead of Z-score
- "Insufficient observations" → Recommends simpler methods
- "No seasonality detected" → Skips seasonal adjustment
- "X-13 binary not found" → Falls back to STL decomposition

## Output Format

All outputs include:

1. **Processed Data:**
   - Same format as input (CSV, pandas DataFrame, etc.)
   - Column naming: original_name_normalized, original_name_sa, etc.

2. **Transformation Report:**
   - Methods applied
   - Parameters used
   - Diagnostic statistics
   - Data removed/modified

3. **Diagnostic Plots:**
   - Before/after time series plots
   - Distribution comparisons (histograms, Q-Q plots)
   - Seasonal decomposition (if applicable)
   - Outlier identification visualization

4. **Recommendations:**
   - Further steps suggested
   - Potential issues flagged
   - Alternative methods to consider

## Advanced Usage

### Combining Multiple Methods

```
/economic-data-normalization clean GDP data: detect outliers using IQR, apply log transformation, then X-13 seasonal adjustment, and finally 13-term Henderson smoothing
```

The skill will apply methods in optimal order automatically.

### Custom Parameters

```
/economic-data-normalization normalize using robust scaling with IQR multiplier 2.0 instead of default 1.5
```

The skill accepts custom parameters for fine-tuning.

### Batch Processing

```
/economic-data-normalization normalize all variables in this dataset using appropriate method for each based on distribution
```

The skill can process multiple variables with different methods.

## References

For detailed methodology, see:
- `references/outlier_detection.md` - Outlier detection techniques
- `references/normalization_methods.md` - Scaling and transformation methods  
- `references/seasonal_adjustment.md` - Seasonal adjustment approaches
- `references/smoothing_techniques.md` - Filtering and smoothing methods

For implementation details:
- `scripts/detect_outliers.py` - Outlier detection implementations
- `scripts/normalize_data.py` - Normalization/scaling functions
- `scripts/seasonal_adjust.py` - Seasonal adjustment methods
- `scripts/smooth_data.py` - Smoothing and filtering functions
- `scripts/analyze_data.py` - Data assessment and method selection

## Support

This skill leverages:
- **scipy.stats** - Statistical tests and distributions
- **statsmodels** - X-13ARIMA-SEATS interface, STL decomposition
- **scikit-learn** - Scaling methods, DBSCAN
- **pandas** - Data manipulation
- **numpy** - Numerical operations

All dependencies are pre-installed in the Writer Agent environment.