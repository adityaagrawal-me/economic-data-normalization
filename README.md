# Economic Data Normalization Skill

Comprehensive skill for cleaning, normalizing, and processing economic time series data.

## Overview

This skill provides intelligent, automated data processing capabilities specifically designed for economic data:

- **Outlier Detection**: Z-score, IQR, Grubbs' test, DBSCAN
- **Normalization**: Min-Max, Z-score, Robust scaling, Log/Box-Cox transforms
- **Seasonal Adjustment**: X-13ARIMA-SEATS, STL, Classical decomposition
- **Smoothing**: Henderson filters, Exponential smoothing, LOESS, Moving averages

## Key Features

✅ **Automatic Method Selection** - Analyses data characteristics and recommends optimal methods  
✅ **Economic-Specific Normalizations** - GDP ratios, per capita, growth rates, real terms, PPP adjustments  
✅ **Production-Ready** - Battle-tested algorithms used in official economic statistics  
✅ **Comprehensive Diagnostics** - Detailed reports and quality metrics  
✅ **Publication-Quality** - Methods suitable for academic and professional use

## Quick Start

### Basic Usage

```python
import pandas as pd
from process_data import EconomicDataProcessor

# Load your economic data
data = pd.read_csv('gdp_data.csv', index_col=0, parse_dates=True)
series = data['GDP_Growth']

# Create processor
processor = EconomicDataProcessor(series, name="GDP Growth")

# Run full pipeline with automatic method selection
processed = processor.process_full_pipeline(
    detect_outliers=True,
    seasonal_adjust=True,
    smooth=True,
    normalize=False
)

# Generate report and plots
report = processor.generate_summary_report()
plots = processor.generate_diagnostic_plots()

print(report)
```

### Step-by-Step Processing

```python
# Manual control over each step
processor = EconomicDataProcessor(data, name="CPI")

# 1. Analyze data
assessment = processor.analyze()

# 2. Detect and remove outliers
processor.detect_outliers(method='iqr', treatment='remove')

# 3. Seasonally adjust (if time series)
processor.seasonal_adjust(method='stl', period=12)

# 4. Normalize for comparison
processor.normalize(method='z-score', purpose='comparison')

# 5. Smooth to reveal trend
processor.smooth(method='henderson')

# Get processed data
final_data = processor.processed_data
```

## Method Selection Guide

### When to Use Each Method

#### Outlier Detection

| Method | Best For | Pros | Cons |
|-------|-----------|------|------|
| Z-Score | Normal distributions | Fast, simple | Assumes normality |
| IQR | Any distribution | Robust, non-parametric | May miss subtle outliers |
| Grubbs | Normal, few outliers | Formal statistical test | Assumes normality |
| DBSCAN | Complex patterns | Finds clusters | Requires parameter tuning |

#### Normalization

| Method | Best For | Output Range |
|--------|----------|--------------|
| Min-Max | Visualization, bounded output | [0, 1] or custom |
| Z-Score | Comparison, ML | Mean=0, Std=1 |
| Robust | Data with outliers | Median-centered |
| Log | Right-skewed data | Reduces skewness |

#### Seasonal Adjustment

| Method | Frequencies | Use Case |
|--------|-------------|----------|
| X-13ARIMA-SEATS | Monthly, Quarterly | Official statistics |
| STL | Any period | General purpose, robust |
| Classical | Any period | Quick exploration |

#### Smoothing

| Method | Best For | Characteristics |
|--------|----------|-----------------|
| Henderson | Economic trends | Preserves polynomials |
| Exponential | Forecasting | Adaptive weights |
| Moving Average | Quick smoothing | Simple, intuitive |
| LOESS | Non-parametric | Very flexible |

## Economic-Specific Normalizations

### Using the Economic Normalizer

```python
from economic_normalizations import EconomicNormalizer

# Initialize normalizer
econ_norm = EconomicNormalizer()

# Example 1: Calculate Debt-to-GDP Ratio
debt_to_gdp = econ_norm.normalize_by_gdp(
    value=government_debt,  # in billions
    gdp=gdp,                # in billions
    multiply_by=100         # for percentage
)
print(f"Debt-to-GDP: {debt_to_gdp.iloc[-1]:.1f}%")

# Example 2: Convert to Per Capita
gdp_per_capita = econ_norm.per_capita(
    value=gdp,
    population=population_millions
)
print(f"GDP per capita: ${gdp_per_capita.iloc[-1]:,.0f}")

# Example 3: Calculate Year-over-Year Inflation
cpi_inflation = econ_norm.year_over_year_growth(
    data=cpi,
    periods=12,  # 12 months for monthly data
    method='percentage'
)
print(f"Latest inflation: {cpi_inflation.iloc[-1]:.1f}%")

# Example 4: Convert to Real Terms
real_wages = econ_norm.real_terms(
    nominal=nominal_wages,
    price_index=cpi,
    base_year_index=100
)

# Example 5: Quarter-over-Quarter Annualized Growth (US GDP style)
gdp_growth_saar = econ_norm.quarter_over_quarter_annualized(
    data=quarterly_gdp,
    periods=1
)
print(f"GDP growth (SAAR): {gdp_growth_saar.iloc[-1]:.1f}%")

# Example 6: PPP Adjustment for Cross-Country Comparison
gdp_ppp = econ_norm.purchasing_power_parity(
    value=gdp_local_currency,
    exchange_rate=local_per_usd,
    ppp_conversion_factor=world_bank_ppp_factor
)

# Example 7: Component Contribution to GDP Growth
consumption_contrib = econ_norm.contribution_to_growth(
    component=consumption,
    total=gdp,
    periods=4  # Year-over-year for quarterly
)
```

### HICP Standardization (EU)

```python
from economic_normalizations import normalize_hicp

# Standardize to 2015 = 100 (EU standard)
hicp_series = normalize_hicp(
    data=price_index,
    reference_year=2015,
    base_index=100.0
)

# Now comparable across EU countries
```

### Chain-Weighted Indices

```python
from economic_normalizations import normalize_chain_weighted

# Calculate chain-weighted real GDP
gdp_chain = normalize_chain_weighted(
    components={
        'C': consumption,
        'I': investment,
        'G': government,
        'NX': net_exports
    },
    weights=None  # Auto-calculated from shares
)
```

### Complete Economic Analysis Workflow

```python
from economic_normalizations import EconomicNormalizer
import pandas as pd

# Load economic data
data = pd.read_csv('macro_data.csv', index_col=0, parse_dates=True)

normalizer = EconomicNormalizer()

# Step 1: Calculate key economic ratios
debt_gdp = normalizer.normalize_by_gdp(data['Debt'], data['GDP'], 100)
deficit_gdp = normalizer.normalize_by_gdp(data['Deficit'], data['GDP'], 100)

# Step 2: Convert to per capita
gdp_pc = normalizer.per_capita(data['GDP'], data['Population'])
income_pc = normalizer.per_capita(data['Income'], data['Population'])

# Step 3: Calculate growth rates
gdp_yoy = normalizer.year_over_year_growth(data['GDP'], periods=4)
cpi_yoy = normalizer.year_over_year_growth(data['CPI'], periods=12)

# Step 4: Convert to real terms
real_gdp = normalizer.real_terms(data['GDP'], data['CPI'], 100)
real_wages = normalizer.real_terms(data['Wages'], data['CPI'], 100)

# Step 5: Index to base period
gdp_index = normalizer.index_to_base_period(data['GDP'], base_value=data['GDP'].iloc[0])

# Create summary DataFrame
summary = pd.DataFrame({
    'Debt_to_GDP_%': debt_gdp,
    'Deficit_to_GDP_%': deficit_gdp,
    'GDP_per_capita': gdp_pc,
    'GDP_growth_YoY_%': gdp_yoy,
    'CPI_inflation_YoY_%': cpi_yoy,
    'Real_GDP': real_gdp,
    'GDP_Index_2020=100': gdp_index
})

# Check transformations applied
print("Applied transformations:", normalizer.transformations_applied)
```

## Advanced Examples

### Custom Pipeline

```python
# Create custom processing sequence
processor = EconomicDataProcessor(data)

# Only clean and normalize (no seasonal/smoothing)
processor.analyze()
processor.detect_outliers(method='iqr', treatment='winsorize')
processor.normalize(method='robust-scaling', purpose='ml')

# Export results
cleaned_data = processor.processed_data.to_csv('cleaned_data.csv')
```

### Multiple Series Processing

```python
# Process multiple related series
gdp_data = pd.read_csv('macro_data.csv', index_col=0, parse_dates=True)

results = {}
for column in ['GDP', 'CPI', 'Unemployment']:
    processor = EconomicDataProcessor(
        gdp_data[column], 
        name=column
    )
    
    results[column] = processor.process_full_pipeline(
        detect_outliers=True,
        seasonal_adjust=True,
        smooth=False,
        normalize=True
    )

# Combine normalized series for comparison
normalized_df = pd.DataFrame(results)
```

### Quality Analysis

```python# Detailed quality assessment
from analyze_data import assess_data, quick_assess

# Full assessment
assessment = assess_data(data.values, "My Series")
print(f"Distribution: {assessment['distribution']}")
print(f"Skewness: {assessment['skewness']:.3f}")
print(f"Outliers: {assessment['n_outliers_z3']}")

# Quick summary
summary = quick_assess(data.values)
print(summary)
```

## Output Files

### Reports

The skill generates comprehensive markdown reports including:
- Descriptive statistics
- Distribution characteristics
- Outlier assessment
- Method recommendations
- Processing summary

### Diagnostic Plots

Automatically generated plots:
1. **Comparison Plot**: Original vs Processed time series
2. **Distribution Plot**: Histograms before and after processing

## Dependencies

Required Python packages (install via `pip install -r requirements.txt`):
- numpy
- pandas
- scipy
- scikit-learn
- statsmodels (required for STL, classical, and X-13 seasonal adjustment; if absent, seasonal adjustment falls back to the moving-average method)
- matplotlib
> **Note:** These packages are **not** automatically installed. Install them before using the skill.
> If `statsmodels` is not available, seasonal adjustment automatically falls back to the
> pure-pandas moving-average method so the pipeline still runs.

## Best Practices

### 1. Always Start with Analysis
```python
processor.analyze()  # Understand your data first
```

### 2. Process in Correct Order
Recommended sequence:
1. Outlier detection
2. Seasonal adjustment (for time series)
3. Normalization
4. Smoothing

### 3. Document Your Choices
```python# Keep track of processing steps
steps = processor.steps_applied
report = processor.generate_summary_report()
```

### 4. Validate Results
```python
# Always review diagnostic plots
plots = processor.generate_diagnostic_plots()

# Check processed data makes sense
print(processor.processed_data.describe())
```

## Common Use Cases

### Use Case 1: Clean GDP Data
```python
processor = EconomicDataProcessor(gdp_series, "GDP")
processor.analyze()
processor.detect_outliers(treatment='winsorize')  # Cap extremes
processor.seasonal_adjust(method='x13')  # Official method
processed_gdp = processor.processed_data
```

### Use Case 2: Prepare Data for ML
```python
processor = EconomicDataProcessor(features, "Features")
processor.detect_outliers(treatment='remove')
processor.normalize(purpose='m,')  # Z-score scaling
ml_ready_data = processor.processed_data
```

### Use Case 3: Visualize Trends
```python
processor = EconomicDataProcessor(stock_prices, "Stock")
processor.smooth(method='henderson')  # Extract smooth trend
processor.normalize(purpose='visualization')  # Scale to [0,1]
viz_data = processor.processed_data
```

### Use Case 4: Academic Research
```python
# Full pipeline with publication-quality methods
processor = EconomicDataProcessor(research_data, "Study Variable")
processed = processor.process_full_pipeline(
    detect_outliers=True,
    seasonal_adjust=True,
    smooth=True,
    outlier_treatment='remove'
)

# Generate publication report
report = processor.generate_summary_report()
# Save for paper methods section
with open('methods_appendix.md', 'w') as f:
    f.write(report)
```

## Troubleshooting

### X-13 Not Available
If X-13ARIMA-SEATS is not installed, the skill automatically falls back to STL decomposition.

### Too Few Observations
- Minimum recommended: 24 observations for seasonal adjustment
- For short series, disable seasonal adjustment or use classical method

### Heavy Outliers
If many outliers detected (>10%):
- Consider DBSCAN method for complex patterns
- Or use robust scaling instead of removal

### Non-positive Data for Log Transform
The skill automatically handles zeros/negatives by adding constants. Check the transform parameters in the report.

## References

All methods implemented follow standard econometric practices:
- X-13ARIMA-SEATS: U.S. Census Bureau
- Henderson Filters: Henderson (1916)
- STL: Cleveland et al. (1990)
- Robust Scaling: Huber (1981)

## Support

This skill is designed to handle the most common economic data processing scenarios. For specialized needs or questions, refer to the technical reference documents in the `references/` directory.