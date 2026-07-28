#!/usr/bin/env python3
"""
Economic-Specific Normalizations Module

Provides domain-specific normalization methods commonly used in economic analysis
to make data comparable across time, regions, or entities.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict


class EconomicNormalizer:
    """
    Economic data normalization toolkit.
    
    Provides methods for common economic normalizations that make data
    comparable across different contexts.
    """
    
    def __init__(self):
        """Initialize economic normalizer."""
        self.transformations_applied = []
    
    def normalize_by_gdp(self, 
                        value: Union[pd.Series, np.ndarray, float],
                        gdp: Union[pd.Series, np.ndarray, float],
                        multiply_by: float = 100.0) -> Union[pd.Series, np.ndarray, float]:
        """
        Normalize monetary values by GDP.
        
        Common use cases:
        - Debt-to-GDP ratio
        - Deficit-to-GDP ratio
        - Trade balance as % of GDP
        - Government spending as % of GDP
        
        Parameters:
        ----------
        value : numeric
            Monetary value to normalize (e.g., debt, deficit)
        gdp : numeric
            GDP value in same units
        multiply_by : float, default=100.0
            Multiplier (100 for percentage, 1 for ratio)
            
        Returns:
        --------
        normalized : numeric
            Value as percentage/ratio of GDP
            
        Example:
        ---------
        debt_to_gdp = normalizer.normalize_by_gdp(debt, gdp, multiply_by=100)
        """
        result = (value / gdp) * multiply_by
        self.transformations_applied.append('normalize_by_gdp')
        return result
    
    def per_capita(self,
                   value: Union[pd.Series, np.ndarray, float],
                   population: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Convert to per capita (per person) basis.
        
        Common use cases:
        - GDP per capita
        - Income per capita
        - Consumption per capita
        - Debt per capita
        
        Parameters:
        -----------
        value : numeric
            Total value to normalize
        population : numeric
            Population count
            
        Returns:
        --------
        per_capita : numeric
            Value per person
            
        Example:
        --------
        gdp_per_capita = normalizer.per_capita(gdp, population)
        """
        result = value / population
        self.transformations_applied.append('per_capita')
        return result
    
    def year_over_year_growth(self,
                             data: pd.Series,
                             periods: int = 1,
                             method: str = 'percentage') -> pd.Series:
        """
        Calculate Year-over-Year (YoY) growth rate.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data (must have datetime index)
        periods : int, default=1
            Number of periods for comparison
            1 = month/quarter over same month/quarter last year
        method : str, default='percentage'
            'percentage': ((New - Old) / Old) * 100
            'log': log(New) - log(Old)
            'absolute': New - Old
            
        Returns:
        --------
        growth : pd.Series
            YoY growth rates
            
        Example:
        --------
        # For monthly data, compare to 12 months ago
        cpi_yoy = normalizer.year_over_year_growth(cpi, periods=12)
        
        # For quarterly data, compare to 4 quarters ago
        gdp_yoy = normalizer.year_over_year_growth(gdp, periods=4)
        """
        if method == 'percentage':
            growth = ((data / data.shift(periods)) - 1) * 100
        elif method == 'log':
            growth = np.log(data) - np.log(data.shift(periods))
        elif method == 'absolute':
            growth = data - data.shift(periods)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.transformations_applied.append(f'year_over_year_growth_{method}')
        return growth
    
    def real_terms(self,
                   nominal: Union[pd.Series, np.ndarray],
                   price_index: Union[pd.Series, np.ndarray],
                   base_year_index: float = 100.0) -> Union[pd.Series, np.ndarray]:
        """
        Convert nominal values to real (inflation-adjusted) values.
        
        Common price indexes:
        - CPI (Consumer Price Index)
        - PCE (Personal Consumption Expenditures)
        - GDP Deflator
        - PPI (Producer Price Index)
        
        Parameters:
        -----------
        nominal : numeric
            Nominal (current price) values
        price_index : numeric
            Price index (CPI, GDP deflator, etc.)
        base_year_index : float, default=100.0
            Index value for base year
            
        Returns:
        --------
        real : numeric
            Real (constant price) values
            
        Formula:
        --------
        Real Value = (Nominal Value / Price Index) × Base Year Index
        
        Example:
        --------
        # Adjust wages for inflation using CPI
        real_wages = normalizer.real_terms(nominal_wages, cpi, base_year_index=100)
        """
        result = (nominal / price_index) * base_year_index
        self.transformations_applied.append('real_terms')
        return result
    
    def index_to_base_period(self,
                            data: Union[pd.Series, np.ndarray],
                            base_value: Optional[float] = None,
                            base_index: float = 100.0) -> Union[pd.Series, np.ndarray]:
        """
        Convert series to index with base period = 100 (or custom value).
        
        Parameters:
        -----------
        data : numeric
            Time series data
        base_value : float, optional
            Value to use as base (if None, uses first non-NaN value)
        base_index : float, default=100.0
            Index value for base period
            
        Returns:
        --------
        indexed : numeric
            Indexed series
            
        Example:
        --------
        # Set 2020 = 100
        stock_index = normalizer.index_to_base_period(stock_prices, 
                                                       base_value=prices_2020)
        """
        if base_value is None:
            if isinstance(data, pd.Series):
                base_value = data.dropna().iloc[0]
            else:
                base_value = data[~np.isnan(data)][0]
        
        result = (data / base_value) * base_index
        self.transformations_applied.append('index_to_base_period')
        return result
    
    def purchasing_power_parity(self,
                                value: Union[pd.Series, np.ndarray, float],
                                exchange_rate: Union[pd.Series, np.ndarray, float],
                                ppp_conversion_factor: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Convert to PPP-adjusted values for cross-country comparison.
        
        Parameters:
        -----------
        value : numeric
            Value in local currency
        exchange_rate : numeric
            Market exchange rate (local currency per USD)
        ppp_conversion_factor : numeric
            PPP conversion factor from World Bank
            
        Returns:
        --------
        ppp_value : numeric
            PPP-adjusted value
            
        Example:
        --------
        # Compare GDPs across countries
        gdp_ppp = normalizer.purchasing_power_parity(gdp_local, exchange_rate, ppp_factor)
        """
        result = value * (ppp_conversion_factor / exchange_rate)
        self.transformations_applied.append('purchasing_power_parity')
        return result
    
    def percentage_of_total(self,
                           value: Union[pd.Series, np.ndarray],
                           total: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray]:
        """
        Express as percentage of total.
        
        Common use cases:
        - Component as % of aggregate
        - Market share
        - Budget allocation percentages
        
        Parameters:
        -----------
        value : numeric
            Component value
        total : numeric
            Total/aggregate value
            
        Returns:
        --------
        percentage : numeric
            Value as percentage of total
            
        Example:
        --------
        export_share = normalizer.percentage_of_total(exports, total_trade)
        """
        result = (value / total) * 100
        self.transformations_applied.append('percentage_of_total')
        return result
    
    def per_labor_force(self,
                       value: Union[pd.Series, np.ndarray, float],
                       labor_force: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Normalize by labor force size.
        
        Common use cases:
        - Output per worker
        - Jobs per labor force participant
        
        Parameters:
        -----------
        value : numeric
            Value to normalize
        labor_force : numeric
            Labor force size
            
        Returns:
        --------
        per_worker : numeric
            Value per labor force participant
            
        Example:
        --------
        gdp_per_worker = normalizer.per_labor_force(gdp, labor_force)
        """
        result = value / labor_force
        self.transformations_applied.append('per_labor_force')
        return result
    
    def cyclically_adjusted(self,
                           actual: pd.Series,
                           potential: pd.Series,
                           elasticity: float = 1.0) -> pd.Series:
        """
        Calculate cyclically adjusted (structural) component.
        
        Removes business cycle effects to reveal underlying trend.
        
        Parameters:
        -----------
        actual : pd.Series
            Actual economic variable (e.g., budget balance, revenue)
        potential : pd.Series
            Potential/trend component (from HP filter, CBO estimates, etc.)
        elasticity : float, default=1.0
            Elasticity of the variable with respect to the output gap.
            A value of 1.0 means the variable moves 1:1 with the gap.
            For budget balances, typical elasticity is ~0.5 (revenues 
            respond less than 1:1 to GDP gaps). Use the appropriate 
            elasticity for the economic variable being adjusted.
            
        Returns:
        --------
        adjusted : pd.Series
            Cyclically adjusted series
            
        Formula:
        --------
        output_gap = (actual - potential) / potential
        cyclical_component = elasticity * output_gap * potential
        adjusted = actual - cyclical_component
            
        Example:
        --------
        # Cyclically adjusted budget balance
        structural_balance = normalizer.cyclically_adjusted(actual_balance, 
                                                            potential_balance,
                                                            elasticity=0.5)
        """
        # Output gap as fraction of potential
        output_gap = (actual - potential) / potential
        
        # Cyclical component: how much of actual is due to the cycle
        cyclical_component = elasticity * output_gap * potential
        
        # Remove cyclical component to get structural/adjusted value
        adjusted = actual - cyclical_component
        
        self.transformations_applied.append('cyclically_adjusted')
        return adjusted
    
    def quarter_over_quarter_annualized(self,
                                       data: pd.Series,
                                       periods: int = 1) -> pd.Series:
        """
        Calculate quarter-over-quarter growth, annualized.
        
        Standard for reporting quarterly GDP growth in the US.
        
        Parameters:
        -----------
        data : pd.Series
            Quarterly time series
        periods : int, default=1
            Number of quarters to compare
            
        Returns:
        --------
        annualized_growth : pd.Series
            Annualized quarterly growth rate (%)
            
        Formula:
        --------
        Annualized Rate = ((Q_t / Q_{t-1})^4 - 1) × 100
        
        Example:
        --------
        gdp_growth_annualized = normalizer.quarter_over_quarter_annualized(gdp)
        """
        growth = (data / data.shift(periods)) - 1
        annualized = ((1 + growth) ** 4 - 1) * 100
        
        self.transformations_applied.append('quarter_over_quarter_annualized')
        return annualized
    
    def month_over_month(self,
                        data: pd.Series,
                        annualized: bool = False) -> pd.Series:
        """
        Calculate month-over-month growth rate.
        
        Parameters:
        -----------
        data : pd.Series
            Monthly time series
        annualized : bool, default=False
            Whether to annualize the rate
            
        Returns:
        --------
        growth : pd.Series
            Monthly growth rate (%)
            
        Example:
        --------
        cpi_mom = normalizer.month_over_month(cpi, annualized=True)
        """
        growth = ((data / data.shift(1)) - 1) * 100
        
        if annualized:
            growth = ((1 + growth/100) ** 12 - 1) * 100
            self.transformations_applied.append('month_over_month_annualized')
        else:
            self.transformations_applied.append('month_over_month')
        
        return growth
    
    def contribution_to_growth(self,
                              component: pd.Series,
                              total: pd.Series,
                              periods: int = 4) -> pd.Series:
        """
        Calculate component's contribution to total growth.
        
        Used to decompose GDP growth into contributions from consumption,
        investment, government spending, and net exports.
        
        Parameters:
        -----------
        component : pd.Series
            Component series (e.g., consumption)
        total : pd.Series
            Total series (e.g., GDP)
        periods : int, default=4
            Lookback periods (4 for quarterly YoY)
            
        Returns:
        --------
        contribution : pd.Series
            Contribution to total growth (percentage points)
            
        Example:
        --------
        consumption_contribution = normalizer.contribution_to_growth(consumption, gdp)
        """
        # Growth rate of component
        component_growth = (component - component.shift(periods)) / component.shift(periods)
        
        # Weight by component's share of total
        component_share = component.shift(periods) / total.shift(periods)
        
        # Contribution = component growth × component share × 100
        contribution = component_growth * component_share * 100
        
        self.transformations_applied.append('contribution_to_growth')
        return contribution
    
    def normalize_by_area(self,
                         value: Union[pd.Series, np.ndarray, float],
                         area_km2: float) -> Union[pd.Series, np.ndarray, float]:
        """
        Normalize by geographic area (density calculation).
        
        Parameters:
        -----------
        value : numeric
            Value to normalize (e.g., population, GDP)
        area_km2 : float
            Area in square kilometers
            
        Returns:
        --------
        density : numeric
            Value per km²
            
        Example:
        --------
        pop_density = normalizer.normalize_by_area(population, area_km2)
        """
        result = value / area_km2
        self.transformations_applied.append('normalize_by_area')
        return result
    
    def cumulative_growth_rate(self,
                              data: pd.Series,
                              start_period: Optional[str] = None) -> pd.Series:
        """
        Calculate cumulative growth from a starting period.
        
        Parameters:
        -----------
        data : pd.Series
            Time series with datetime index
        start_period : str, optional
            Starting period (if None, uses first value)
            
        Returns:
        --------
        cumulative : pd.Series
            Cumulative growth rate (%)
            
        Example:
        --------
        # Growth since start of 2020
        cumulative = normalizer.cumulative_growth_rate(stock_prices, '2020-01-01')
        """
        if start_period is None:
            base_value = data.iloc[0]
        else:
            base_value = data.loc[start_period]
        
        cumulative = ((data / base_value) - 1) * 100
        
        self.transformations_applied.append('cumulative_growth_rate')
        return cumulative
    
    def three_month_moving_average(self,
                                   data: pd.Series) -> pd.Series:
        """
        Calculate 3-month moving average (common for smoothing monthly data).
        
        Parameters:
        -----------
        data : pd.Series
            Monthly time series
            
        Returns:
        --------
        ma3 : pd.Series
            3-month moving average
            
        Example:
        --------
        smoothed_employment = normalizer.three_month_moving_average(employment)
        """
        ma3 = data.rolling(window=3, center=False).mean()
        self.transformations_applied.append('three_month_moving_average')
        return ma3


def normalize_hicp(data: pd.Series,
                   reference_year: Optional[int] = None,
                   base_index: float = 100.0) -> pd.Series:
    """
    Harmonized Index of Consumer Prices (HICP) standardization.
    
    HICP is the European equivalent of CPI, used for inflation measurement
    across EU countries. Standardizes to base year = 100.
    
    Parameters:
    -----------
    data : pd.Series
        Price index series
    reference_year : int, optional
        Year to use as base (if None, uses 2015 as standard)
    base_index : float, default=100.0
        Index value for reference year
        
    Returns:
    --------
    hicp_indexed : pd.Series
        HICP-standardized series
        
    Example:
    --------
    hicp_2015 = normalize_hicp(price_index, reference_year=2015)
    """
    if reference_year is None:
        reference_year = 2015  # Standard HICP base
    
    # Find value in reference year
    if isinstance(data.index, pd.DatetimeIndex):
        ref_values = data[data.index.year == reference_year]
        if len(ref_values) > 0:
            ref_value = ref_values.mean()
        else:
            raise ValueError(f"No data found for reference year {reference_year}")
    else:
        # Assume index is already year-based
        ref_value = data[reference_year]
    
    return (data / ref_value) * base_index


def normalize_chain_weighted(components: Dict[str, pd.Series],
                             weights: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
    """
    Calculate chain-weighted index for GDP components.
    
    Chain weighting updates the base year annually to reflect changing
    composition of the economy.
    
    Parameters:
    -----------
    components : dict
        Dictionary of component series (e.g., {'consumption': series, 'investment': series})
    weights : dict, optional
        Dictionary of weight series (if None, uses equal weights)
        
    Returns:
    --------
    chain_index : pd.Series
        Chain-weighted aggregate index
        
    Example:
    --------
    gdp_chain = normalize_chain_weighted({
        'C': consumption,
        'I': investment,
        'G': government,
        'NX': net_exports
    })
    """
    if weights is None:
        # Equal weights
        n = len(components)
        weights = {k: pd.Series(1/n, index=v.index) for k, v in components.items()}
    
    # Calculate weighted sum period by period
    result_index = components[list(components.keys())[0]].index
    chain_index = pd.Series(index=result_index, dtype=float)
    
    for i, date in enumerate(result_index):
        if i == 0:
            # Base period
            chain_index.iloc[i] = 100.0
        else:
            # Growth rate weighted by previous period shares
            prev_date = result_index[i-1]
            
            weighted_growth = 0
            for component_name, component_series in components.items():
                if date in component_series.index and prev_date in component_series.index:
                    growth = (component_series[date] / component_series[prev_date]) - 1
                    weight = weights[component_name][prev_date]
                    weighted_growth += growth * weight
            
            chain_index.iloc[i] = chain_index.iloc[i-1] * (1 + weighted_growth)
    
    return chain_index


if __name__ == "__main__":
    print("Economic Normalizations Module - Test Suite")
    print("=" * 60)
    
    # Create test data
    dates = pd.date_range('2020-01-01', periods=24, freq='Q')
    
    # Mock data
    gdp = pd.Series(np.linspace(20000, 22000, 24), index=dates)
    debt = pd.Series(np.linspace(15000, 17000, 24), index=dates)
    population = 330  # millions
    cpi = pd.Series(np.linspace(250, 270, 24), index=dates)
    
    normalizer = EconomicNormalizer()
    
    # Test normalizations
    print("\n1. Debt-to-GDP Ratio:")
    debt_gdp = normalizer.normalize_by_gdp(debt, gdp, multiply_by=100)
    print(f"   Latest: {debt_gdp.iloc[-1]:.2f}%")
    
    print("\n2. GDP Per Capita:")
    gdp_pc = normalizer.per_capita(gdp.iloc[-1], population)
    print(f"   ${gdp_pc:.2f} per person")
    
    print("\n3. Year-over-Year Growth:")
    gdp_yoy = normalizer.year_over_year_growth(gdp, periods=4)
    print(f"   Latest YoY: {gdp_yoy.iloc[-1]:.2f}%")
    
    print("\n4. Real GDP (inflation-adjusted):")
    real_gdp = normalizer.real_terms(gdp, cpi, base_year_index=100)
    print(f"   Real GDP latest: ${real_gdp.iloc[-1]:.2f}")
    
    print("\n5. Index to Base Period:")
    gdp_indexed = normalizer.index_to_base_period(gdp, base_value=gdp.iloc[0])
    print(f"   Index latest: {gdp_indexed.iloc[-1]:.2f} (base=100)")
    
    print(f"\n✓ All transformations applied: {len(normalizer.transformations_applied)}")