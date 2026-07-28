#!/usr/bin/env python3
"""
Economic Data Processing Pipeline

Main orchestration script that integrates all data cleaning, normalization,
seasonal adjustment, and smoothing capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Import local modules
from analyze_data import (
    assess_data, 
    recommend_outlier_method,
    recommend_normalization_method,
    recommend_smoothing_method,
    check_seasonality,
    recommend_seasonal_method,
    generate_report
)
from detect_outliers import detect_outliers_auto, treat_outliers
from normalize_data import normalize_auto
from seasonal_adjust import seasonal_adjust_auto
from smooth_data import smooth_auto
from economic_normalizations import EconomicNormalizer


class EconomicDataProcessor:
    """
    Comprehensive economic data processing pipeline.
    
    Automatically selects and applies appropriate methods for:
    - Outlier detection and treatment
    - Data normalization/scaling
    - Seasonal adjustment
    - Trend smoothing
    """
    
    def __init__(self, data: pd.Series, name: str = "Economic Series"):
        """
        Initialize processor with time series data.
        
        Parameters:
        -----------
        data : pd.Series
            Input time series (preferably with datetime index)
        name : str, default="Economic Series"
            Name of the series for reporting
        """
        self.original_data = data.copy()
        self.processed_data = data.copy()
        self.name = name
        
        # Storage for processing steps
        self.steps_applied = []
        self.assessment = None
        self.outlier_info = None
        self.normalization_info = None
        self.seasonal_info = None
        self.smoothing_info = None
        
    def _reassess(self):
        """Re-run data assessment on the current processed data.
        
        Called after each pipeline step (outlier removal, normalization, etc.)
        so that downstream method selection reflects the updated distribution
        rather than the original data characteristics.
        """
        data_array = self.processed_data.values
        self.assessment = assess_data(data_array, self.name)
        # Also refresh seasonality info for time series
        if isinstance(self.processed_data.index, pd.DatetimeIndex):
            self.seasonality_info = check_seasonality(self.processed_data)
    
    def analyze(self) -> Dict:
        """Run comprehensive data assessment."""
        print(f"Analyzing: {self.name}")
        print("=" * 60)
        
        # Convert to numpy for assessment
        data_array = self.processed_data.values
        
        # Run assessment
        self.assessment = assess_data(data_array, self.name)
        
        # Check for seasonality if time series
        if isinstance(self.processed_data.index, pd.DatetimeIndex):
            self.seasonality_info = check_seasonality(self.processed_data)
        else:
            self.seasonality_info = {'has_seasonality': False}
        
        print(f"\n✓ Analysis complete")
        print(f"  Observations: {self.assessment['n_obs']}")
        print(f"  Distribution: {self.assessment['distribution']}")
        print(f"  Outliers: {self.assessment['n_outliers_z3']} ({self.assessment['outlier_pct_z3']:.1f}%)")
        
        return self.assessment
    
    def detect_outliers(self, method: Optional[str] = None, treatment: str = 'remove') -> pd.Series:
        """
        Detect and optionally treat outliers.
        
        Parameters:
        -----------
        method : str, optional
            Outlier detection method (auto-select if None)
        treatment : str, default='remove'
            Treatment: 'remove', 'winsorize', 'median', 'mean', 'keep'
            
        Returns:
        --------
        processed : pd.Series
            Data after outlier treatment
        """
        print(f"\nDetecting outliers...")
        
        data_array = self.processed_data.values
        
        if method is None:
            outlier_mask, method_used, details = detect_outliers_auto(data_array, self.assessment)
        else:
            # Use specified method
            from detect_outliers import (
                detect_outliers_zscore, 
                detect_outliers_iqr,
                iterative_grubbs,
                detect_outliers_dbscan
            )
            
            if method == 'z-score':
                outlier_mask, z_scores = detect_outliers_zscore(data_array)
                details = {'z_scores': z_scores}
            elif method == 'iqr':
                outlier_mask, bounds = detect_outliers_iqr(data_array)
                details = {'bounds': bounds}
            elif method == 'grubbs':
                outlier_mask, detected_values = iterative_grubbs(data_array)
                details = {'detected_values': detected_values}
            elif method == 'dbscan':
                outlier_mask, labels = detect_outliers_dbscan(data_array)
                details = {'labels': labels}
            else:
                raise ValueError(f"Unknown method: {method}")
            
            method_used = method
            details['n_outliers'] = np.sum(outlier_mask)
            details['outlier_pct'] = np.sum(outlier_mask) / len(data_array) * 100
        
        print(f"  Method: {method_used}")
        print(f"  Outliers detected: {details['n_outliers']} ({details['outlier_pct']:.1f}%)")
        
        # Treat outliers
        if treatment != 'keep':
            treated_array = treat_outliers(data_array, outlier_mask, method=treatment)
            self.processed_data = pd.Series(treated_array, index=self.processed_data.index)
            print(f"  Treatment: {treatment}")
            self.steps_applied.append(f"outlier_detection_{method_used}_{treatment}")
        else:
            self.steps_applied.append(f"outlier_detection_{method_used}_keep")
        
        self.outlier_info = {
            'method': method_used,
            'treatment': treatment,
            'mask': outlier_mask,
            'details': details
        }
        
        # Re-assess data after outlier treatment (distribution has changed)
        if treatment != 'keep':
            self._reassess()
        
        return self.processed_data
    
    def normalize(self, method: Optional[str] = None, purpose: str = 'comparison') -> pd.Series:
        """
        Normalize or scale data.
        
        Parameters:
        -----------
        method : str, optional
            Normalization method (auto-select if None)
        purpose : str, default='comparison'
            Purpose: 'comparison', 'ml', 'visualization'
            
        Returns:
        --------
        normalized : pd.Series
            Normalized data
        """
        print(f"\nNormalizing data (purpose: {purpose})...")
        
        data_array = self.processed_data.values
        
        if method is None:
            normalized_array, method_used, details = normalize_auto(
                data_array, 
                purpose=purpose, 
                assessment=self.assessment
            )
        else:
            # Use specified method
            from normalize_data import (
                min_max_scale,
                z_score_standardize,
                robust_scale,
                log_transform
            )
            
            if method == 'min-max':
                normalized_array, scaler = min_max_scale(data_array)
                details = {'scaler': scaler}
            elif method == 'z-score':
                normalized_array, scaler = z_score_standardize(data_array)
                details = {'scaler': scaler}
            elif method == 'robust':
                normalized_array, scaler = robust_scale(data_array)
                details = {'scaler': scaler}
            elif method == 'log':
                normalized_array, params = log_transform(data_array)
                details = params
            else:
                raise ValueError(f"Unknown method: {method}")
            
            method_used = method
        
        self.processed_data = pd.Series(normalized_array, index=self.processed_data.index)
        
        print(f"  Method: {method_used}")
        print(f"  New range: [{self.processed_data.min():.4f}, {self.processed_data.max():.4f}]")
        
        self.steps_applied.append(f"normalize_{method_used}")
        self.normalization_info = {
            'method': method_used,
            'details': details
        }
        
        # Re-assess data after normalization (distribution has changed)
        self._reassess()
        
        return self.processed_data
    
    def seasonal_adjust(self, method: Optional[str] = None, period: Optional[int] = None) -> pd.Series:
        """
        Apply seasonal adjustment.
        
        Parameters:
        -----------
        method : str, optional
            Seasonal adjustment method (auto-select if None)
        period : int, optional
            Seasonal period (auto-detect if None)
            
        Returns:
        --------
        adjusted : pd.Series
            Seasonally adjusted data
        """
        if not self.seasonality_info or not self.seasonality_info['has_seasonality']:
            print("\n⚠ No significant seasonality detected - skipping seasonal adjustment")
            return self.processed_data
        
        print(f"\nApplying seasonal adjustment...")
        
        if method is None:
            adjusted, method_used, details = seasonal_adjust_auto(self.processed_data, period)
        else:
            # Use specified method
            from seasonal_adjust import (
                stl_seasonal_adjustment,
                classical_seasonal_adjustment,
                x13_seasonal_adjustment
            )
            
            if period is None:
                period = 12  # Default monthly
            
            if method == 'stl':
                adjusted, result = stl_seasonal_adjustment(self.processed_data, period=period)
                details = {'period': period}
            elif method == 'classical':
                adjusted, result = classical_seasonal_adjustment(self.processed_data, period=period)
                details = {'period': period}
            elif method == 'x13':
                adjusted, result = x13_seasonal_adjustment(self.processed_data, freq=period)
                details = {'period': period}
            else:
                raise ValueError(f"Unknown method: {method}")
            
            method_used = method
        
        self.processed_data = adjusted
        
        print(f"  Method: {method_used}")
        print(f"  Period: {details.get('period', 'N/A')}")
        
        self.steps_applied.append(f"seasonal_adjust_{method_used}")
        self.seasonal_info = {
            'method': method_used,
            'details': details
        }
        
        # Re-assess data after seasonal adjustment (seasonal component removed)
        self._reassess()
        
        return self.processed_data
    
    def smooth(self, method: Optional[str] = None) -> pd.Series:
        """
        Apply smoothing/filtering.
        
        Parameters:
        -----------
        method : str, optional
            Smoothing method (auto-select if None)
            
        Returns:
        --------
        smoothed : pd.Series
            Smoothed data
        """
        print(f"\nSmoothing data...")
        
        data_array = self.processed_data.values
        
        if method is None:
            smoothed_array, method_used, details = smooth_auto(
                data_array,
                is_time_series=isinstance(self.processed_data.index, pd.DatetimeIndex),
                assessment=self.assessment
            )
        else:
            # Use specified method
            from smooth_data import (
                henderson_filter,
                exponential_smoothing,
                simple_moving_average,
                loess_smoothing
            )
            
            if method == 'henderson':
                smoothed_array = henderson_filter(data_array, length=13)
                details = {'length': 13}
            elif method == 'exponential':
                smoothed_array = exponential_smoothing(data_array, alpha=0.3)
                details = {'alpha': 0.3}
            elif method == 'moving-average':
                smoothed_array = simple_moving_average(data_array, window=12)
                details = {'window': 12}
            elif method == 'loess':
                smoothed_array = loess_smoothing(data_array, frac=0.1)
                details = {'frac': 0.1}
            else:
                raise ValueError(f"Unknown method: {method}")
            
            method_used = method
        
        self.processed_data = pd.Series(smoothed_array, index=self.processed_data.index)
        
        print(f"  Method: {method_used}")
        
        self.steps_applied.append(f"smooth_{method_used}")
        self.smoothing_info = {
            'method': method_used,
            'details': details
        }
        
        # Re-assess data after smoothing (distribution has changed)
        self._reassess()
        
        return self.processed_data
    
    def process_full_pipeline(self, 
                             detect_outliers: bool = True,
                             normalize: bool = False,
                             seasonal_adjust: bool = True,
                             smooth: bool = True,
                             outlier_treatment: str = 'remove',
                             normalization_purpose: str = 'comparison') -> pd.Series:
        """
        Run complete processing pipeline.
        
        Parameters:
        -----------
        detect_outliers : bool, default=True
            Whether to detect/treat outliers
        normalize : bool, default=False
            Whether to normalize data
        seasonal_adjust : bool, default=True
            Whether to apply seasonal adjustment
        smooth : bool, default=True
            Whether to smooth data
        outlier_treatment : str, default='remove'
            Outlier treatment method
        normalization_purpose : str, default='comparison'
            Purpose for normalization
            
        Returns:
        --------
        processed : pd.Series
            Fully processed data
        """
        print(f"\n{'='*60}")
        print(f"FULL PIPELINE: {self.name}")
        print(f"{'='*60}")
        
        # Step 1: Analyze
        self.analyze()
        
        # Step 2: Outliers
        if detect_outliers:
            self.detect_outliers(treatment=outlier_treatment)
        
        # Step 3: Seasonal Adjustment (before normalization/smoothing)
        if seasonal_adjust and isinstance(self.processed_data.index, pd.DatetimeIndex):
            self.seasonal_adjust()
        
        # Step 4: Normalization
        if normalize:
            self.normalize(purpose=normalization_purpose)
        
        # Step 5: Smoothing
        if smooth:
            self.smooth()
        
        print(f"\n{'='*60}")
        print(f"✓ PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Steps applied: {len(self.steps_applied)}")
        for i, step in enumerate(self.steps_applied, 1):
            print(f"  {i}. {step}")
        
        return self.processed_data
    
    def generate_diagnostic_plots(self, output_dir: str = "output") -> List[str]:
        """
        Generate diagnostic plots.
        
        Returns:
        --------
        plot_paths : list
            Paths to saved plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plot_paths = []
        
        # Plot 1: Original vs Processed
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(self.original_data.index, self.original_data.values, 
                label='Original', alpha=0.7)
        ax1.set_title(f'{self.name} - Original Data')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.processed_data.index, self.processed_data.values,
                label='Processed', color='green', alpha=0.7)
        ax2.set_title(f'{self.name} - Processed Data')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path1 = output_path / f'{self.name.replace(" ", "_")}_comparison.png'
        plt.savefig(path1, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(path1))
        
        # Plot 2: Distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(self.original_data.dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('Original Distribution')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(self.processed_data.dropna(), bins=30, alpha=0.7, 
                color='green', edgecolor='black')
        ax2.set_title('Processed Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path2 = output_path / f'{self.name.replace(" ", "_")}_distributions.png'
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(path2))
        
        return plot_paths
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        
        # Generate recommendations
        outlier_rec = recommend_outlier_method(self.assessment)
        norm_rec = recommend_normalization_method(self.assessment, purpose='comparison')
        smooth_rec = recommend_smoothing_method(self.assessment)
        seasonal_rec = recommend_seasonal_method(self.seasonality_info, self.assessment['n_obs'])
        
        # Generate base report
        report = generate_report(self.assessment, outlier_rec, norm_rec, smooth_rec, seasonal_rec)
        
        # Add processing summary
        report += f"""
---

## Processing Summary

### Steps Applied
{chr(10).join(f"{i}. {step}" for i, step in enumerate(self.steps_applied, 1)) if self.steps_applied else "No processing steps applied"}

### Results
- **Original observations**: {len(self.original_data)}
- **Processed observations**: {len(self.processed_data.dropna())}
- **Data loss**: {len(self.original_data) - len(self.processed_data.dropna())} observations

### Final Data Statistics
- **Mean**: {self.processed_data.mean():.4f}
- **Std**: {self.processed_data.std():.4f}
- **Range**: [{self.processed_data.min():.4f}, {self.processed_data.max():.4f}]
"""
        
        return report


if __name__ == "__main__":
    print("Economic Data Processing Pipeline - Test")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=60, freq='MS')
    trend = np.linspace(100, 120, 60)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise = np.random.normal(0, 2, 60)
    test_series = pd.Series(trend + seasonal + noise, index=dates)
    
    # Add some outliers
    test_series.iloc[10] = 150
    test_series.iloc[40] = 70
    
    # Create processor
    processor = EconomicDataProcessor(test_series, "Test Economic Series")
    
    # Run full pipeline
    processed = processor.process_full_pipeline(
        detect_outliers=True,
        seasonal_adjust=True,
        smooth=True,
        normalize=False
    )
    
    # Generate outputs
    print("\nGenerating outputs...")
    plots = processor.generate_diagnostic_plots("test_output")
    print(f"  Plots saved: {len(plots)}")
    
    report = processor.generate_summary_report()
    print("\n" + "="*60)
    print(report)