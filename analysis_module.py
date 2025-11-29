"""
Statistical Analysis Module for Macro Signal Validation System

This module implements comprehensive statistical analysis for testing the
predictive power of macro indicators on asset returns.

Key Features:
- Forward correlation analysis at multiple horizons
- Granger causality testing
- Rolling correlation analysis
- Lead-lag relationship detection
- Statistical significance testing with multiple comparison adjustments

Author: Quantitative Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, acf, pacf
from statsmodels.stats.multitest import multipletests
import logging

logger = logging.getLogger(__name__)


class MacroSignalAnalyzer:
    """
    Main analyzer for testing macro signal predictive power.
    
    Implements rigorous statistical testing following academic standards
    for financial econometrics research.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        logger.info(f"Initialized MacroSignalAnalyzer with {confidence_level:.1%} confidence")
    
    def calculate_forward_correlations(
        self,
        indicator: pd.Series,
        returns: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Calculate forward correlations between indicator and future returns.
        
        Forward correlation measures how well the current indicator value
        predicts returns over various future horizons.
        
        Parameters:
        -----------
        indicator : pd.Series
            Macro indicator (e.g., yield curve)
        returns : pd.DataFrame
            Asset returns for multiple asset classes
        horizons : List[int]
            List of forward-looking horizons in days
            
        Returns:
        --------
        pd.DataFrame
            Correlation coefficients, p-values, and confidence intervals
        """
        logger.info(f"Calculating forward correlations for {len(horizons)} horizons...")
        
        results = []
        
        for asset in returns.columns:
            for horizon in horizons:
                # Calculate forward returns
                forward_returns = returns[asset].shift(-horizon)
                
                # Align data
                common_idx = indicator.index.intersection(forward_returns.index)
                ind_aligned = indicator.loc[common_idx]
                ret_aligned = forward_returns.loc[common_idx]
                
                # Remove NaN
                valid_mask = ind_aligned.notna() & ret_aligned.notna()
                ind_clean = ind_aligned[valid_mask]
                ret_clean = ret_aligned[valid_mask]
                
                if len(ind_clean) < 30:
                    logger.warning(f"Insufficient data for {asset} at {horizon}d horizon")
                    continue
                
                # Calculate correlation
                corr, p_value = stats.pearsonr(ind_clean, ret_clean)
                
                # Calculate Spearman correlation (non-parametric)
                spearman_corr, spearman_p = stats.spearmanr(ind_clean, ret_clean)
                
                # Calculate confidence interval
                n = len(ind_clean)
                se = np.sqrt((1 - corr**2) / (n - 2))
                ci_lower = corr - stats.t.ppf(1 - self.alpha/2, n-2) * se
                ci_upper = corr + stats.t.ppf(1 - self.alpha/2, n-2) * se
                
                results.append({
                    'asset': asset,
                    'horizon_days': horizon,
                    'horizon_label': self._format_horizon(horizon),
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'n_observations': n
                })
        
        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction (Bonferroni)
        if len(results_df) > 0:
            _, p_adjusted, _, _ = multipletests(
                results_df['p_value'],
                alpha=self.alpha,
                method='bonferroni'
            )
            results_df['p_value_adjusted'] = p_adjusted
            results_df['significant_adjusted'] = p_adjusted < self.alpha
        
        logger.info(f"Completed forward correlation analysis: {len(results_df)} tests")
        
        return results_df
    
    def test_granger_causality(
        self,
        indicator: pd.Series,
        returns: pd.DataFrame,
        max_lag: int = 12,
        significance: float = 0.05
    ) -> pd.DataFrame:
        """
        Test Granger causality from indicator to returns.
        
        Granger causality tests whether past values of the indicator help
        predict future returns beyond what past returns already predict.
        
        H0: Indicator does NOT Granger-cause returns
        H1: Indicator Granger-causes returns
        
        Parameters:
        -----------
        indicator : pd.Series
            Macro indicator
        returns : pd.DataFrame
            Asset returns
        max_lag : int, default=12
            Maximum lag to test
        significance : float, default=0.05
            Significance level
            
        Returns:
        --------
        pd.DataFrame
            Granger causality test results for each asset and lag
        """
        logger.info(f"Testing Granger causality with max_lag={max_lag}...")
        
        results = []
        
        for asset in returns.columns:
            # Align data
            common_idx = indicator.index.intersection(returns[asset].index)
            ind_aligned = indicator.loc[common_idx]
            ret_aligned = returns[asset].loc[common_idx]
            
            # Combine into DataFrame
            data = pd.DataFrame({
                'returns': ret_aligned,
                'indicator': ind_aligned
            }).dropna()
            
            if len(data) < max_lag * 2 + 10:
                logger.warning(f"Insufficient data for Granger test on {asset}")
                continue
            
            try:
                # Run Granger causality test
                gc_results = grangercausalitytests(
                    data[['returns', 'indicator']],
                    maxlag=max_lag,
                    verbose=False
                )
                
                # Extract results for each lag
                for lag in range(1, max_lag + 1):
                    # Get F-test results (most common test)
                    f_test = gc_results[lag][0]['ssr_ftest']
                    f_stat = f_test[0]
                    p_value = f_test[1]
                    
                    # Get LR test results
                    lr_test = gc_results[lag][0]['lrtest']
                    lr_stat = lr_test[0]
                    lr_p = lr_test[1]
                    
                    results.append({
                        'asset': asset,
                        'lag': lag,
                        'f_statistic': f_stat,
                        'f_p_value': p_value,
                        'lr_statistic': lr_stat,
                        'lr_p_value': lr_p,
                        'granger_causes': p_value < significance,
                        'n_observations': len(data)
                    })
                    
            except Exception as e:
                logger.error(f"Granger test failed for {asset}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Find optimal lag (minimum p-value)
            optimal_lags = results_df.loc[results_df.groupby('asset')['f_p_value'].idxmin()]
            logger.info(f"Granger causality detected for {optimal_lags['granger_causes'].sum()} "
                       f"out of {len(optimal_lags)} assets")
        
        return results_df
    
    def calculate_rolling_correlation(
        self,
        indicator: pd.Series,
        returns: pd.Series,
        window: int = 252,
        min_periods: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation over time.
        
        Rolling correlation shows how the relationship between indicator
        and returns changes over time, useful for identifying regime shifts.
        
        Parameters:
        -----------
        indicator : pd.Series
            Macro indicator
        returns : pd.Series
            Asset returns
        window : int, default=252
            Rolling window size
        min_periods : int, default=60
            Minimum observations required
            
        Returns:
        --------
        pd.DataFrame
            Rolling correlation with confidence bands
        """
        logger.info(f"Calculating rolling correlation with {window}d window...")
        
        # Align data
        common_idx = indicator.index.intersection(returns.index)
        ind_aligned = indicator.loc[common_idx]
        ret_aligned = returns.loc[common_idx]
        
        # Calculate rolling correlation
        rolling_corr = ind_aligned.rolling(window, min_periods=min_periods).corr(ret_aligned)
        
        # Calculate rolling p-values
        def rolling_pvalue(x, y, window):
            corrs = []
            p_values = []
            for i in range(len(x)):
                if i < window - 1:
                    corrs.append(np.nan)
                    p_values.append(np.nan)
                else:
                    x_window = x.iloc[i-window+1:i+1]
                    y_window = y.iloc[i-window+1:i+1]
                    if len(x_window.dropna()) >= min_periods:
                        corr, p_val = stats.pearsonr(x_window.dropna(), y_window.dropna())
                        corrs.append(corr)
                        p_values.append(p_val)
                    else:
                        corrs.append(np.nan)
                        p_values.append(np.nan)
            return pd.Series(p_values, index=x.index)
        
        rolling_p = rolling_pvalue(ind_aligned, ret_aligned, window)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'correlation': rolling_corr,
            'p_value': rolling_p,
            'significant': rolling_p < self.alpha
        })
        
        # Add confidence bands
        se = np.sqrt((1 - result['correlation']**2) / (window - 2))
        result['ci_lower'] = result['correlation'] - 1.96 * se
        result['ci_upper'] = result['correlation'] + 1.96 * se
        
        return result
    
    def detect_lead_lag_relationship(
        self,
        indicator: pd.Series,
        returns: pd.Series,
        max_lag: int = 60
    ) -> Dict:
        """
        Detect optimal lead-lag relationship using cross-correlation.
        
        Identifies at what lag the indicator has maximum correlation with returns.
        
        Parameters:
        -----------
        indicator : pd.Series
            Macro indicator
        returns : pd.Series
            Asset returns
        max_lag : int, default=60
            Maximum lag to test
            
        Returns:
        --------
        Dict
            Lead-lag analysis results
        """
        logger.info("Detecting lead-lag relationships...")
        
        # Align and standardize data
        common_idx = indicator.index.intersection(returns.index)
        ind_std = (indicator.loc[common_idx] - indicator.loc[common_idx].mean()) / indicator.loc[common_idx].std()
        ret_std = (returns.loc[common_idx] - returns.loc[common_idx].mean()) / returns.loc[common_idx].std()
        
        # Calculate cross-correlation
        cross_corr = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            if lag < 0:
                # Indicator leads
                corr = ind_std.iloc[:lag].corr(ret_std.iloc[-lag:])
            elif lag > 0:
                # Returns lead
                corr = ind_std.iloc[lag:].corr(ret_std.iloc[:-lag])
            else:
                # Contemporaneous
                corr = ind_std.corr(ret_std)
            
            cross_corr.append(corr)
        
        cross_corr = np.array(cross_corr)
        
        # Find optimal lag
        optimal_lag_idx = np.argmax(np.abs(cross_corr))
        optimal_lag = lags[optimal_lag_idx]
        optimal_corr = cross_corr[optimal_lag_idx]
        
        result = {
            'lags': list(lags),
            'cross_correlation': cross_corr.tolist(),
            'optimal_lag': optimal_lag,
            'optimal_correlation': optimal_corr,
            'interpretation': self._interpret_lag(optimal_lag)
        }
        
        return result
    
    def _format_horizon(self, days: int) -> str:
        """Format horizon in human-readable form."""
        if days < 30:
            return f"{days}d"
        elif days < 365:
            months = days // 21
            return f"{months}m"
        else:
            years = days // 252
            return f"{years}y"
    
    def _interpret_lag(self, lag: int) -> str:
        """Interpret lead-lag relationship."""
        if lag < 0:
            return f"Indicator leads returns by {abs(lag)} days"
        elif lag > 0:
            return f"Returns lead indicator by {lag} days"
        else:
            return "Contemporaneous relationship"


class StatisticalTestSuite:
    """
    Comprehensive statistical testing suite for signal validation.
    
    Implements additional tests beyond basic correlation and Granger causality.
    """
    
    @staticmethod
    def test_predictive_regression(
        indicator: pd.Series,
        returns: pd.Series,
        horizon: int
    ) -> Dict:
        """
        Run predictive regression: returns_t+h = α + β*indicator_t + ε
        
        Parameters:
        -----------
        indicator : pd.Series
            Predictor variable
        returns : pd.Series
            Target variable
        horizon : int
            Forward horizon
            
        Returns:
        --------
        Dict
            Regression results
        """
        from statsmodels.api import OLS, add_constant
        
        # Create forward returns
        forward_returns = returns.shift(-horizon)
        
        # Align data
        common_idx = indicator.index.intersection(forward_returns.index)
        X = indicator.loc[common_idx].values.reshape(-1, 1)
        y = forward_returns.loc[common_idx].values
        
        # Remove NaN
        valid_mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Add constant
        X = add_constant(X)
        
        # Run regression
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': horizon})
        
        return {
            'alpha': model.params[0],
            'beta': model.params[1],
            'alpha_pvalue': model.pvalues[0],
            'beta_pvalue': model.pvalues[1],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'n_observations': len(y)
        }
    
    @staticmethod
    def test_information_ratio(
        signal: pd.Series,
        returns: pd.Series,
        signal_threshold: float = 0
    ) -> Dict:
        """
        Calculate information ratio of signal-based strategy.
        
        IR = mean(excess returns) / std(excess returns)
        
        Parameters:
        -----------
        signal : pd.Series
            Trading signal
        returns : pd.Series
            Asset returns
        signal_threshold : float, default=0
            Threshold for signal activation
            
        Returns:
        --------
        Dict
            Information ratio and related metrics
        """
        # Generate strategy returns
        strategy_signal = (signal > signal_threshold).astype(float)
        strategy_returns = strategy_signal.shift(1) * returns
        
        # Remove NaN
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {'ir': np.nan}
        
        # Calculate metrics
        mean_ret = strategy_returns.mean() * 252  # Annualized
        std_ret = strategy_returns.std() * np.sqrt(252)  # Annualized
        ir = mean_ret / std_ret if std_ret > 0 else 0
        
        # T-statistic for mean return
        t_stat = np.sqrt(len(strategy_returns)) * (mean_ret / std_ret) if std_ret > 0 else 0
        
        return {
            'information_ratio': ir,
            'mean_return': mean_ret,
            'volatility': std_ret,
            't_statistic': t_stat,
            'p_value': 2 * (1 - stats.t.cdf(abs(t_stat), len(strategy_returns) - 1)),
            'n_observations': len(strategy_returns)
        }
    
    @staticmethod
    def calculate_transfer_entropy(
        indicator: pd.Series,
        returns: pd.Series,
        k: int = 1,
        l: int = 1
    ) -> float:
        """
        Calculate transfer entropy from indicator to returns.
        
        Transfer entropy measures information flow between time series.
        Higher values indicate stronger predictive relationship.
        
        Parameters:
        -----------
        indicator : pd.Series
            Source time series
        returns : pd.Series
            Target time series
        k : int, default=1
            History length for target
        l : int, default=1
            History length for source
            
        Returns:
        --------
        float
            Transfer entropy value
        """
        # This is a simplified implementation
        # Full implementation would require more sophisticated probability estimation
        
        from sklearn.metrics import mutual_info_score
        
        # Discretize data
        n_bins = 10
        ind_discrete = pd.cut(indicator, bins=n_bins, labels=False)
        ret_discrete = pd.cut(returns, bins=n_bins, labels=False)
        
        # Calculate mutual information terms
        # TE(X→Y) = I(Y_t; X_{t-1:t-l} | Y_{t-1:t-k})
        
        # This is simplified - full calculation requires conditional MI
        mi = mutual_info_score(ind_discrete.dropna(), ret_discrete.dropna())
        
        return mi


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2007-01-01', '2024-12-31', freq='D')
    
    # Synthetic yield curve with predictive power
    indicator = pd.Series(
        np.sin(np.linspace(0, 8*np.pi, len(dates))) + np.random.normal(0, 0.3, len(dates)),
        index=dates,
        name='Yield Curve'
    )
    
    # Synthetic returns correlated with lagged indicator
    returns_data = {}
    for asset in ['stocks', 'bonds', 'commodities']:
        lag = np.random.randint(20, 40)
        correlation = np.random.uniform(0.2, 0.5)
        noise = np.random.normal(0, 0.01, len(dates))
        returns_data[asset] = correlation * indicator.shift(lag) + noise
    
    returns = pd.DataFrame(returns_data, index=dates)
    
    # Initialize analyzer
    analyzer = MacroSignalAnalyzer(confidence_level=0.95)
    
    # Test forward correlations
    print("\n" + "="*80)
    print("FORWARD CORRELATION ANALYSIS")
    print("="*80)
    
    forward_corr = analyzer.calculate_forward_correlations(
        indicator,
        returns,
        horizons=[21, 63, 126, 252]
    )
    print(forward_corr[['asset', 'horizon_label', 'correlation', 'p_value', 'significant_adjusted']])
    
    # Test Granger causality
    print("\n" + "="*80)
    print("GRANGER CAUSALITY TESTS")
    print("="*80)
    
    granger = analyzer.test_granger_causality(
        indicator,
        returns,
        max_lag=12
    )
    
    if len(granger) > 0:
        # Show best lag for each asset
        best_lags = granger.loc[granger.groupby('asset')['f_p_value'].idxmin()]
        print(best_lags[['asset', 'lag', 'f_statistic', 'f_p_value', 'granger_causes']])
    
    # Test lead-lag
    print("\n" + "="*80)
    print("LEAD-LAG ANALYSIS")
    print("="*80)
    
    lead_lag = analyzer.detect_lead_lag_relationship(
        indicator,
        returns['stocks'],
        max_lag=60
    )
    print(f"Optimal lag: {lead_lag['optimal_lag']} days")
    print(f"Optimal correlation: {lead_lag['optimal_correlation']:.4f}")
    print(f"Interpretation: {lead_lag['interpretation']}")
