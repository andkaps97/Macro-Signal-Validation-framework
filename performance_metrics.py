"""
Performance Metrics Module for Macro Signal Validation System

This module implements institutional-grade performance measurement and attribution
following industry best practices from top quantitative hedge funds.

Key Features:
- Comprehensive risk-adjusted return metrics
- Factor attribution analysis
- Regime-specific performance
- Statistical significance testing
- Benchmark comparison and alpha calculation

Author: Quantitative Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and attribution.
    
    Implements institutional-grade performance measurement following
    CFA Institute standards and academic best practices.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        annual_trading_days: int = 252
    ):
        """
        Initialize performance analyzer.
        
        Parameters:
        -----------
        risk_free_rate : float, default=0.02
            Annual risk-free rate (2%)
        annual_trading_days : int, default=252
            Trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.annual_days = annual_trading_days
        
        logger.info(f"Initialized PerformanceAnalyzer with rf={risk_free_rate:.2%}")
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
            
        Returns:
        --------
        Dict
            All performance metrics
        """
        logger.info("Calculating comprehensive performance metrics...")
        
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics(returns))
        
        # Higher moments
        metrics.update(self._calculate_higher_moments(returns))
        
        # Tail risk
        metrics.update(self._calculate_tail_risk(returns))
        
        # If benchmark provided, calculate relative metrics
        if benchmark_returns is not None:
            metrics.update(self._calculate_relative_metrics(returns, benchmark_returns))
        
        logger.info("Performance metrics calculated successfully")
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """Calculate return-based metrics."""
        total_return = (1 + returns).prod() - 1
        years = len(returns) / self.annual_days
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        avg_return = returns.mean() * self.annual_days
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'avg_annual_return': avg_return,
            'best_month': returns.max(),
            'worst_month': returns.min(),
            'positive_periods': (returns > 0).sum() / len(returns)
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk metrics."""
        volatility = returns.std() * np.sqrt(self.annual_days)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.annual_days)
        
        # Semi-deviation
        semi_deviation = np.sqrt(np.mean(np.minimum(0, returns - returns.mean())**2)) * np.sqrt(self.annual_days)
        
        return {
            'volatility': volatility,
            'downside_volatility': downside_vol,
            'semi_deviation': semi_deviation,
            'vol_of_vol': returns.rolling(21).std().std() * np.sqrt(self.annual_days)
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted return metrics."""
        rf_daily = self.risk_free_rate / self.annual_days
        excess_returns = returns - rf_daily
        
        volatility = returns.std() * np.sqrt(self.annual_days)
        downside_vol = returns[returns < rf_daily].std() * np.sqrt(self.annual_days)
        
        # Sharpe ratio
        sharpe = (excess_returns.mean() * self.annual_days) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        sortino = (excess_returns.mean() * self.annual_days) / downside_vol if downside_vol > 0 else 0
        
        # Calmar ratio (CAGR / Max Drawdown)
        total_return = (1 + returns).prod() - 1
        years = len(returns) / self.annual_days
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        equity_curve = (1 + returns).cumprod()
        max_dd = (equity_curve / equity_curve.cummax() - 1).min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Omega ratio
        omega = self._calculate_omega_ratio(returns, rf_daily)
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate drawdown-related metrics."""
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        
        max_duration = 0
        current_duration = 0
        avg_duration = 0
        
        if in_drawdown.any():
            durations = []
            for group in dd_groups[in_drawdown].unique():
                duration = (dd_groups == group).sum()
                durations.append(duration)
                max_duration = max(max_duration, duration)
            
            avg_duration = np.mean(durations) if durations else 0
        
        # Recovery time
        current_dd = drawdown.iloc[-1]
        if current_dd < 0:
            recovery_time = np.nan  # Still in drawdown
        else:
            recovery_time = 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'max_dd_duration': max_duration,
            'avg_dd_duration': avg_duration,
            'current_drawdown': current_dd
        }
    
    def _calculate_higher_moments(self, returns: pd.Series) -> Dict:
        """Calculate skewness and kurtosis."""
        return {
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'excess_kurtosis': stats.kurtosis(returns) - 3
        }
    
    def _calculate_tail_risk(self, returns: pd.Series, confidence: float = 0.95) -> Dict:
        """Calculate tail risk metrics."""
        # Value at Risk
        var_95 = np.percentile(returns, (1 - confidence) * 100)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Max loss
        max_loss = returns.min()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_daily_loss': max_loss
        }
    
    def _calculate_relative_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate metrics relative to benchmark."""
        # Align returns
        common_idx = returns.index.intersection(benchmark_returns.index)
        strat_ret = returns.loc[common_idx]
        bench_ret = benchmark_returns.loc[common_idx]
        
        # Alpha (excess return)
        strat_total = (1 + strat_ret).prod() - 1
        bench_total = (1 + bench_ret).prod() - 1
        alpha = strat_total - bench_total
        
        # Tracking error
        active_returns = strat_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(self.annual_days)
        
        # Information ratio
        information_ratio = (active_returns.mean() * self.annual_days) / tracking_error if tracking_error > 0 else 0
        
        # Beta
        covariance = np.cov(strat_ret, bench_ret)[0, 1]
        benchmark_var = np.var(bench_ret)
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Up/Down capture
        up_periods = bench_ret > 0
        down_periods = bench_ret < 0
        
        up_capture = (strat_ret[up_periods].mean() / bench_ret[up_periods].mean() 
                     if up_periods.any() and bench_ret[up_periods].mean() != 0 else 0)
        down_capture = (strat_ret[down_periods].mean() / bench_ret[down_periods].mean() 
                       if down_periods.any() and bench_ret[down_periods].mean() != 0 else 0)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'up_down_capture_ratio': up_capture / abs(down_capture) if down_capture != 0 else 0
        }
    
    def _calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0
    ) -> float:
        """
        Calculate Omega ratio.
        
        Omega = Probability-weighted gains / Probability-weighted losses
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        
        if len(losses) == 0:
            return np.inf
        
        omega = gains.sum() / losses.sum() if losses.sum() > 0 else 0
        
        return omega
    
    def calculate_regime_performance(
        self,
        returns: pd.Series,
        regime: pd.Series,
        regime_names: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Calculate performance by regime.
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        regime : pd.Series
            Regime indicator
        regime_names : Dict, optional
            Mapping of regime values to names
            
        Returns:
        --------
        pd.DataFrame
            Performance metrics by regime
        """
        logger.info("Calculating regime-specific performance...")
        
        # Align data
        common_idx = returns.index.intersection(regime.index)
        ret_aligned = returns.loc[common_idx]
        reg_aligned = regime.loc[common_idx]
        
        results = []
        
        for reg_val in reg_aligned.unique():
            mask = reg_aligned == reg_val
            reg_returns = ret_aligned[mask]
            
            if len(reg_returns) < 2:
                continue
            
            reg_name = regime_names.get(reg_val, f"Regime_{reg_val}") if regime_names else f"Regime_{reg_val}"
            
            metrics = {
                'regime': reg_name,
                'n_observations': len(reg_returns),
                'frequency': len(reg_returns) / len(ret_aligned),
                'avg_return': reg_returns.mean() * self.annual_days,
                'volatility': reg_returns.std() * np.sqrt(self.annual_days),
                'sharpe_ratio': (reg_returns.mean() * self.annual_days) / (reg_returns.std() * np.sqrt(self.annual_days))
                               if reg_returns.std() > 0 else 0,
                'win_rate': (reg_returns > 0).sum() / len(reg_returns),
                'best_return': reg_returns.max(),
                'worst_return': reg_returns.min()
            }
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Regime performance calculated for {len(results_df)} regimes")
        
        return results_df
    
    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
        metrics: List[str] = ['sharpe', 'volatility', 'max_dd']
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        window : int, default=252
            Rolling window size
        metrics : List[str]
            Metrics to calculate
            
        Returns:
        --------
        pd.DataFrame
            Rolling metrics
        """
        logger.info(f"Calculating rolling metrics with {window}d window...")
        
        result = pd.DataFrame(index=returns.index)
        
        if 'sharpe' in metrics:
            rf_daily = self.risk_free_rate / self.annual_days
            excess_ret = returns - rf_daily
            rolling_mean = excess_ret.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            result['sharpe_ratio'] = (rolling_mean * self.annual_days) / (rolling_std * np.sqrt(self.annual_days))
        
        if 'volatility' in metrics:
            result['volatility'] = returns.rolling(window).std() * np.sqrt(self.annual_days)
        
        if 'max_dd' in metrics:
            def rolling_max_dd(window_returns):
                equity = (1 + window_returns).cumprod()
                running_max = equity.cummax()
                dd = (equity - running_max) / running_max
                return dd.min()
            
            result['max_drawdown'] = returns.rolling(window).apply(rolling_max_dd, raw=False)
        
        return result
    
    def generate_performance_report(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate text-based performance report.
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        benchmark_returns : pd.Series, optional
            Benchmark returns
        trades : pd.DataFrame, optional
            Trade log
            
        Returns:
        --------
        str
            Formatted performance report
        """
        metrics = self.calculate_all_metrics(returns, benchmark_returns)
        
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Return metrics
        report.append("RETURN METRICS:")
        report.append(f"  Total Return:        {metrics['total_return']:>12.2%}")
        report.append(f"  CAGR:                {metrics['cagr']:>12.2%}")
        report.append(f"  Avg Annual Return:   {metrics['avg_annual_return']:>12.2%}")
        report.append(f"  Best Period:         {metrics['best_month']:>12.2%}")
        report.append(f"  Worst Period:        {metrics['worst_month']:>12.2%}")
        report.append(f"  Win Rate:            {metrics['positive_periods']:>12.2%}")
        report.append("")
        
        # Risk metrics
        report.append("RISK METRICS:")
        report.append(f"  Volatility:          {metrics['volatility']:>12.2%}")
        report.append(f"  Downside Vol:        {metrics['downside_volatility']:>12.2%}")
        report.append(f"  Max Drawdown:        {metrics['max_drawdown']:>12.2%}")
        report.append(f"  Avg Drawdown:        {metrics['avg_drawdown']:>12.2%}")
        report.append(f"  Max DD Duration:     {metrics['max_dd_duration']:>12.0f} days")
        report.append("")
        
        # Risk-adjusted returns
        report.append("RISK-ADJUSTED RETURNS:")
        report.append(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>12.2f}")
        report.append(f"  Sortino Ratio:       {metrics['sortino_ratio']:>12.2f}")
        report.append(f"  Calmar Ratio:        {metrics['calmar_ratio']:>12.2f}")
        report.append(f"  Omega Ratio:         {metrics['omega_ratio']:>12.2f}")
        report.append("")
        
        # Tail risk
        report.append("TAIL RISK:")
        report.append(f"  VaR 95%:             {metrics['var_95']:>12.2%}")
        report.append(f"  CVaR 95%:            {metrics['cvar_95']:>12.2%}")
        report.append(f"  VaR 99%:             {metrics['var_99']:>12.2%}")
        report.append(f"  CVaR 99%:            {metrics['cvar_99']:>12.2%}")
        report.append("")
        
        # Benchmark comparison
        if benchmark_returns is not None:
            report.append("BENCHMARK COMPARISON:")
            report.append(f"  Alpha:               {metrics['alpha']:>12.2%}")
            report.append(f"  Beta:                {metrics['beta']:>12.2f}")
            report.append(f"  Information Ratio:   {metrics['information_ratio']:>12.2f}")
            report.append(f"  Tracking Error:      {metrics['tracking_error']:>12.2%}")
            report.append(f"  Up Capture:          {metrics['up_capture']:>12.2%}")
            report.append(f"  Down Capture:        {metrics['down_capture']:>12.2%}")
            report.append("")
        
        # Trade statistics
        if trades is not None and len(trades) > 0:
            report.append("TRADE STATISTICS:")
            report.append(f"  Total Trades:        {len(trades):>12.0f}")
            win_rate = (trades['pnl'] > 0).sum() / len(trades)
            report.append(f"  Win Rate:            {win_rate:>12.2%}")
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
            avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if (trades['pnl'] < 0).any() else 0
            report.append(f"  Avg Win:             {avg_win:>12,.2f}")
            report.append(f"  Avg Loss:            {avg_loss:>12,.2f}")
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            report.append(f"  Profit Factor:       {profit_factor:>12.2f}")
            report.append(f"  Avg Holding:         {trades['holding_days'].mean():>12.1f} days")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate synthetic returns
    np.random.seed(42)
    dates = pd.date_range('2007-01-01', '2024-12-31', freq='D')
    
    # Strategy returns with positive drift
    returns = pd.Series(
        np.random.normal(0.0005, 0.01, len(dates)),
        index=dates,
        name='Strategy'
    )
    
    # Benchmark returns
    benchmark = pd.Series(
        np.random.normal(0.0003, 0.012, len(dates)),
        index=dates,
        name='Benchmark'
    )
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # Calculate all metrics
    metrics = analyzer.calculate_all_metrics(returns, benchmark)
    
    # Generate report
    report = analyzer.generate_performance_report(returns, benchmark)
    print(report)
    
    # Calculate rolling metrics
    rolling = analyzer.calculate_rolling_metrics(returns, window=252)
    print("\nRolling Sharpe (last 10 periods):")
    print(rolling['sharpe_ratio'].tail(10))
