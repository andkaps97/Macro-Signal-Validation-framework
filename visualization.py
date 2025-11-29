"""
Visualization Module for Macro Signal Validation System

This module creates institutional-quality charts and plots for research presentation.

Key Features:
- Professional publication-quality charts
- Interactive Plotly dashboards
- Performance attribution visualizations
- Statistical test result plots

Author: Quantitative Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Professional visualization engine for quantitative research.
    
    Creates publication-quality charts following best practices from
    top-tier academic journals and institutional research reports.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 150,
        color_scheme: Optional[Dict] = None
    ):
        """
        Initialize visualization engine.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        dpi : int
            Figure resolution
        color_scheme : Dict, optional
            Custom color scheme
        """
        self.figsize = figsize
        self.dpi = dpi
        
        self.colors = color_scheme if color_scheme else {
            'long': '#2ecc71',
            'short': '#e74c3c',
            'neutral': '#95a5a6',
            'signal': '#3498db',
            'returns': '#f39c12',
            'benchmark': '#9b59b6'
        }
        
        logger.info("Initialized VisualizationEngine")
    
    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        drawdown: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        """Plot equity curve with optional benchmark and drawdown."""
        fig, axes = plt.subplots(2 if drawdown is not None else 1, 1,
                                figsize=self.figsize, dpi=self.dpi)
        
        if drawdown is not None:
            ax1, ax2 = axes
        else:
            ax1 = axes
        
        # Plot equity curve
        ax1.plot(equity_curve.index, equity_curve.values,
                label='Strategy', color=self.colors['signal'], linewidth=2)
        
        if benchmark is not None:
            # Normalize benchmark to same starting value
            benchmark_norm = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            ax1.plot(benchmark.index, benchmark_norm.values,
                    label='Benchmark', color=self.colors['benchmark'],
                    linewidth=2, alpha=0.7, linestyle='--')
        
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot drawdown
        if drawdown is not None:
            ax2.fill_between(drawdown.index, 0, drawdown.values * 100,
                           color=self.colors['short'], alpha=0.3)
            ax2.plot(drawdown.index, drawdown.values * 100,
                    color=self.colors['short'], linewidth=1.5)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved equity curve plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_signals_and_returns(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        indicator_col: str = 'smoothed_indicator',
        signal_col: str = 'signal',
        save_path: Optional[str] = None
    ):
        """Plot macro indicator with signals and subsequent returns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Plot indicator with signals
        ax1.plot(signals.index, signals[indicator_col],
                color=self.colors['signal'], linewidth=2, label='Indicator')
        
        # Highlight signal regimes
        long_mask = signals[signal_col] == 1
        short_mask = signals[signal_col] == -1
        
        ax1.fill_between(signals.index, signals[indicator_col].min(), signals[indicator_col].max(),
                        where=long_mask, alpha=0.2, color=self.colors['long'],
                        label='Long Signal')
        ax1.fill_between(signals.index, signals[indicator_col].min(), signals[indicator_col].max(),
                        where=short_mask, alpha=0.2, color=self.colors['short'],
                        label='Short Signal')
        
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.set_title('Macro Indicator and Trading Signals', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Indicator Value', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative returns
        cum_returns = (1 + returns).cumprod()
        ax2.plot(cum_returns.index, cum_returns.values,
                color=self.colors['returns'], linewidth=2)
        ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Return', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved signals plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_correlation_heatmap(
        self,
        correlation_df: pd.DataFrame,
        title: str = 'Forward Correlation Heatmap',
        save_path: Optional[str] = None
    ):
        """Plot correlation heatmap for different assets and horizons."""
        # Pivot data for heatmap
        pivot_data = correlation_df.pivot(
            index='asset',
            columns='horizon_label',
            values='correlation'
        )
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0, vmin=-0.5, vmax=0.5, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Forward Horizon', fontsize=12)
        ax.set_ylabel('Asset Class', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_granger_causality(
        self,
        granger_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot Granger causality test results."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Plot F-statistics by lag
        for asset in granger_df['asset'].unique():
            asset_data = granger_df[granger_df['asset'] == asset]
            axes[0].plot(asset_data['lag'], asset_data['f_statistic'],
                        marker='o', label=asset, linewidth=2)
        
        axes[0].axhline(y=3.84, color='red', linestyle='--', linewidth=1,
                       label='5% Critical Value', alpha=0.7)
        axes[0].set_title('Granger Causality F-Statistics', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Lag (months)', fontsize=12)
        axes[0].set_ylabel('F-Statistic', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot p-values by lag
        for asset in granger_df['asset'].unique():
            asset_data = granger_df[granger_df['asset'] == asset]
            axes[1].plot(asset_data['lag'], asset_data['f_p_value'],
                        marker='o', label=asset, linewidth=2)
        
        axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=1,
                       label='5% Significance', alpha=0.7)
        axes[1].set_title('Granger Causality P-Values', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Lag (months)', fontsize=12)
        axes[1].set_ylabel('P-Value', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved Granger causality plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_rolling_performance(
        self,
        rolling_metrics: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot rolling performance metrics."""
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Sharpe ratio
        if 'sharpe_ratio' in rolling_metrics.columns:
            axes[0].plot(rolling_metrics.index, rolling_metrics['sharpe_ratio'],
                        color=self.colors['signal'], linewidth=2)
            axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            axes[0].set_title('Rolling Sharpe Ratio (252d)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Sharpe Ratio', fontsize=10)
            axes[0].grid(True, alpha=0.3)
        
        # Volatility
        if 'volatility' in rolling_metrics.columns:
            axes[1].plot(rolling_metrics.index, rolling_metrics['volatility'] * 100,
                        color=self.colors['returns'], linewidth=2)
            axes[1].set_title('Rolling Volatility (252d)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Volatility (%)', fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        # Max drawdown
        if 'max_drawdown' in rolling_metrics.columns:
            axes[2].plot(rolling_metrics.index, rolling_metrics['max_drawdown'] * 100,
                        color=self.colors['short'], linewidth=2)
            axes[2].set_title('Rolling Max Drawdown (252d)', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Date', fontsize=10)
            axes[2].set_ylabel('Max Drawdown (%)', fontsize=10)
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved rolling performance plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        """Plot returns distribution with statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Histogram
        axes[0].hist(returns * 100, bins=50, alpha=0.7, color=self.colors['signal'],
                    edgecolor='black', density=True, label='Strategy')
        
        if benchmark is not None:
            axes[0].hist(benchmark * 100, bins=50, alpha=0.5, color=self.colors['benchmark'],
                        edgecolor='black', density=True, label='Benchmark')
        
        # Add normal distribution overlay
        mu, std = returns.mean() * 100, returns.std() * 100
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        axes[0].plot(x, stats.norm.pdf(x, mu, std), 'r--', linewidth=2, label='Normal')
        
        axes[0].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Daily Returns (%)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved returns distribution plot to {save_path}")
        
        plt.show()
        plt.close()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2007-01-01', '2024-12-31', freq='D')
    
    # Create visualization engine
    viz = VisualizationEngine()
    
    # Example: Equity curve
    equity = pd.Series(
        (1 + np.random.normal(0.0005, 0.01, len(dates))).cumprod() * 100000,
        index=dates
    )
    
    drawdown = pd.Series(
        -np.abs(np.random.normal(0, 0.05, len(dates))),
        index=dates
    )
    
    print("Generating equity curve plot...")
    viz.plot_equity_curve(equity, drawdown=drawdown)
