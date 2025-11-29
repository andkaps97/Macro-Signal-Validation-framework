"""
Quick Start Example - Macro Signal Validation

This script demonstrates a simplified version of the macro signal validation
workflow for quick testing and learning.

Author: Quantitative Research Team
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_simple_example():
    """Run a simple example using synthetic data."""
    
    logger.info("="*80)
    logger.info("MACRO SIGNAL VALIDATION - QUICK START EXAMPLE")
    logger.info("="*80)
    
    # Generate synthetic yield curve data
    logger.info("\n1. Generating synthetic yield curve data...")
    dates = pd.date_range('2010-01-01', '2024-12-31', freq='D')
    
    # Simulate yield curve: starts steep, flattens, inverts, then steepens again
    trend = np.linspace(1.5, -0.5, len(dates) // 2)
    trend = np.concatenate([trend, np.linspace(-0.5, 1.0, len(dates) - len(dates)//2)])
    noise = np.random.normal(0, 0.2, len(dates))
    yield_curve = pd.Series(trend + noise, index=dates, name='Yield Curve (10Y-2Y)')
    
    logger.info(f"  Generated {len(yield_curve)} days of data")
    logger.info(f"  Yield curve range: {yield_curve.min():.2f} to {yield_curve.max():.2f}")
    
    # Generate synthetic stock returns (correlated with lagged yield curve)
    logger.info("\n2. Generating synthetic stock returns...")
    lag = 21  # 1-month lag
    correlation = 0.35
    
    returns = pd.Series(
        correlation * yield_curve.shift(lag) / 100 + np.random.normal(0.0003, 0.01, len(dates)),
        index=dates,
        name='Stock Returns'
    )
    
    logger.info(f"  Generated {len(returns)} days of returns")
    
    # Simple signal generation
    logger.info("\n3. Generating trading signals...")
    
    # Signal logic
    long_threshold = 0.5
    short_threshold = 0.0
    
    signal = pd.Series(0, index=yield_curve.index)
    signal[yield_curve > long_threshold] = 1   # Long when steep
    signal[yield_curve < short_threshold] = -1  # Short when inverted
    
    logger.info(f"  Long signals: {(signal == 1).sum()} ({(signal == 1).sum() / len(signal) * 100:.1f}%)")
    logger.info(f"  Short signals: {(signal == -1).sum()} ({(signal == -1).sum() / len(signal) * 100:.1f}%)")
    logger.info(f"  Neutral: {(signal == 0).sum()} ({(signal == 0).sum() / len(signal) * 100:.1f}%)")
    
    # Calculate forward correlation
    logger.info("\n4. Testing predictive power (forward correlation)...")
    
    forward_horizon = 21  # 1 month
    forward_returns = returns.shift(-forward_horizon)
    
    # Align and calculate correlation
    common_idx = yield_curve.index.intersection(forward_returns.index)
    corr = yield_curve.loc[common_idx].corr(forward_returns.loc[common_idx])
    
    logger.info(f"  Forward correlation (1-month ahead): {corr:.4f}")
    logger.info(f"  Interpretation: {'Positive' if corr > 0 else 'Negative'} relationship")
    
    # Simple backtest
    logger.info("\n5. Running simple backtest...")
    
    # Shift signal to avoid look-ahead bias
    position = signal.shift(1).fillna(0)
    
    # Calculate strategy returns
    strategy_returns = position * returns
    
    # Calculate cumulative returns
    cum_returns_strategy = (1 + strategy_returns).cumprod()
    cum_returns_buy_hold = (1 + returns).cumprod()
    
    # Performance metrics
    total_return_strategy = cum_returns_strategy.iloc[-1] - 1
    total_return_buy_hold = cum_returns_buy_hold.iloc[-1] - 1
    
    volatility_strategy = strategy_returns.std() * np.sqrt(252)
    sharpe_strategy = (strategy_returns.mean() * 252) / volatility_strategy
    
    logger.info(f"\n  STRATEGY PERFORMANCE:")
    logger.info(f"  Total Return:        {total_return_strategy:.2%}")
    logger.info(f"  Annualized Vol:      {volatility_strategy:.2%}")
    logger.info(f"  Sharpe Ratio:        {sharpe_strategy:.2f}")
    
    logger.info(f"\n  BUY & HOLD PERFORMANCE:")
    logger.info(f"  Total Return:        {total_return_buy_hold:.2%}")
    
    logger.info(f"\n  ALPHA:")
    logger.info(f"  Excess Return:       {(total_return_strategy - total_return_buy_hold):.2%}")
    
    # Calculate drawdown
    running_max = cum_returns_strategy.cummax()
    drawdown = (cum_returns_strategy - running_max) / running_max
    max_drawdown = drawdown.min()
    
    logger.info(f"  Max Drawdown:        {max_drawdown:.2%}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("CONCLUSION:")
    logger.info("="*80)
    
    logger.info(f"\nThis simple example demonstrates that:")
    logger.info(f"1. The yield curve has a correlation of {corr:.3f} with future returns")
    logger.info(f"2. A simple strategy based on this signal achieves:")
    logger.info(f"   - Sharpe Ratio: {sharpe_strategy:.2f}")
    logger.info(f"   - Total Return: {total_return_strategy:.2%}")
    logger.info(f"   - Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"\nFor the complete analysis with:")
    logger.info(f"  - Real market data")
    logger.info(f"  - Statistical significance tests")
    logger.info(f"  - Transaction costs")
    logger.info(f"  - Multiple asset classes")
    logger.info(f"  - Robustness tests")
    logger.info(f"\nRun: python main.py")
    logger.info("="*80)


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    run_simple_example()
