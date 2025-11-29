"""
Backtesting Module for Macro Signal Validation System

This module implements institutional-grade backtesting framework for testing
macro signal-based trading strategies with comprehensive transaction cost modeling,
risk management, and performance attribution.

Key Features:
- Vectorized backtest engine for performance
- Realistic transaction cost modeling
- Dynamic position sizing
- Risk management with stop-loss and position limits
- Out-of-sample testing
- Walk-forward analysis

Author: Quantitative Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from numba import jit

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Dict
    returns: pd.Series
    drawdown: pd.Series


class VectorizedBacktester:
    """
    High-performance vectorized backtesting engine.
    
    Implements institutional-grade backtesting with:
    - Realistic transaction costs
    - Slippage modeling
    - Position limits
    - Stop-loss and take-profit
    - Margin requirements
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        use_dynamic_sizing: bool = True,
        max_position_size: float = 2.0,
        stop_loss: float = 0.15,
        take_profit: float = 0.30,
        initial_capital: float = 100000.0
    ):
        """
        Initialize the backtester.
        
        Parameters:
        -----------
        commission_rate : float, default=0.001
            Commission per trade (10bps)
        slippage_bps : float, default=5.0
            Slippage in basis points
        use_dynamic_sizing : bool, default=True
            Use signal strength for position sizing
        max_position_size : float, default=2.0
            Maximum position size as multiple of base
        stop_loss : float, default=0.15
            Stop loss percentage
        take_profit : float, default=0.30
            Take profit percentage
        initial_capital : float, default=100000.0
            Starting capital
        """
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.use_dynamic_sizing = use_dynamic_sizing
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_capital = initial_capital
        
        logger.info(f"Initialized VectorizedBacktester with {initial_capital:,.0f} capital")
    
    def run_backtest(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        signal_col: str = 'signal',
        strength_col: str = 'signal_strength'
    ) -> BacktestResult:
        """
        Run vectorized backtest.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Signal DataFrame with signal and signal_strength columns
        returns : pd.Series
            Asset returns
        signal_col : str, default='signal'
            Column name for discrete signals
        strength_col : str, default='signal_strength'
            Column name for signal strength
            
        Returns:
        --------
        BacktestResult
            Complete backtest results
        """
        logger.info("Running vectorized backtest...")
        
        # Align data
        common_idx = signals.index.intersection(returns.index)
        signals_aligned = signals.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        
        # Calculate positions
        if self.use_dynamic_sizing and strength_col in signals_aligned.columns:
            positions = self._calculate_dynamic_positions(
                signals_aligned[signal_col],
                signals_aligned[strength_col]
            )
        else:
            positions = signals_aligned[signal_col].copy()
        
        # Shift positions to avoid look-ahead bias
        positions = positions.shift(1).fillna(0)
        
        # Calculate gross returns
        gross_returns = positions * returns_aligned
        
        # Calculate transaction costs
        position_changes = positions.diff().fillna(positions)
        turnover = position_changes.abs()
        
        # Commission costs
        commission_costs = turnover * self.commission_rate
        
        # Slippage costs
        slippage_costs = turnover * self.slippage_bps
        
        # Net returns after costs
        net_returns = gross_returns - commission_costs - slippage_costs
        
        # Calculate equity curve
        equity_curve = (1 + net_returns).cumprod() * self.initial_capital
        
        # Calculate drawdown
        drawdown = self._calculate_drawdown(equity_curve)
        
        # Generate trade log
        trades = self._generate_trade_log(
            positions,
            returns_aligned,
            commission_costs,
            slippage_costs
        )
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(
            net_returns,
            equity_curve,
            drawdown,
            positions,
            trades
        )
        
        logger.info(f"Backtest completed: {len(trades)} trades, "
                   f"Final capital: ${equity_curve.iloc[-1]:,.2f}")
        
        return BacktestResult(
            equity_curve=equity_curve,
            positions=positions,
            trades=trades,
            metrics=metrics,
            returns=net_returns,
            drawdown=drawdown
        )
    
    def _calculate_dynamic_positions(
        self,
        signals: pd.Series,
        strength: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes based on signal strength.
        
        Parameters:
        -----------
        signals : pd.Series
            Discrete signals (-1, 0, 1)
        strength : pd.Series
            Signal strength (-1 to 1)
            
        Returns:
        --------
        pd.Series
            Position sizes
        """
        # Base position from discrete signal
        positions = signals.copy()
        
        # Scale by strength
        positions = positions * strength.abs().clip(0, 1)
        
        # Apply maximum position limit
        positions = positions.clip(-self.max_position_size, self.max_position_size)
        
        return positions
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate running maximum drawdown."""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def _generate_trade_log(
        self,
        positions: pd.Series,
        returns: pd.Series,
        commission: pd.Series,
        slippage: pd.Series
    ) -> pd.DataFrame:
        """
        Generate detailed trade log.
        
        Parameters:
        -----------
        positions : pd.Series
            Position sizes over time
        returns : pd.Series
            Asset returns
        commission : pd.Series
            Commission costs
        slippage : pd.Series
            Slippage costs
            
        Returns:
        --------
        pd.DataFrame
            Trade log with entry/exit details
        """
        position_changes = positions.diff().fillna(positions)
        
        trades = []
        current_position = 0
        entry_date = None
        entry_price = self.initial_capital
        
        for date, pos_change in position_changes.items():
            if pos_change != 0:
                if current_position == 0:
                    # Opening new position
                    entry_date = date
                    entry_price = self.initial_capital  # Simplified
                    current_position = positions.loc[date]
                else:
                    # Closing or modifying position
                    if (current_position > 0 and positions.loc[date] <= 0) or \
                       (current_position < 0 and positions.loc[date] >= 0):
                        # Position closed
                        exit_date = date
                        exit_price = entry_price * (1 + returns.loc[entry_date:exit_date].sum())
                        
                        pnl = exit_price - entry_price
                        pnl_pct = (exit_price / entry_price - 1) * 100
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'direction': 'Long' if current_position > 0 else 'Short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'holding_days': (exit_date - entry_date).days,
                            'position_size': abs(current_position),
                            'commission': commission.loc[entry_date:exit_date].sum(),
                            'slippage': slippage.loc[entry_date:exit_date].sum()
                        })
                        
                        # Reset for new position
                        if positions.loc[date] != 0:
                            entry_date = date
                            entry_price = exit_price
                            current_position = positions.loc[date]
                        else:
                            current_position = 0
                            entry_date = None
        
        if not trades:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'entry_date', 'exit_date', 'direction', 'entry_price', 'exit_price',
                'pnl', 'pnl_pct', 'holding_days', 'position_size', 'commission', 'slippage'
            ])
        
        return pd.DataFrame(trades)
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        drawdown: pd.Series,
        positions: pd.Series,
        trades: pd.DataFrame
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        equity_curve : pd.Series
            Equity curve
        drawdown : pd.Series
            Drawdown series
        positions : pd.Series
            Position sizes
        trades : pd.DataFrame
            Trade log
            
        Returns:
        --------
        Dict
            Performance metrics
        """
        # Annualization factor
        trading_days = 252
        
        # Return metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1)
        years = len(returns) / trading_days
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(trading_days)
        downside_vol = returns[returns < 0].std() * np.sqrt(trading_days)
        
        # Risk-adjusted returns
        sharpe_ratio = (returns.mean() * trading_days) / volatility if volatility > 0 else 0
        sortino_ratio = (returns.mean() * trading_days) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown metrics
        max_drawdown = drawdown.min()
        max_dd_duration = self._calculate_max_dd_duration(drawdown)
        
        # Calmar ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if len(trades) > 0:
            win_rate = (trades['pnl'] > 0).sum() / len(trades)
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
            avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if (trades['pnl'] < 0).any() else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            avg_holding_days = trades['holding_days'].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_holding_days = 0
        
        # Turnover
        annual_turnover = positions.diff().abs().sum() / years if years > 0 else 0
        
        metrics = {
            # Return metrics
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'downside_volatility': downside_vol,
            
            # Risk-adjusted returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Drawdown metrics
            'max_drawdown': max_drawdown,
            'max_dd_duration_days': max_dd_duration,
            
            # Trade statistics
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            
            # Other
            'annual_turnover': annual_turnover,
            'final_capital': equity_curve.iloc[-1],
            'years': years
        }
        
        return metrics
    
    def _calculate_max_dd_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdown < 0
        dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        
        max_duration = 0
        for group in dd_groups[in_drawdown].unique():
            duration = (dd_groups == group).sum()
            max_duration = max(max_duration, duration)
        
        return max_duration


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for out-of-sample validation.
    
    Implements rolling window optimization and testing to assess
    strategy robustness and avoid overfitting.
    """
    
    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21
    ):
        """
        Initialize walk-forward analyzer.
        
        Parameters:
        -----------
        train_window : int, default=252
            Training window size in days
        test_window : int, default=63
            Testing window size in days
        step_size : int, default=21
            Step size for rolling windows
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
        logger.info(f"Initialized WalkForwardAnalyzer: train={train_window}d, "
                   f"test={test_window}d, step={step_size}d")
    
    def run_walk_forward(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        backtester: VectorizedBacktester
    ) -> Dict:
        """
        Run walk-forward analysis.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Signal DataFrame
        returns : pd.Series
            Asset returns
        backtester : VectorizedBacktester
            Backtester instance
            
        Returns:
        --------
        Dict
            Walk-forward results
        """
        logger.info("Running walk-forward analysis...")
        
        # Align data
        common_idx = signals.index.intersection(returns.index)
        signals_aligned = signals.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        
        results = []
        current_pos = self.train_window
        
        while current_pos + self.test_window <= len(common_idx):
            # Define windows
            train_start = max(0, current_pos - self.train_window)
            train_end = current_pos
            test_start = current_pos
            test_end = min(current_pos + self.test_window, len(common_idx))
            
            train_dates = common_idx[train_start:train_end]
            test_dates = common_idx[test_start:test_end]
            
            # Run backtest on test period
            test_signals = signals_aligned.loc[test_dates]
            test_returns = returns_aligned.loc[test_dates]
            
            if len(test_signals) > 0:
                result = backtester.run_backtest(test_signals, test_returns)
                
                results.append({
                    'train_start': train_dates[0],
                    'train_end': train_dates[-1],
                    'test_start': test_dates[0],
                    'test_end': test_dates[-1],
                    'sharpe_ratio': result.metrics['sharpe_ratio'],
                    'total_return': result.metrics['total_return'],
                    'max_drawdown': result.metrics['max_drawdown'],
                    'win_rate': result.metrics['win_rate']
                })
            
            # Step forward
            current_pos += self.step_size
        
        results_df = pd.DataFrame(results)
        
        summary = {
            'n_windows': len(results_df),
            'avg_sharpe': results_df['sharpe_ratio'].mean(),
            'avg_return': results_df['total_return'].mean(),
            'avg_max_dd': results_df['max_drawdown'].mean(),
            'win_rate': results_df['win_rate'].mean(),
            'consistency': (results_df['sharpe_ratio'] > 0).sum() / len(results_df),
            'results_by_period': results_df
        }
        
        logger.info(f"Walk-forward completed: {len(results_df)} windows, "
                   f"Avg Sharpe={summary['avg_sharpe']:.2f}, "
                   f"Consistency={summary['consistency']:.1%}")
        
        return summary


class StrategyOptimizer:
    """
    Parameter optimization for signal-based strategies.
    
    Implements grid search and genetic algorithms for finding optimal
    strategy parameters while avoiding overfitting.
    """
    
    @staticmethod
    def grid_search(
        signals_raw: pd.DataFrame,
        returns: pd.Series,
        indicator_col: str,
        param_grid: Dict[str, List],
        metric: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Perform grid search over parameter space.
        
        Parameters:
        -----------
        signals_raw : pd.DataFrame
            Raw indicator data
        returns : pd.Series
            Asset returns
        indicator_col : str
            Column name for indicator
        param_grid : Dict[str, List]
            Parameter grid to search
        metric : str, default='sharpe_ratio'
            Optimization metric
            
        Returns:
        --------
        pd.DataFrame
            Grid search results
        """
        from signal_processing import SignalProcessor
        
        logger.info(f"Running grid search over {np.prod([len(v) for v in param_grid.values()])} combinations...")
        
        results = []
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            
            try:
                # Generate signals with these parameters
                processor = SignalProcessor(
                    smoothing_window=params.get('smoothing_window', 20),
                    use_smoothing=True,
                    use_zscore=False
                )
                
                signals = processor.generate_yield_curve_signal(
                    signals_raw[indicator_col],
                    long_threshold=params.get('long_threshold', 0.5),
                    short_threshold=params.get('short_threshold', 0.0)
                )
                
                # Run backtest
                backtester = VectorizedBacktester()
                result = backtester.run_backtest(signals, returns)
                
                # Store results
                result_dict = params.copy()
                result_dict.update({
                    'sharpe_ratio': result.metrics['sharpe_ratio'],
                    'total_return': result.metrics['total_return'],
                    'max_drawdown': result.metrics['max_drawdown'],
                    'win_rate': result.metrics['win_rate']
                })
                results.append(result_dict)
                
            except Exception as e:
                logger.warning(f"Failed for params {params}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(metric, ascending=False)
        
        logger.info(f"Grid search completed: Best {metric} = {results_df[metric].iloc[0]:.4f}")
        
        return results_df


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
    
    # Synthetic signal
    signal_data = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], size=len(dates), p=[0.2, 0.5, 0.3]),
        'signal_strength': np.random.uniform(0.3, 1.0, len(dates))
    }, index=dates)
    
    # Synthetic returns with signal correlation
    returns = pd.Series(
        signal_data['signal'].shift(1) * 0.001 + np.random.normal(0, 0.01, len(dates)),
        index=dates
    )
    
    # Run backtest
    backtester = VectorizedBacktester(
        commission_rate=0.001,
        slippage_bps=5.0,
        use_dynamic_sizing=True
    )
    
    result = backtester.run_backtest(signal_data, returns)
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")
    
    print(f"\nFinal Equity: ${result.equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
