"""
Main Execution Script for Macro Signal Validation System

This script orchestrates the complete macro signal validation workflow:
1. Data acquisition
2. Signal generation
3. Statistical analysis (correlations, Granger causality)
4. Backtesting
5. Performance evaluation
6. Robustness testing
7. Report generation

Author: Quantitative Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from datetime import datetime

# Import custom modules
from config import config
from data_module import DataFetcher, DataQualityChecker
from signal_processing import SignalProcessor
from analysis_module import MacroSignalAnalyzer, StatisticalTestSuite
from backtest_module import VectorizedBacktester, WalkForwardAnalyzer, StrategyOptimizer
from performance_metrics import PerformanceAnalyzer
from visualization import VisualizationEngine

warnings.filterwarnings('ignore')


# Setup logging
def setup_logging():
    """Configure logging for the entire system."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if config.logging.LOG_TO_FILE:
        logging.basicConfig(
            level=getattr(logging, config.logging.LOG_LEVEL),
            format=log_format,
            handlers=[
                logging.FileHandler(config.logging.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, config.logging.LOG_LEVEL),
            format=log_format
        )
    
    return logging.getLogger(__name__)


def create_output_directory():
    """Create output directory for results."""
    output_dir = Path(config.visualization.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def main():
    """Main execution function."""
    logger = setup_logging()
    output_dir = create_output_directory()
    
    logger.info("="*80)
    logger.info("MACRO SIGNAL VALIDATION SYSTEM")
    logger.info("Starting complete analysis pipeline...")
    logger.info("="*80)
    
    try:
        # =================================================================
        # STEP 1: DATA ACQUISITION
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("="*80)
        
        # Initialize data fetcher
        # NOTE: Replace with your actual FRED API key
        FRED_API_KEY = config.data.FRED_API_KEY
        
        if FRED_API_KEY == "YOUR_FRED_API_KEY":
            logger.warning("FRED API KEY not set. Using synthetic data for demonstration.")
            # Generate synthetic data
            dates = pd.date_range(config.data.START_DATE, config.data.END_DATE, freq='D')
            
            # Synthetic yield curve with realistic dynamics
            trend = np.linspace(1.5, -0.2, len(dates))
            seasonal = 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)
            noise = np.random.normal(0, 0.2, len(dates))
            yield_curve = pd.Series(trend + seasonal + noise, index=dates, name='Yield Curve')
            
            # Synthetic asset returns
            returns_data = {}
            for asset, ticker in config.data.ASSET_TICKERS.items():
                # Returns correlated with lagged yield curve
                lag = 21
                correlation = 0.3 if asset == 'stocks' else -0.2
                returns_data[asset] = (
                    correlation * yield_curve.shift(lag) / 100 + 
                    np.random.normal(0.0003, 0.01, len(dates))
                )
            
            returns = pd.DataFrame(returns_data, index=dates)
            macro_data = pd.DataFrame({'yield_curve': yield_curve})
            
            logger.info("Generated synthetic data for demonstration")
            
        else:
            # Real data fetching
            fetcher = DataFetcher(FRED_API_KEY)
            
            # Fetch ALL macro indicators from config
            logger.info(f"Fetching {len(config.data.MACRO_INDICATORS)} macro indicators...")
            macro_data = fetcher.fetch_multiple_macro_indicators(
                config.data.MACRO_INDICATORS,
                config.data.START_DATE,
                config.data.END_DATE
            )

            logger.info(f"Successfully fetched {len(macro_data.columns)} macro indicators")
            logger.info(f"Indicators: {', '.join(macro_data.columns)}")

            # Fetch asset prices
            asset_data = fetcher.fetch_multiple_assets(
                config.data.ASSET_TICKERS,
                config.data.START_DATE,
                config.data.END_DATE
            )

            # Align data and calculate returns
            macro_data, returns = fetcher.align_data(
                macro_data,
                asset_data,
                frequency=config.data.FREQUENCY
            )

            logger.info(f"Fetched {len(macro_data)} observations from {macro_data.index[0]} to {macro_data.index[-1]}")
            logger.info(f"Macro indicators available: {list(macro_data.columns)}")
            logger.info(f"Assets available: {list(returns.columns)}")

        # Data quality check
        checker = DataQualityChecker()

        # Check stationarity for all indicators
        logger.info("\nMacro Indicator Quality Checks:")
        for indicator_name in macro_data.columns:
            stationarity = checker.check_stationarity(macro_data[indicator_name])
            logger.info(f"  {indicator_name}: p-value = {stationarity['p_value']:.4f}, "
                       f"stationary = {stationarity['is_stationary']}")

        # Select primary indicator for signal generation
        # Default to yield_curve if available, otherwise use first indicator
        if 'yield_curve' in macro_data.columns:
            primary_indicator = 'yield_curve'
        else:
            primary_indicator = macro_data.columns[0]

        logger.info(f"\nUsing '{primary_indicator}' as primary signal indicator")

        # =================================================================
        # STEP 2: SIGNAL GENERATION
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 2: SIGNAL GENERATION")
        logger.info("="*80)

        processor = SignalProcessor(
            smoothing_window=config.signal.SMOOTHING_WINDOW,
            use_smoothing=config.signal.USE_SMOOTHING,
            zscore_window=config.signal.ZSCORE_WINDOW,
            use_zscore=config.signal.USE_ZSCORE
        )

        signals = processor.generate_yield_curve_signal(
            macro_data[primary_indicator],
            long_threshold=config.backtest.LONG_THRESHOLD,
            short_threshold=config.backtest.SHORT_THRESHOLD
        )

        logger.info(f"Generated signals from '{primary_indicator}': "
                   f"Long={(signals['signal']==1).sum()}, "
                   f"Neutral={(signals['signal']==0).sum()}, "
                   f"Short={(signals['signal']==-1).sum()}")

        # =================================================================
        # STEP 3: STATISTICAL ANALYSIS
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 3: STATISTICAL ANALYSIS")
        logger.info("="*80)

        analyzer = MacroSignalAnalyzer(
            confidence_level=config.analysis.CONFIDENCE_LEVEL
        )

        # Forward correlations for ALL indicators
        logger.info("Calculating forward correlations for all indicators...")

        all_correlations = []
        for indicator_name in macro_data.columns:
            logger.info(f"\nTesting {indicator_name}...")
            corr_results = analyzer.calculate_forward_correlations(
                macro_data[indicator_name],
                returns,
                horizons=config.analysis.FORWARD_HORIZONS
            )
            corr_results['indicator'] = indicator_name
            all_correlations.append(corr_results)

        # Combine all correlation results
        forward_corr = pd.concat(all_correlations, ignore_index=True)

        logger.info("\n=== FORWARD CORRELATION SUMMARY ===")
        logger.info(f"Total tests performed: {len(forward_corr)}")
        logger.info(f"Significant correlations: {forward_corr['significant'].sum()}")
        logger.info(f"Significant after correction: {forward_corr['significant_adjusted'].sum()}")

        # Show strongest correlations
        logger.info("\nStrongest Correlations (Top 10):")
        top_corr = forward_corr.nlargest(10, 'correlation')
        for _, row in top_corr.iterrows():
            logger.info(f"  {row['indicator']:20s} -> {row['asset']:15s} @ {row['horizon_label']:4s}: "
                       f"r={row['correlation']:6.3f}, p={row['p_value']:.4f}")

        # Save results
        forward_corr.to_csv(output_dir / 'forward_correlations_all_indicators.csv', index=False)

        # Granger causality test for ALL indicators
        logger.info("\n\nTesting Granger causality for all indicators...")

        all_granger = []
        for indicator_name in macro_data.columns:
            logger.info(f"\nGranger test: {indicator_name}...")
            granger = analyzer.test_granger_causality(
                macro_data[indicator_name],
                returns,
                max_lag=config.analysis.GRANGER_MAX_LAG,
                significance=config.analysis.GRANGER_SIGNIFICANCE
            )
            if len(granger) > 0:
                granger['indicator'] = indicator_name
                all_granger.append(granger)

        if all_granger:
            granger_results = pd.concat(all_granger, ignore_index=True)

            logger.info("\n=== GRANGER CAUSALITY SUMMARY ===")
            logger.info(f"Total tests performed: {len(granger_results)}")
            logger.info(f"Significant causal relationships: {granger_results['granger_causes'].sum()}")

            # Show best results
            best_granger = granger_results[granger_results['granger_causes']].nsmallest(10, 'f_p_value')
            if len(best_granger) > 0:
                logger.info("\nStrongest Granger Causality (Top 10):")
                for _, row in best_granger.iterrows():
                    logger.info(f"  {row['indicator']:20s} -> {row['asset']:15s} @ lag {row['lag']:2d}: "
                               f"F={row['f_statistic']:6.2f}, p={row['f_p_value']:.4f}")

            granger_results.to_csv(output_dir / 'granger_causality_all_indicators.csv', index=False)
        else:
            granger_results = pd.DataFrame()
            logger.warning("No Granger causality tests succeeded")

        # =================================================================
        # STEP 4: BACKTESTING
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 4: BACKTESTING")
        logger.info("="*80)

        # Run backtest for each asset
        backtest_results = {}

        for asset in returns.columns:
            logger.info(f"\nBacktesting strategy on {asset}...")

            backtester = VectorizedBacktester(
                commission_rate=config.backtest.COMMISSION_RATE,
                slippage_bps=config.backtest.SLIPPAGE_BPS,
                use_dynamic_sizing=config.backtest.USE_DYNAMIC_SIZING,
                max_position_size=config.backtest.MAX_POSITION_SIZE,
                stop_loss=config.backtest.STOP_LOSS,
                take_profit=config.backtest.TAKE_PROFIT,
                initial_capital=100000.0
            )

            result = backtester.run_backtest(signals, returns[asset])
            backtest_results[asset] = result

            logger.info(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Total Return: {result.metrics['total_return']:.2%}")
            logger.info(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
            logger.info(f"  Win Rate: {result.metrics['win_rate']:.2%}")

        # =================================================================
        # STEP 5: PERFORMANCE EVALUATION
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 5: PERFORMANCE EVALUATION")
        logger.info("="*80)

        perf_analyzer = PerformanceAnalyzer(
            risk_free_rate=config.performance.RISK_FREE_RATE,
            annual_trading_days=config.performance.ANNUAL_TRADING_DAYS
        )

        # Generate detailed reports for each asset
        for asset, result in backtest_results.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"PERFORMANCE REPORT: {asset.upper()}")
            logger.info('='*80)

            report = perf_analyzer.generate_performance_report(
                result.returns,
                benchmark_returns=returns[asset] if asset != config.performance.BENCHMARK_TICKER else None,
                trades=result.trades
            )

            print(report)

            # Save report to file
            with open(output_dir / f'performance_report_{asset}.txt', 'w') as f:
                f.write(report)

            # Save detailed results
            result.equity_curve.to_csv(output_dir / f'equity_curve_{asset}.csv')
            result.returns.to_csv(output_dir / f'returns_{asset}.csv')
            if len(result.trades) > 0:
                result.trades.to_csv(output_dir / f'trades_{asset}.csv', index=False)

        # =================================================================
        # STEP 6: ROBUSTNESS TESTING
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 6: ROBUSTNESS TESTING")
        logger.info("="*80)

        # Split data for out-of-sample testing
        split_idx = int(len(signals) * config.robustness.TRAIN_TEST_SPLIT)

        train_signals = signals.iloc[:split_idx]
        test_signals = signals.iloc[split_idx:]

        logger.info(f"Train period: {train_signals.index[0]} to {train_signals.index[-1]}")
        logger.info(f"Test period: {test_signals.index[0]} to {test_signals.index[-1]}")

        # Test on main asset (stocks)
        main_asset = 'stocks'
        train_returns = returns[main_asset].iloc[:split_idx]
        test_returns = returns[main_asset].iloc[split_idx:]

        # Train backtest
        train_result = backtester.run_backtest(train_signals, train_returns)
        logger.info(f"\nIn-Sample Performance ({main_asset}):")
        logger.info(f"  Sharpe Ratio: {train_result.metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Total Return: {train_result.metrics['total_return']:.2%}")

        # Test backtest
        test_result = backtester.run_backtest(test_signals, test_returns)
        logger.info(f"\nOut-of-Sample Performance ({main_asset}):")
        logger.info(f"  Sharpe Ratio: {test_result.metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Total Return: {test_result.metrics['total_return']:.2%}")

        # Walk-forward analysis
        logger.info("\nRunning walk-forward analysis...")
        wf_analyzer = WalkForwardAnalyzer(
            train_window=config.robustness.WALK_FORWARD_WINDOW,
            test_window=63,
            step_size=config.robustness.WALK_FORWARD_STEP
        )

        wf_results = wf_analyzer.run_walk_forward(signals, returns[main_asset], backtester)
        logger.info(f"Walk-Forward Results:")
        logger.info(f"  Average Sharpe: {wf_results['avg_sharpe']:.2f}")
        logger.info(f"  Consistency: {wf_results['consistency']:.1%}")

        # Save walk-forward results
        wf_results['results_by_period'].to_csv(output_dir / 'walk_forward_results.csv', index=False)

        # =================================================================
        # STEP 7: VISUALIZATION
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 7: GENERATING VISUALIZATIONS")
        logger.info("="*80)

        viz = VisualizationEngine(
            figsize=config.visualization.FIGURE_SIZE,
            dpi=config.visualization.DPI
        )

        # Plot for main asset
        main_result = backtest_results[main_asset]

        logger.info("Generating equity curve plot...")
        viz.plot_equity_curve(
            main_result.equity_curve,
            benchmark=None,
            drawdown=main_result.drawdown,
            save_path=output_dir / f'equity_curve_{main_asset}.png'
        )

        logger.info("Generating signals and returns plot...")
        viz.plot_signals_and_returns(
            signals,
            returns[main_asset],
            save_path=output_dir / f'signals_returns_{main_asset}.png'
        )

        logger.info("Generating correlation heatmap...")
        # Create heatmap for primary indicator
        primary_corr = forward_corr[forward_corr['indicator'] == primary_indicator]
        viz.plot_correlation_heatmap(
            primary_corr,
            title=f'Forward Correlation: {primary_indicator}',
            save_path=output_dir / f'correlation_heatmap_{primary_indicator}.png'
        )

        if len(granger_results) > 0:
            logger.info("Generating Granger causality plot...")
            # Plot for primary indicator
            primary_granger = granger_results[granger_results['indicator'] == primary_indicator]
            if len(primary_granger) > 0:
                viz.plot_granger_causality(
                    primary_granger,
                    save_path=output_dir / f'granger_causality_{primary_indicator}.png'
                )

        logger.info("Generating returns distribution plot...")
        viz.plot_returns_distribution(
            main_result.returns,
            benchmark=returns[main_asset],
            save_path=output_dir / 'returns_distribution.png'
        )

        # Rolling performance
        logger.info("Generating rolling performance plot...")
        rolling_metrics = perf_analyzer.calculate_rolling_metrics(
            main_result.returns,
            window=252
        )
        viz.plot_rolling_performance(
            rolling_metrics,
            save_path=output_dir / 'rolling_performance.png'
        )

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)

        logger.info(f"\nAll results saved to: {output_dir.absolute()}")
        logger.info("\nKey Findings:")
        logger.info(f"1. Macro Indicators Tested: {len(macro_data.columns)}")
        logger.info(f"   - {', '.join(macro_data.columns)}")
        logger.info(f"2. Signal Statistics: {len(signals)} observations from {primary_indicator}")
        logger.info(f"3. Forward Correlations: {len(forward_corr)} tests performed across all indicators")
        logger.info(f"   - Significant: {forward_corr['significant_adjusted'].sum()} after multiple testing correction")
        logger.info(f"4. Granger Causality: {len(granger_results) if len(granger_results) > 0 else 0} tests performed")
        if len(granger_results) > 0:
            logger.info(f"   - Causal relationships detected: {granger_results['granger_causes'].sum()}")
        logger.info(f"5. Backtest Results: {len(backtest_results)} assets tested")
        logger.info(f"6. Walk-Forward Tests: {wf_results['n_windows']} windows analyzed")

        logger.info("\nBest Performing Asset:")
        best_asset = max(backtest_results.items(),
                        key=lambda x: x[1].metrics['sharpe_ratio'])
        logger.info(f"  Asset: {best_asset[0]}")
        logger.info(f"  Sharpe Ratio: {best_asset[1].metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Total Return: {best_asset[1].metrics['total_return']:.2%}")
        logger.info(f"  Max Drawdown: {best_asset[1].metrics['max_drawdown']:.2%}")

        # Indicator Performance Summary
        logger.info("\n" + "="*80)
        logger.info("MACRO INDICATOR PERFORMANCE RANKING")
        logger.info("="*80)

        # Calculate average correlation strength by indicator
        indicator_performance = []
        for indicator_name in macro_data.columns:
            ind_corr = forward_corr[forward_corr['indicator'] == indicator_name]
            avg_corr = ind_corr['correlation'].abs().mean()
            max_corr = ind_corr['correlation'].abs().max()
            sig_count = ind_corr['significant_adjusted'].sum()

            indicator_performance.append({
                'indicator': indicator_name,
                'avg_abs_correlation': avg_corr,
                'max_abs_correlation': max_corr,
                'significant_tests': sig_count,
                'total_tests': len(ind_corr)
            })

        perf_df = pd.DataFrame(indicator_performance).sort_values('avg_abs_correlation', ascending=False)

        logger.info("\nIndicators Ranked by Predictive Power:")
        for idx, row in perf_df.iterrows():
            logger.info(f"  {row['indicator']:20s}: "
                       f"Avg|r|={row['avg_abs_correlation']:.3f}, "
                       f"Max|r|={row['max_abs_correlation']:.3f}, "
                       f"Sig={row['significant_tests']}/{row['total_tests']}")

        # Save indicator performance
        perf_df.to_csv(output_dir / 'indicator_performance_ranking.csv', index=False)

        logger.info("\n" + "="*80)
        logger.info("SUCCESS: Macro Signal Validation Complete!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()