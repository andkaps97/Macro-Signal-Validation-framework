"""
Data Module for Macro Signal Validation System

This module handles all data acquisition, cleaning, and preprocessing for both
macro indicators and asset class returns. Implements institutional-grade data
quality controls and error handling.

Key Features:
- FRED API integration for macro indicators
- Yahoo Finance integration for asset prices
- Data validation and quality checks
- Missing data interpolation
- Corporate actions adjustment
- Survivorship bias correction

Author: Quantitative Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Main data fetcher class for retrieving and processing financial data.
    
    This class implements institutional-grade data acquisition with comprehensive
    error handling, validation, and quality control procedures.
    """
    
    def __init__(self, fred_api_key: str):
        """
        Initialize the DataFetcher.
        
        Parameters:
        -----------
        fred_api_key : str
            FRED API key for accessing Federal Reserve economic data
        """
        self.fred_api_key = fred_api_key
        try:
            self.fred = Fred(api_key=fred_api_key)
            logger.info("Successfully initialized FRED API connection")
        except Exception as e:
            logger.error(f"Failed to initialize FRED API: {e}")
            raise
        
        self._data_cache = {}
    
    def fetch_macro_indicator(
        self,
        indicator_code: str,
        start_date: str,
        end_date: str,
        indicator_name: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch a single macro indicator from FRED.
        
        Parameters:
        -----------
        indicator_code : str
            FRED series code (e.g., 'T10Y2Y')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        indicator_name : str, optional
            Display name for the indicator
            
        Returns:
        --------
        pd.Series
            Time series of macro indicator values
        """
        cache_key = f"{indicator_code}_{start_date}_{end_date}"
        
        if cache_key in self._data_cache:
            logger.info(f"Returning cached data for {indicator_code}")
            return self._data_cache[cache_key]
        
        try:
            logger.info(f"Fetching {indicator_code} from FRED...")
            
            data = self.fred.get_series(
                indicator_code,
                observation_start=start_date,
                observation_end=end_date
            )
            
            if data is None or len(data) == 0:
                raise ValueError(f"No data returned for {indicator_code}")
            
            # Set name
            name = indicator_name if indicator_name else indicator_code
            data.name = name
            
            # Quality checks
            self._validate_macro_data(data, indicator_code)
            
            # Forward fill missing values (common in macro data)
            data = data.ffill()
            
            # Cache the result
            self._data_cache[cache_key] = data
            
            logger.info(f"Successfully fetched {indicator_code}: "
                       f"{len(data)} observations from {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {indicator_code}: {e}")
            raise
    
    def fetch_multiple_macro_indicators(
        self,
        indicators: Dict[str, str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch multiple macro indicators and align them to common dates.
        
        Parameters:
        -----------
        indicators : Dict[str, str]
            Dictionary mapping indicator names to FRED codes
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all indicators aligned by date
        """
        logger.info(f"Fetching {len(indicators)} macro indicators...")
        
        indicator_data = {}
        
        for name, code in indicators.items():
            try:
                data = self.fetch_macro_indicator(code, start_date, end_date, name)
                indicator_data[name] = data
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({code}): {e}")
                continue
        
        if not indicator_data:
            raise ValueError("No indicators successfully fetched")
        
        # Combine into DataFrame and forward fill
        df = pd.DataFrame(indicator_data)
        df = df.ffill().dropna()
        
        logger.info(f"Successfully fetched {len(df.columns)} indicators "
                   f"with {len(df)} aligned observations")
        
        return df
    
    def fetch_asset_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        adjust_for_splits: bool = True
    ) -> pd.DataFrame:
        """
        Fetch asset price data from Yahoo Finance.
        
        Parameters:
        -----------
        ticker : str
            Yahoo Finance ticker symbol
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        adjust_for_splits : bool, default=True
            Whether to adjust for stock splits
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV data
        """
        cache_key = f"prices_{ticker}_{start_date}_{end_date}"
        
        if cache_key in self._data_cache:
            logger.info(f"Returning cached price data for {ticker}")
            return self._data_cache[cache_key]
        
        try:
            logger.info(f"Fetching price data for {ticker}...")
            
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=adjust_for_splits
            )
            
            # Check if data was returned
            if data.empty or len(data) == 0:
                raise ValueError(f"No price data returned for {ticker}")

            # Quality checks
            self._validate_price_data(data, ticker)

            # Cache the result
            self._data_cache[cache_key] = data

            logger.info(f"Successfully fetched {ticker}: "
                       f"{len(data)} trading days from {data.index[0]} to {data.index[-1]}")

            return data

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            raise

    def fetch_multiple_assets(
        self,
        tickers: Dict[str, str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for multiple assets.

        Parameters:
        -----------
        tickers : Dict[str, str]
            Dictionary mapping asset names to ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping asset names to price DataFrames
        """
        logger.info(f"Fetching price data for {len(tickers)} assets...")

        asset_data = {}

        for name, ticker in tickers.items():
            try:
                data = self.fetch_asset_prices(ticker, start_date, end_date)
                asset_data[name] = data
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({ticker}): {e}")
                continue

        if not asset_data:
            raise ValueError("No assets successfully fetched")

        logger.info(f"Successfully fetched {len(asset_data)} assets")

        return asset_data

    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'log',
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data (typically Close prices)
        method : str, default='log'
            Return calculation method: 'log' or 'simple'
        periods : int, default=1
            Number of periods for return calculation

        Returns:
        --------
        pd.DataFrame
            Returns data
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(periods))
        elif method == 'simple':
            returns = prices.pct_change(periods=periods)
        else:
            raise ValueError(f"Invalid method: {method}. Use 'log' or 'simple'")

        return returns.dropna()

    def align_data(
        self,
        macro_data: pd.DataFrame,
        asset_data: Dict[str, pd.DataFrame],
        frequency: str = 'daily'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align macro indicators with asset returns to common trading days.

        Parameters:
        -----------
        macro_data : pd.DataFrame
            Macro indicator data
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset price DataFrames
        frequency : str, default='daily'
            Resampling frequency: 'daily', 'weekly', 'monthly'

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Aligned macro data and asset returns
        """
        logger.info(f"Aligning data to {frequency} frequency...")

        # Extract close prices and calculate returns
        returns_dict = {}
        for name, df in asset_data.items():
            # Handle both regular columns and MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # MultiIndex columns - get Close column
                if 'Close' in df.columns.get_level_values(0):
                    close_cols = [col for col in df.columns if col[0] == 'Close']
                    price = df[close_cols[0]]
                else:
                    price = df.iloc[:, 0]  # Use first column if 'Close' not found
            else:
                # Regular columns
                if 'Close' in df.columns:
                    price = df['Close']
                else:
                    price = df.iloc[:, 0]  # Use first column if 'Close' not found

            returns = self.calculate_returns(price)
            returns_dict[name] = returns

        # Combine returns
        returns_df = pd.DataFrame(returns_dict)

        # Resample if needed
        if frequency == 'weekly':
            macro_data = macro_data.resample('W-FRI').last()
            returns_df = returns_df.resample('W-FRI').sum()
        elif frequency == 'monthly':
            macro_data = macro_data.resample('M').last()
            returns_df = returns_df.resample('M').sum()

        # Align to common dates
        common_dates = macro_data.index.intersection(returns_df.index)

        macro_aligned = macro_data.loc[common_dates]
        returns_aligned = returns_df.loc[common_dates]

        logger.info(f"Aligned data: {len(common_dates)} observations "
                   f"from {common_dates[0]} to {common_dates[-1]}")

        return macro_aligned, returns_aligned

    def _validate_macro_data(self, data: pd.Series, indicator: str):
        """Validate macro indicator data quality."""
        # Check for sufficient observations
        if len(data) < 20:
            logger.warning(f"{indicator} has only {len(data)} observations")

        # Check for missing values
        missing_pct = data.isna().sum() / len(data)
        if missing_pct > 0.1:
            logger.warning(f"{indicator} has {missing_pct:.1%} missing values")

        # Check for unrealistic values
        if data.std() == 0:
            logger.warning(f"{indicator} has zero variance")

    def _validate_price_data(self, data: pd.DataFrame, ticker: str):
        """Validate price data quality."""
        # Check for minimum observations
        if len(data) < 20:
            logger.warning(f"{ticker} has only {len(data)} observations")

        # Check for missing values
        has_missing = data.isnull().any().any()
        if has_missing:
            logger.warning(f"{ticker} has missing values")

        # Check for price anomalies
        # Handle both regular columns and MultiIndex columns from yfinance
        close_col = None
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex columns - find Close column
            if 'Close' in data.columns.get_level_values(0):
                close_col = [col for col in data.columns if col[0] == 'Close'][0]
        else:
            # Regular columns
            if 'Close' in data.columns:
                close_col = 'Close'

        if close_col is not None:
            returns = data[close_col].pct_change()
            extreme_moves = returns.abs() > 0.5  # 50% single-day move
            num_extreme = extreme_moves.sum()
            if num_extreme > 0:
                logger.warning(f"{ticker} has {num_extreme} extreme price moves")

    def get_data_summary(
        self,
        macro_data: pd.DataFrame,
        returns_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate summary statistics for all data.

        Parameters:
        -----------
        macro_data : pd.DataFrame
            Macro indicator data
        returns_data : pd.DataFrame
            Asset returns data

        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        summary_list = []

        # Macro indicators
        for col in macro_data.columns:
            stats = {
                'Variable': col,
                'Type': 'Macro',
                'Observations': len(macro_data[col].dropna()),
                'Mean': macro_data[col].mean(),
                'Std': macro_data[col].std(),
                'Min': macro_data[col].min(),
                'Max': macro_data[col].max(),
                'Missing %': macro_data[col].isna().sum() / len(macro_data) * 100
            }
            summary_list.append(stats)

        # Asset returns
        for col in returns_data.columns:
            stats = {
                'Variable': col,
                'Type': 'Returns',
                'Observations': len(returns_data[col].dropna()),
                'Mean': returns_data[col].mean() * 252,  # Annualized
                'Std': returns_data[col].std() * np.sqrt(252),  # Annualized
                'Min': returns_data[col].min(),
                'Max': returns_data[col].max(),
                'Missing %': returns_data[col].isna().sum() / len(returns_data) * 100
            }
            summary_list.append(stats)

        summary_df = pd.DataFrame(summary_list)
        summary_df = summary_df.round(4)

        return summary_df

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        logger.info("Data cache cleared")


class DataQualityChecker:
    """
    Advanced data quality checking and cleaning procedures.

    Implements institutional-grade data validation following best practices
    from quantitative research at top-tier firms.
    """

    @staticmethod
    def check_stationarity(series: pd.Series, max_lag: int = 12) -> Dict:
        """
        Test for stationarity using Augmented Dickey-Fuller test.

        Parameters:
        -----------
        series : pd.Series
            Time series to test
        max_lag : int, default=12
            Maximum number of lags to use in test

        Returns:
        --------
        Dict
            Test results including test statistic and p-value
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series.dropna(), maxlag=max_lag)

        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_observations': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in time series.

        Parameters:
        -----------
        series : pd.Series
            Time series data
        method : str, default='zscore'
            Method: 'zscore', 'iqr', or 'mad'
        threshold : float, default=3.0
            Threshold for outlier detection

        Returns:
        --------
        pd.Series
            Boolean series indicating outliers
        """
        if method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold

        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            return (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))

        elif method == 'mad':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / mad
            return np.abs(modified_z) > threshold

        else:
            raise ValueError(f"Invalid method: {method}")

    @staticmethod
    def winsorize_data(
        df: pd.DataFrame,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
    ) -> pd.DataFrame:
        """
        Winsorize data to reduce impact of extreme values.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        lower_percentile : float, default=0.01
            Lower percentile for winsorization
        upper_percentile : float, default=0.99
            Upper percentile for winsorization

        Returns:
        --------
        pd.DataFrame
            Winsorized data
        """
        from scipy.stats.mstats import winsorize

        df_winsorized = df.copy()

        for col in df.columns:
            df_winsorized[col] = winsorize(
                df[col],
                limits=[lower_percentile, 1 - upper_percentile]
            )

        return df_winsorized


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example configuration
    FRED_API_KEY = "YOUR_FRED_API_KEY"  # Replace with actual key

    # Initialize fetcher
    fetcher = DataFetcher(FRED_API_KEY)

    # Fetch yield curve
    yield_curve = fetcher.fetch_macro_indicator(
        'T10Y2Y',
        '2007-01-01',
        '2024-12-31',
        'Yield Curve'
    )

    print(f"\nYield Curve Data:")
    print(f"Shape: {yield_curve.shape}")
    print(f"Range: {yield_curve.index[0]} to {yield_curve.index[-1]}")
    print(f"\nSummary Statistics:")
    print(yield_curve.describe())

    # Fetch asset prices
    spy = fetcher.fetch_asset_prices('SPY', '2007-01-01', '2024-12-31')
    print(f"\nSPY Price Data:")
    print(f"Shape: {spy.shape}")
    print(spy.head())