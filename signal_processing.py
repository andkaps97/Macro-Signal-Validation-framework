"""
Signal Processing Module for Macro Signal Validation System

This module implements sophisticated signal generation and processing techniques
for macro indicators, with special focus on yield curve analysis.

Key Features:
- Signal generation from yield curve data
- Multi-timeframe signal smoothing and filtering
- Regime detection using HMM and threshold methods
- Signal strength quantification
- Z-score normalization and standardization

Author: Quantitative Research Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import butter, filtfilt
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Main signal processor class for generating trading signals from macro indicators.
    
    Implements institutional-grade signal processing with multiple filtering
    techniques and regime detection capabilities.
    """
    
    def __init__(
        self,
        smoothing_window: int = 20,
        use_smoothing: bool = True,
        zscore_window: int = 252,
        use_zscore: bool = True
    ):
        """
        Initialize the SignalProcessor.
        
        Parameters:
        -----------
        smoothing_window : int, default=20
            Window size for moving average smoothing
        use_smoothing : bool, default=True
            Whether to apply smoothing to signals
        zscore_window : int, default=252
            Rolling window for z-score calculation
        use_zscore : bool, default=True
            Whether to normalize signals using z-scores
        """
        self.smoothing_window = smoothing_window
        self.use_smoothing = use_smoothing
        self.zscore_window = zscore_window
        self.use_zscore = use_zscore
        
        logger.info(f"Initialized SignalProcessor with smoothing={use_smoothing}, "
                   f"zscore={use_zscore}")
    
    def generate_yield_curve_signal(
        self,
        yield_curve: pd.Series,
        long_threshold: float = 0.5,
        short_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate trading signals from yield curve data.
        
        The yield curve (10Y-2Y spread) is a well-documented recession predictor.
        Inversions (negative spread) historically precede recessions by 6-24 months.
        
        Signal Logic:
        - Long (1): Yield curve > long_threshold (steep curve, economic expansion)
        - Neutral (0): Short_threshold < yield curve < long_threshold
        - Short (-1): Yield curve < short_threshold (inverted, recession risk)
        
        Parameters:
        -----------
        yield_curve : pd.Series
            10Y-2Y Treasury spread data
        long_threshold : float, default=0.5
            Threshold for long signal (steep curve)
        short_threshold : float, default=0.0
            Threshold for short signal (inverted curve)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with raw signal, processed signal, and signal strength
        """
        logger.info("Generating yield curve signals...")
        
        # Create signal DataFrame
        signals = pd.DataFrame(index=yield_curve.index)
        signals['raw_indicator'] = yield_curve
        
        # Apply smoothing if enabled
        if self.use_smoothing:
            signals['smoothed_indicator'] = self._smooth_signal(
                yield_curve,
                window=self.smoothing_window
            )
        else:
            signals['smoothed_indicator'] = yield_curve
        
        # Apply z-score normalization if enabled
        if self.use_zscore:
            signals['zscore'] = self._calculate_zscore(
                signals['smoothed_indicator'],
                window=self.zscore_window
            )
            signal_input = signals['zscore']
        else:
            signal_input = signals['smoothed_indicator']
        
        # Generate discrete signals
        signals['signal'] = self._generate_discrete_signal(
            signal_input,
            long_threshold,
            short_threshold
        )
        
        # Calculate signal strength (continuous)
        signals['signal_strength'] = self._calculate_signal_strength(
            signal_input,
            long_threshold,
            short_threshold
        )
        
        # Calculate signal changes
        signals['signal_change'] = signals['signal'].diff()
        
        # Add regime classification
        signals['regime'] = self._classify_regime(signals['smoothed_indicator'])
        
        logger.info(f"Generated signals: {len(signals)} observations")
        logger.info(f"Signal distribution: Long={( signals['signal']==1).sum()}, "
                   f"Neutral={(signals['signal']==0).sum()}, "
                   f"Short={(signals['signal']==-1).sum()}")
        
        return signals
    
    def _smooth_signal(
        self,
        signal: pd.Series,
        window: int,
        method: str = 'ema'
    ) -> pd.Series:
        """
        Apply smoothing to reduce noise in signals.
        
        Parameters:
        -----------
        signal : pd.Series
            Raw signal data
        window : int
            Smoothing window size
        method : str, default='ema'
            Smoothing method: 'sma', 'ema', or 'gaussian'
            
        Returns:
        --------
        pd.Series
            Smoothed signal
        """
        if method == 'sma':
            return signal.rolling(window=window, min_periods=1).mean()
        
        elif method == 'ema':
            return signal.ewm(span=window, min_periods=1).mean()
        
        elif method == 'gaussian':
            # Gaussian kernel smoothing
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(signal.fillna(method='ffill'), sigma=window/4)
            return pd.Series(smoothed, index=signal.index)
        
        else:
            raise ValueError(f"Invalid smoothing method: {method}")
    
    def _calculate_zscore(
        self,
        signal: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Calculate rolling z-score for signal normalization.
        
        Z-score normalization allows for dynamic threshold adjustment based on
        historical volatility and levels.
        
        Parameters:
        -----------
        signal : pd.Series
            Input signal
        window : int
            Rolling window for mean and std calculation
            
        Returns:
        --------
        pd.Series
            Z-score normalized signal
        """
        rolling_mean = signal.rolling(window=window, min_periods=20).mean()
        rolling_std = signal.rolling(window=window, min_periods=20).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (signal - rolling_mean) / rolling_std
        
        return zscore
    
    def _generate_discrete_signal(
        self,
        indicator: pd.Series,
        long_threshold: float,
        short_threshold: float
    ) -> pd.Series:
        """
        Generate discrete trading signals from continuous indicator.
        
        Parameters:
        -----------
        indicator : pd.Series
            Continuous indicator (possibly z-score normalized)
        long_threshold : float
            Threshold for long signal
        short_threshold : float
            Threshold for short signal
            
        Returns:
        --------
        pd.Series
            Discrete signals: 1 (long), 0 (neutral), -1 (short)
        """
        signal = pd.Series(0, index=indicator.index)
        signal[indicator > long_threshold] = 1
        signal[indicator < short_threshold] = -1
        
        return signal
    
    def _calculate_signal_strength(
        self,
        indicator: pd.Series,
        long_threshold: float,
        short_threshold: float
    ) -> pd.Series:
        """
        Calculate continuous signal strength from indicator.
        
        Signal strength is useful for position sizing and confidence assessment.
        
        Parameters:
        -----------
        indicator : pd.Series
            Continuous indicator
        long_threshold : float
            Long threshold
        short_threshold : float
            Short threshold
            
        Returns:
        --------
        pd.Series
            Signal strength ranging from -1 to 1
        """
        # Normalize to [-1, 1] range
        strength = pd.Series(index=indicator.index, dtype=float)
        
        # For values above long threshold
        long_mask = indicator > long_threshold
        if long_mask.any():
            max_val = indicator[long_mask].max()
            if max_val > long_threshold:
                strength[long_mask] = (indicator[long_mask] - long_threshold) / (max_val - long_threshold)
            else:
                strength[long_mask] = 0.5
        
        # For values below short threshold
        short_mask = indicator < short_threshold
        if short_mask.any():
            min_val = indicator[short_mask].min()
            if min_val < short_threshold:
                strength[short_mask] = (indicator[short_mask] - short_threshold) / (short_threshold - min_val)
            else:
                strength[short_mask] = -0.5
        
        # For neutral range
        neutral_mask = (indicator >= short_threshold) & (indicator <= long_threshold)
        strength[neutral_mask] = 0
        
        return strength.clip(-1, 1)
    
    def _classify_regime(
        self,
        indicator: pd.Series,
        n_regimes: int = 3
    ) -> pd.Series:
        """
        Classify market regime based on indicator value.
        
        Regimes:
        - 'steep' (2): Strong expansion signal
        - 'normal' (1): Moderate expansion
        - 'flat' (0): Neutral
        - 'inverted' (-1): Recession warning
        
        Parameters:
        -----------
        indicator : pd.Series
            Macro indicator
        n_regimes : int, default=3
            Number of regimes to identify
            
        Returns:
        --------
        pd.Series
            Regime classification
        """
        # Use quantile-based classification
        regime = pd.Series(index=indicator.index, dtype=int)
        
        q33 = indicator.quantile(0.33)
        q67 = indicator.quantile(0.67)
        
        regime[indicator < q33] = -1  # Inverted/flat
        regime[(indicator >= q33) & (indicator < q67)] = 0  # Normal
        regime[indicator >= q67] = 1  # Steep
        
        return regime
    
    def detect_regime_changes(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regime changes and transition points.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Signal DataFrame from generate_yield_curve_signal
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime change indicators
        """
        regime_changes = pd.DataFrame(index=signals.index)
        
        # Detect signal changes
        regime_changes['signal_change'] = signals['signal'].diff()
        regime_changes['regime_change'] = signals['regime'].diff()
        
        # Flag transition points
        regime_changes['to_long'] = (regime_changes['signal_change'] > 0).astype(int)
        regime_changes['to_short'] = (regime_changes['signal_change'] < 0).astype(int)
        regime_changes['to_neutral'] = (
            (signals['signal'] == 0) & (signals['signal'].shift(1) != 0)
        ).astype(int)
        
        # Calculate time in regime
        regime_changes['time_in_regime'] = self._calculate_time_in_state(signals['regime'])
        
        return regime_changes
    
    def _calculate_time_in_state(self, state: pd.Series) -> pd.Series:
        """
        Calculate how long the system has been in current state.
        
        Parameters:
        -----------
        state : pd.Series
            State indicator
            
        Returns:
        --------
        pd.Series
            Days in current state
        """
        state_changes = state != state.shift(1)
        groups = state_changes.cumsum()
        time_in_state = groups.groupby(groups).cumcount() + 1
        
        return time_in_state


class AdvancedSignalProcessor:
    """
    Advanced signal processing techniques for institutional-grade research.
    
    Implements:
    - Hidden Markov Models for regime detection
    - Kalman filtering for signal smoothing
    - Wavelets for multi-scale analysis
    - Machine learning-based signal classification
    """
    
    @staticmethod
    def apply_kalman_filter(
        signal: pd.Series,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2
    ) -> pd.Series:
        """
        Apply Kalman filter for optimal signal estimation.
        
        Kalman filtering provides optimal state estimation in the presence of noise.
        
        Parameters:
        -----------
        signal : pd.Series
            Noisy signal
        process_variance : float, default=1e-5
            Process noise variance (Q)
        measurement_variance : float, default=1e-2
            Measurement noise variance (R)
            
        Returns:
        --------
        pd.Series
            Filtered signal
        """
        from pykalman import KalmanFilter
        
        # Initialize Kalman Filter
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=signal.iloc[0],
            initial_state_covariance=1,
            observation_covariance=measurement_variance,
            transition_covariance=process_variance
        )
        
        # Apply filter
        state_means, _ = kf.filter(signal.values)
        
        return pd.Series(state_means.flatten(), index=signal.index)
    
    @staticmethod
    def detect_regimes_hmm(
        signal: pd.Series,
        n_regimes: int = 3
    ) -> Tuple[pd.Series, Dict]:
        """
        Detect regimes using Hidden Markov Model.
        
        HMM is more sophisticated than threshold-based regime detection,
        capturing state persistence and transition probabilities.
        
        Parameters:
        -----------
        signal : pd.Series
            Input signal
        n_regimes : int, default=3
            Number of hidden states/regimes
            
        Returns:
        --------
        Tuple[pd.Series, Dict]
            Regime predictions and model parameters
        """
        from hmmlearn import hmm
        
        # Prepare data
        X = signal.values.reshape(-1, 1)
        
        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model.fit(X)
        
        # Predict regimes
        regimes = model.predict(X)
        
        # Extract model parameters
        params = {
            'means': model.means_,
            'covariances': model.covars_,
            'transition_matrix': model.transmat_,
            'score': model.score(X)
        }
        
        return pd.Series(regimes, index=signal.index), params
    
    @staticmethod
    def apply_butterworth_filter(
        signal: pd.Series,
        cutoff_freq: float = 0.1,
        filter_order: int = 3,
        filter_type: str = 'low'
    ) -> pd.Series:
        """
        Apply Butterworth filter for signal smoothing.
        
        Butterworth filter provides smooth frequency response without ripples.
        
        Parameters:
        -----------
        signal : pd.Series
            Input signal
        cutoff_freq : float, default=0.1
            Cutoff frequency (normalized, 0-1)
        filter_order : int, default=3
            Filter order
        filter_type : str, default='low'
            Filter type: 'low', 'high', 'band'
            
        Returns:
        --------
        pd.Series
            Filtered signal
        """
        # Design filter
        b, a = butter(filter_order, cutoff_freq, btype=filter_type)
        
        # Apply filter
        filtered = filtfilt(b, a, signal.fillna(method='ffill'))
        
        return pd.Series(filtered, index=signal.index)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate synthetic yield curve data for testing
    np.random.seed(42)
    dates = pd.date_range('2007-01-01', '2024-12-31', freq='D')
    
    # Simulate yield curve with regime changes
    trend = np.linspace(1.5, -0.5, len(dates))
    noise = np.random.normal(0, 0.3, len(dates))
    yield_curve = pd.Series(trend + noise, index=dates, name='Yield Curve')
    
    # Initialize processor
    processor = SignalProcessor(
        smoothing_window=20,
        use_smoothing=True,
        zscore_window=252,
        use_zscore=False
    )
    
    # Generate signals
    signals = processor.generate_yield_curve_signal(
        yield_curve,
        long_threshold=0.5,
        short_threshold=0.0
    )
    
    print("\nSignal Statistics:")
    print(signals.describe())
    
    print("\nSignal Distribution:")
    print(signals['signal'].value_counts())
    
    print("\nRecent Signals:")
    print(signals.tail(10))
    
    # Detect regime changes
    regime_changes = processor.detect_regime_changes(signals)
    print("\nRegime Changes:")
    print(f"Total transitions: {regime_changes['regime_change'].abs().sum()}")
    print(f"To Long: {regime_changes['to_long'].sum()}")
    print(f"To Short: {regime_changes['to_short'].sum()}")
