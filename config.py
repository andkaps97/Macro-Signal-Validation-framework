"""
Configuration Module for Macro Signal Validation System

This module contains all configuration parameters for the macro signal validation
framework, following institutional best practices for parameter management.

Author: Quantitative Research Team
Date: 2025-11-27
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime


@dataclass
class DataConfig:
    """Configuration for data sources and asset classes."""
    
    # FRED Macro Indicators
    FRED_API_KEY: str = ""  # Replace with your FRED API key
    MACRO_INDICATORS: Dict[str, str] = field(default_factory=lambda: {
        'yield_curve': 'T10Y2Y',  # 10Y-2Y Treasury spread
        'ted_spread': 'TEDRATE',  # TED spread (credit risk)
        'vix': 'VIXCLS',  # VIX volatility index
        'ism_manufacturing': 'MANEMP',  # ISM Manufacturing: Employment Index (NAPM is discontinued)
        'unemployment': 'UNRATE',  # Unemployment rate
        'cpi': 'CPIAUCSL',  # Consumer Price Index
        'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth
        'fed_funds': 'FEDFUNDS',  # Federal Funds Rate
    })

    # Asset Class Tickers
    ASSET_TICKERS: Dict[str, str] = field(default_factory=lambda: {
        'stocks': 'SPY',  # S&P 500 ETF
        'bonds': 'TLT',  # 20+ Year Treasury Bond ETF
        'commodities': 'DBC',  # Invesco DB Commodity Index
        'gold': 'GLD',  # Gold ETF
        'reits': 'VNQ',  # Vanguard Real Estate ETF
        'emerging_markets': 'EEM',  # Emerging Markets ETF
    })

    # Data Parameters
    START_DATE: str = '2007-01-01'
    END_DATE: str = datetime.now().strftime('%Y-%m-%d')
    FREQUENCY: str = 'daily'  # 'daily', 'weekly', 'monthly'


@dataclass
class SignalConfig:
    """Configuration for signal generation and processing."""

    # Yield Curve Signal Thresholds
    INVERSION_THRESHOLD: float = 0.0  # Below this = inverted
    STEEP_THRESHOLD: float = 0.5  # Above this = steep curve

    # Signal Smoothing
    SMOOTHING_WINDOW: int = 20  # Moving average window for signal smoothing
    USE_SMOOTHING: bool = True

    # Z-score normalization
    ZSCORE_WINDOW: int = 252  # Rolling window for z-score calculation
    USE_ZSCORE: bool = True


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""

    # Forward Correlation Horizons (in trading days)
    FORWARD_HORIZONS: List[int] = field(default_factory=lambda: [
        21,   # 1 month
        63,   # 3 months
        126,  # 6 months
        252,  # 1 year
    ])

    # Granger Causality Test Parameters
    GRANGER_MAX_LAG: int = 12  # Maximum lag for Granger test
    GRANGER_SIGNIFICANCE: float = 0.05  # Significance level

    # Rolling Correlation Windows
    ROLLING_CORR_WINDOW: int = 252  # 1 year rolling window

    # Statistical Tests
    CONFIDENCE_LEVEL: float = 0.95
    N_BOOTSTRAP_SAMPLES: int = 1000


@dataclass
class BacktestConfig:
    """Configuration for backtesting strategy."""

    # Strategy Parameters
    LONG_THRESHOLD: float = 0.5  # Enter long when yield curve > this
    SHORT_THRESHOLD: float = 0.0  # Enter short when yield curve < this
    NEUTRAL_RANGE: Tuple[float, float] = (0.0, 0.5)  # Stay neutral in this range

    # Position Sizing
    BASE_POSITION_SIZE: float = 1.0  # Base allocation per signal
    USE_DYNAMIC_SIZING: bool = True  # Scale position by signal strength
    MAX_POSITION_SIZE: float = 2.0  # Maximum position size

    # Risk Management
    STOP_LOSS: float = 0.15  # 15% stop loss
    TAKE_PROFIT: float = 0.30  # 30% take profit
    MAX_DRAWDOWN_LIMIT: float = 0.25  # Exit if drawdown exceeds 25%

    # Transaction Costs
    COMMISSION_RATE: float = 0.001  # 10bps per trade
    SLIPPAGE_BPS: float = 5.0  # 5bps slippage

    # Rebalancing
    REBALANCE_FREQUENCY: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    MIN_HOLDING_PERIOD: int = 5  # Minimum days to hold position

    # Leverage
    USE_LEVERAGE: bool = False
    MAX_LEVERAGE: float = 1.0


@dataclass
class PerformanceConfig:
    """Configuration for performance metrics calculation."""

    # Risk-Free Rate
    RISK_FREE_RATE: float = 0.02  # 2% annual risk-free rate
    RISK_FREE_SOURCE: str = '^IRX'  # 13-week Treasury Bill

    # Performance Metrics
    ANNUAL_TRADING_DAYS: int = 252

    # Benchmark
    BENCHMARK_TICKER: str = 'SPY'

    # Attribution
    CALCULATE_FACTOR_ATTRIBUTION: bool = True
    FAMA_FRENCH_FACTORS: List[str] = field(default_factory=lambda: [
        'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'
    ])


@dataclass
class RobustnessConfig:
    """Configuration for robustness testing."""

    # Out-of-Sample Testing
    TRAIN_TEST_SPLIT: float = 0.7  # 70% train, 30% test
    WALK_FORWARD_WINDOW: int = 252  # 1 year training window
    WALK_FORWARD_STEP: int = 21  # 1 month step

    # Parameter Sensitivity
    PARAMETER_RANGES: Dict[str, List[float]] = field(default_factory=lambda: {
        'long_threshold': [0.3, 0.5, 0.7, 1.0],
        'short_threshold': [-0.5, -0.3, 0.0, 0.2],
        'smoothing_window': [5, 10, 20, 40],
    })

    # Subperiod Analysis
    CRISIS_PERIODS: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        'gfc': ('2007-07-01', '2009-06-30'),  # Global Financial Crisis
        'covid': ('2020-02-01', '2020-06-30'),  # COVID-19 Crash
        'taper_tantrum': ('2013-05-01', '2013-12-31'),  # Taper Tantrum
        'dot_com': ('2000-03-01', '2002-10-31'),  # Dot-com Bubble
    })

    # Monte Carlo Simulation
    N_MONTE_CARLO_SIMS: int = 10000
    BLOCK_BOOTSTRAP_LENGTH: int = 21  # Block length for bootstrap


@dataclass
class VisualizationConfig:
    """Configuration for visualization and reporting."""

    # Chart Style
    STYLE: str = 'seaborn-v0_8-darkgrid'
    FIGURE_SIZE: Tuple[int, int] = (14, 8)
    DPI: int = 150

    # Colors
    COLOR_LONG: str = '#2ecc71'  # Green
    COLOR_SHORT: str = '#e74c3c'  # Red
    COLOR_NEUTRAL: str = '#95a5a6'  # Gray
    COLOR_SIGNAL: str = '#3498db'  # Blue
    COLOR_RETURNS: str = '#f39c12'  # Orange

    # Export
    SAVE_FORMAT: str = 'png'
    OUTPUT_DIR: str = 'output'


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    # Logging Levels
    LOG_LEVEL: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_TO_FILE: bool = True
    LOG_FILE: str = 'macro_signal_validation.log'

    # Performance Monitoring
    TRACK_EXECUTION_TIME: bool = True
    PROFILE_CODE: bool = False

    # Alerts
    ENABLE_ALERTS: bool = False
    ALERT_THRESHOLD_DRAWDOWN: float = 0.10  # Alert if drawdown > 10%


# Global Configuration Instance
@dataclass
class Config:
    """Master configuration class combining all configuration modules."""

    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate dates
        try:
            datetime.strptime(self.data.START_DATE, '%Y-%m-%d')
            datetime.strptime(self.data.END_DATE, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")

        # Validate thresholds
        if self.backtest.LONG_THRESHOLD < self.backtest.SHORT_THRESHOLD:
            raise ValueError("LONG_THRESHOLD must be >= SHORT_THRESHOLD")

        # Validate split ratio
        if not 0 < self.robustness.TRAIN_TEST_SPLIT < 1:
            raise ValueError("TRAIN_TEST_SPLIT must be between 0 and 1")

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'signal': self.signal.__dict__,
            'analysis': self.analysis.__dict__,
            'backtest': self.backtest.__dict__,
            'performance': self.performance.__dict__,
            'robustness': self.robustness.__dict__,
            'visualization': self.visualization.__dict__,
            'logging': self.logging.__dict__,
        }


# Create default configuration instance
config = Config()