import pandas as pd
import ta
import logging

logger = logging.getLogger(__name__)

def calculate_rsi(data: pd.DataFrame, period: int = 14, price_column: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: DataFrame with price data
        period: RSI period (default 14)
        price_column: Column name for price data
        
    Returns:
        Series with RSI values
    """
    try:
        rsi = ta.momentum.RSIIndicator(data[price_column], window=period).rsi()
        logger.debug(f"Calculated RSI with period {period}")
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=data.index, dtype=float)

def calculate_sma(data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        data: DataFrame with price data
        period: SMA period
        price_column: Column name for price data
        
    Returns:
        Series with SMA values
    """
    try:
        sma = ta.trend.SMAIndicator(data[price_column], window=period).sma_indicator()
        logger.debug(f"Calculated SMA with period {period}")
        return sma
    except Exception as e:
        logger.error(f"Error calculating SMA: {str(e)}")
        return pd.Series(index=data.index, dtype=float)

def calculate_ema(data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data: DataFrame with price data
        period: EMA period
        price_column: Column name for price data
        
    Returns:
        Series with EMA values
    """
    try:
        ema = ta.trend.EMAIndicator(data[price_column], window=period).ema_indicator()
        logger.debug(f"Calculated EMA with period {period}")
        return ema
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return pd.Series(index=data.index, dtype=float)

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, 
                  price_column: str = 'close') -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: DataFrame with price data
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        price_column: Column name for price data
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    try:
        macd_indicator = ta.trend.MACD(data[price_column], window_fast=fast, window_slow=slow, window_sign=signal)
        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()
        
        logger.debug(f"Calculated MACD with periods {fast}/{slow}/{signal}")
        return macd_line, signal_line, histogram
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series, empty_series

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                             price_column: str = 'close') -> tuple:
    """
    Calculate Bollinger Bands
    
    Args:
        data: DataFrame with price data
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
        price_column: Column name for price data
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    try:
        bb_indicator = ta.volatility.BollingerBands(data[price_column], window=period, window_dev=std_dev)
        upper_band = bb_indicator.bollinger_hband()
        middle_band = bb_indicator.bollinger_mavg()
        lower_band = bb_indicator.bollinger_lband()
        
        logger.debug(f"Calculated Bollinger Bands with period {period} and std_dev {std_dev}")
        return upper_band, middle_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        empty_series = pd.Series(index=data.index, dtype=float)
        return empty_series, empty_series, empty_series

def calculate_volume_indicators(data: pd.DataFrame) -> dict:
    """
    Calculate volume-based indicators
    
    Args:
        data: DataFrame with price and volume data
        
    Returns:
        Dictionary with volume indicators
    """
    try:
        indicators = {}
        
        # Volume moving average
        indicators['volume_sma_20'] = ta.volume.VolumeSMAIndicator(
            data['close'], data['volume'], window=20
        ).volume_sma()
        
        # On-Balance Volume
        indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(
            data['close'], data['volume']
        ).on_balance_volume()
        
        # Volume delta (current vs previous)
        indicators['volume_delta'] = data['volume'].pct_change()
        
        logger.debug("Calculated volume indicators")
        return indicators
    except Exception as e:
        logger.error(f"Error calculating volume indicators: {str(e)}")
        return {}

def calculate_momentum_indicators(data: pd.DataFrame) -> dict:
    """
    Calculate momentum indicators
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Dictionary with momentum indicators
    """
    try:
        indicators = {}
        
        # Rate of Change
        indicators['roc_10'] = ta.momentum.ROCIndicator(data['close'], window=10).roc()
        
        # Williams %R
        indicators['williams_r'] = ta.momentum.WilliamsRIndicator(
            data['high'], data['low'], data['close'], lbp=14
        ).williams_r()
        
        # Stochastic Oscillator
        stoch_indicator = ta.momentum.StochasticOscillator(
            data['high'], data['low'], data['close'], window=14, smooth_window=3
        )
        indicators['stoch_k'] = stoch_indicator.stoch()
        indicators['stoch_d'] = stoch_indicator.stoch_signal()
        
        # Price momentum (rate of change)
        indicators['momentum'] = data['close'].pct_change(periods=10)
        
        logger.debug("Calculated momentum indicators")
        return indicators
    except Exception as e:
        logger.error(f"Error calculating momentum indicators: {str(e)}")
        return {}

def calculate_all_indicators(data: pd.DataFrame, rsi_period: int = 14, 
                           sma_short: int = 20, sma_long: int = 50) -> pd.DataFrame:
    """
    Calculate all technical indicators for the strategy
    
    Args:
        data: DataFrame with OHLCV data
        rsi_period: RSI period
        sma_short: Short SMA period
        sma_long: Long SMA period
        
    Returns:
        DataFrame with all indicators added
    """
    df = data.copy()
    
    try:
        # Basic indicators
        df['rsi'] = calculate_rsi(df, rsi_period)
        df['sma_20'] = calculate_sma(df, sma_short)
        df['sma_50'] = calculate_sma(df, sma_long)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(df)
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Volume indicators
        volume_indicators = calculate_volume_indicators(df)
        for key, value in volume_indicators.items():
            df[key] = value
        
        # Momentum indicators
        momentum_indicators = calculate_momentum_indicators(df)
        for key, value in momentum_indicators.items():
            df[key] = value
        
        # Price-based features
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['close_to_high_pct'] = (df['high'] - df['close']) / df['close'] * 100
        df['close_to_low_pct'] = (df['close'] - df['low']) / df['close'] * 100
        
        # SMA slopes (for trend direction)
        df['sma_20_slope'] = df['sma_20'].pct_change(periods=5)
        df['sma_50_slope'] = df['sma_50'].pct_change(periods=5)
        
        logger.info(f"Calculated all indicators for {len(df)} data points")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df
