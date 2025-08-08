import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from config import config

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Rule-based trading strategy using RSI and Moving Average crossover"""
    
    def __init__(self, rsi_oversold: float = None, rsi_overbought: float = None,
                 sma_short: int = None, sma_long: int = None):
        """
        Initialize trading strategy
        
        Args:
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            sma_short: Short moving average period
            sma_long: Long moving average period
        """
        self.rsi_oversold = rsi_oversold or config.RSI_OVERSOLD
        self.rsi_overbought = rsi_overbought or config.RSI_OVERBOUGHT
        self.sma_short = sma_short or config.SMA_SHORT
        self.sma_long = sma_long or config.SMA_LONG
        
        logger.info(f"Initialized strategy: RSI({self.rsi_oversold}/{self.rsi_overbought}), "
                   f"SMA({self.sma_short}/{self.sma_long})")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on strategy rules
        
        Args:
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with signal columns added
        """
        df = data.copy()
        
        try:
            # Initialize signal columns
            df['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
            df['signal_type'] = 'HOLD'
            df['signal_reason'] = ''
            
            # Calculate MA crossover signals
            df['ma_crossover'] = self._calculate_ma_crossover(df)
            
            # Generate buy signals
            buy_condition = (
                (df['rsi'] < self.rsi_oversold) &  # RSI oversold
                (df['ma_crossover'] == 1) &        # 20-MA crosses above 50-MA
                (df['rsi'].notna()) &
                (df['sma_20'].notna()) &
                (df['sma_50'].notna())
            )
            
            # Generate sell signals
            sell_condition = (
                (df['rsi'] > self.rsi_overbought) |  # RSI overbought OR
                (df['ma_crossover'] == -1)           # 20-MA crosses below 50-MA
            ) & (
                (df['rsi'].notna()) &
                (df['sma_20'].notna()) &
                (df['sma_50'].notna())
            )
            
            # Apply signals
            df.loc[buy_condition, 'signal'] = 1
            df.loc[buy_condition, 'signal_type'] = 'BUY'
            df.loc[buy_condition, 'signal_reason'] = self._get_buy_reason(df.loc[buy_condition])
            
            df.loc[sell_condition, 'signal'] = -1
            df.loc[sell_condition, 'signal_type'] = 'SELL'
            df.loc[sell_condition, 'signal_reason'] = self._get_sell_reason(df.loc[sell_condition])
            
            # Log signal summary
            buy_signals = (df['signal'] == 1).sum()
            sell_signals = (df['signal'] == -1).sum()
            logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return df
    
    def _calculate_ma_crossover(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate moving average crossover signals
        
        Args:
            data: DataFrame with SMA data
            
        Returns:
            Series with crossover signals (1: bullish, -1: bearish, 0: no crossover)
        """
        # Check if short MA crosses above long MA (bullish)
        bullish_cross = (
            (data['sma_20'] > data['sma_50']) &
            (data['sma_20'].shift(1) <= data['sma_50'].shift(1))
        )
        
        # Check if short MA crosses below long MA (bearish)
        bearish_cross = (
            (data['sma_20'] < data['sma_50']) &
            (data['sma_20'].shift(1) >= data['sma_50'].shift(1))
        )
        
        crossover = pd.Series(0, index=data.index)
        crossover[bullish_cross] = 1
        crossover[bearish_cross] = -1
        
        return crossover
    
    def _get_buy_reason(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy signal reasons"""
        reasons = []
        for idx, row in data.iterrows():
            reason_parts = []
            if row['rsi'] < self.rsi_oversold:
                reason_parts.append(f"RSI oversold ({row['rsi']:.1f})")
            if row['ma_crossover'] == 1:
                reason_parts.append("MA bullish crossover")
            reasons.append("; ".join(reason_parts))
        return pd.Series(reasons, index=data.index)
    
    def _get_sell_reason(self, data: pd.DataFrame) -> pd.Series:
        """Generate sell signal reasons"""
        reasons = []
        for idx, row in data.iterrows():
            reason_parts = []
            if row['rsi'] > self.rsi_overbought:
                reason_parts.append(f"RSI overbought ({row['rsi']:.1f})")
            if row['ma_crossover'] == -1:
                reason_parts.append("MA bearish crossover")
            reasons.append("; ".join(reason_parts))
        return pd.Series(reasons, index=data.index)
    
    def filter_signals(self, data: pd.DataFrame, min_gap_days: int = 5) -> pd.DataFrame:
        """
        Filter signals to avoid too frequent trading
        
        Args:
            data: DataFrame with signals
            min_gap_days: Minimum days between signals
            
        Returns:
            DataFrame with filtered signals
        """
        df = data.copy()
        
        # Get signal dates
        signal_dates = df[df['signal'] != 0].index
        
        if len(signal_dates) == 0:
            return df
        
        # Filter signals with minimum gap
        filtered_signals = [signal_dates[0]]  # Keep first signal
        
        for signal_date in signal_dates[1:]:
            # Convert to pandas Timestamp if needed for proper date arithmetic
            if hasattr(signal_date, 'to_pydatetime'):
                signal_date_conv = signal_date.to_pydatetime()
            else:
                signal_date_conv = pd.to_datetime(signal_date)
                
            if hasattr(filtered_signals[-1], 'to_pydatetime'):
                last_signal_conv = filtered_signals[-1].to_pydatetime()
            else:
                last_signal_conv = pd.to_datetime(filtered_signals[-1])
                
            days_since_last = (signal_date_conv - last_signal_conv).days
            if days_since_last >= min_gap_days:
                filtered_signals.append(signal_date)
        
        # Reset signals
        df['signal'] = 0
        df['signal_type'] = 'HOLD'
        df['signal_reason'] = ''
        
        # Restore filtered signals
        for signal_date in filtered_signals:
            original_signal = data.loc[signal_date, 'signal']
            df.loc[signal_date, 'signal'] = original_signal
            df.loc[signal_date, 'signal_type'] = data.loc[signal_date, 'signal_type']
            df.loc[signal_date, 'signal_reason'] = data.loc[signal_date, 'signal_reason']
        
        logger.info(f"Filtered signals from {len(signal_dates)} to {len(filtered_signals)}")
        return df
    
    def get_current_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get the most recent signal
        
        Args:
            data: DataFrame with signals
            
        Returns:
            Dictionary with current signal information
        """
        try:
            latest_data = data.iloc[-1]
            
            signal_info = {
                'date': latest_data.name if hasattr(latest_data.name, 'strftime') else str(latest_data.name),
                'symbol': latest_data.get('symbol', 'Unknown'),
                'price': latest_data['close'],
                'signal': latest_data['signal'],
                'signal_type': latest_data['signal_type'],
                'signal_reason': latest_data['signal_reason'],
                'rsi': latest_data['rsi'],
                'sma_20': latest_data['sma_20'],
                'sma_50': latest_data['sma_50'],
                'volume': latest_data['volume']
            }
            
            return signal_info
            
        except Exception as e:
            logger.error(f"Error getting current signal: {str(e)}")
            return {}
    
    def validate_signal_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate the quality of generated signals
        
        Args:
            data: DataFrame with signals
            
        Returns:
            Dictionary with signal quality metrics
        """
        try:
            metrics = {}
            
            # Total signals
            total_signals = (data['signal'] != 0).sum()
            buy_signals = (data['signal'] == 1).sum()
            sell_signals = (data['signal'] == -1).sum()
            
            metrics['total_signals'] = total_signals
            metrics['buy_signals'] = buy_signals
            metrics['sell_signals'] = sell_signals
            metrics['signal_ratio'] = buy_signals / sell_signals if sell_signals > 0 else 0
            
            # Signal frequency (signals per 100 days)
            trading_days = len(data)
            metrics['signal_frequency'] = (total_signals / trading_days) * 100
            
            # Average gap between signals
            signal_dates = data[data['signal'] != 0].index
            if len(signal_dates) > 1:
                gaps = []
                for i in range(1, len(signal_dates)):
                    # Convert to pandas Timestamp for proper date arithmetic
                    date1 = pd.to_datetime(signal_dates[i])
                    date2 = pd.to_datetime(signal_dates[i-1])
                    gaps.append((date1 - date2).days)
                metrics['avg_signal_gap'] = np.mean(gaps)
            else:
                metrics['avg_signal_gap'] = 0
            
            logger.info(f"Signal quality - Total: {total_signals}, "
                       f"Frequency: {metrics['signal_frequency']:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating signal quality: {str(e)}")
            return {}
