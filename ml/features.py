import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for ML models"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_column = 'next_day_positive'
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML model
        
        Args:
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        try:
            # Ensure we have required indicators
            required_indicators = ['rsi', 'macd_histogram', 'volume_delta', 'momentum', 
                                 'sma_20_slope', 'sma_50_slope']
            missing_indicators = [col for col in required_indicators if col not in df.columns]
            if missing_indicators:
                logger.warning(f"Missing indicators: {missing_indicators}")
            
            # Primary features from requirements
            features = {}
            
            # RSI features
            if 'rsi' in df.columns:
                features['rsi'] = df['rsi']
                features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
                features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
                features['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize around 50
            
            # MACD features
            if 'macd_histogram' in df.columns:
                features['macd_histogram'] = df['macd_histogram']
                features['macd_positive'] = (df['macd_histogram'] > 0).astype(int)
                features['macd_momentum'] = df['macd_histogram'].diff()
            
            # Volume features
            if 'volume_delta' in df.columns:
                features['volume_delta'] = df['volume_delta'].fillna(0)
                features['volume_spike'] = (df['volume_delta'] > df['volume_delta'].rolling(20).mean() + 
                                          2 * df['volume_delta'].rolling(20).std()).astype(int)
            
            # Momentum features
            if 'momentum' in df.columns:
                features['momentum'] = df['momentum'].fillna(0)
                features['momentum_positive'] = (df['momentum'] > 0).astype(int)
            
            # SMA slope features
            if 'sma_20_slope' in df.columns:
                features['sma_20_slope'] = df['sma_20_slope'].fillna(0)
                features['sma_20_rising'] = (df['sma_20_slope'] > 0).astype(int)
            
            if 'sma_50_slope' in df.columns:
                features['sma_50_slope'] = df['sma_50_slope'].fillna(0)
                features['sma_50_rising'] = (df['sma_50_slope'] > 0).astype(int)
            
            # Additional technical features
            self._add_price_features(df, features)
            self._add_volatility_features(df, features)
            self._add_trend_features(df, features)
            self._add_pattern_features(df, features)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(features, index=df.index)
            
            # Store feature column names
            self.feature_columns = feature_df.columns.tolist()
            
            # Create target variable (next day positive return)
            feature_df[self.target_column] = self._create_target(df)
            
            logger.info(f"Created {len(self.feature_columns)} features")
            return feature_df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return pd.DataFrame()
    
    def _add_price_features(self, data: pd.DataFrame, features: dict):
        """Add price-based features"""
        try:
            # Price position within day's range
            features['price_position'] = ((data['close'] - data['low']) / 
                                        (data['high'] - data['low'])).fillna(0.5)
            
            # Daily return
            features['daily_return'] = data['close'].pct_change().fillna(0)
            
            # Gap from previous close
            features['gap'] = ((data['open'] - data['close'].shift(1)) / 
                             data['close'].shift(1)).fillna(0)
            
            # Intraday movement
            features['intraday_return'] = ((data['close'] - data['open']) / data['open']).fillna(0)
            
            # Price vs moving averages
            if 'sma_20' in data.columns:
                features['price_vs_sma20'] = ((data['close'] - data['sma_20']) / 
                                            data['sma_20']).fillna(0)
            
            if 'sma_50' in data.columns:
                features['price_vs_sma50'] = ((data['close'] - data['sma_50']) / 
                                            data['sma_50']).fillna(0)
                
        except Exception as e:
            logger.error(f"Error adding price features: {str(e)}")
    
    def _add_volatility_features(self, data: pd.DataFrame, features: dict):
        """Add volatility-based features"""
        try:
            # True Range
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            features['true_range'] = true_range
            
            # Average True Range
            features['atr'] = true_range.rolling(14).mean().fillna(0)
            
            # Volatility (rolling standard deviation of returns)
            returns = data['close'].pct_change()
            features['volatility'] = returns.rolling(20).std().fillna(0)
            
            # High-Low percentage
            features['hl_pct'] = ((data['high'] - data['low']) / data['close'] * 100).fillna(0)
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {str(e)}")
    
    def _add_trend_features(self, data: pd.DataFrame, features: dict):
        """Add trend-based features"""
        try:
            # Price trend (linear regression slope)
            for period in [5, 10, 20]:
                slope_values = []
                for i in range(len(data)):
                    if i < period:
                        slope_values.append(0)
                    else:
                        y = data['close'].iloc[i-period:i].values
                        x = np.arange(len(y))
                        if len(y) > 1:
                            slope = np.polyfit(x, y, 1)[0]
                            slope_values.append(slope / data['close'].iloc[i])  # Normalize by price
                        else:
                            slope_values.append(0)
                
                features[f'price_trend_{period}'] = slope_values
            
            # Consecutive up/down days
            returns = data['close'].pct_change()
            features['consecutive_up'] = self._count_consecutive(returns > 0)
            features['consecutive_down'] = self._count_consecutive(returns < 0)
            
        except Exception as e:
            logger.error(f"Error adding trend features: {str(e)}")
    
    def _add_pattern_features(self, data: pd.DataFrame, features: dict):
        """Add pattern recognition features"""
        try:
            # Doji pattern (open ≈ close)
            body_size = abs(data['close'] - data['open']) / data['close']
            features['doji'] = (body_size < 0.01).astype(int)
            
            # Hammer pattern (small body, long lower shadow)
            lower_shadow = (data['open'].combine(data['close'], min) - data['low']) / data['close']
            upper_shadow = (data['high'] - data['open'].combine(data['close'], max)) / data['close']
            features['hammer'] = ((lower_shadow > 2 * body_size) & 
                                (upper_shadow < body_size)).astype(int)
            
            # Engulfing patterns
            prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
            curr_body = abs(data['close'] - data['open'])
            features['engulfing'] = (curr_body > prev_body * 1.5).astype(int)
            
        except Exception as e:
            logger.error(f"Error adding pattern features: {str(e)}")
    
    def _count_consecutive(self, condition: pd.Series) -> List[int]:
        """Count consecutive True values in a boolean series"""
        consecutive = []
        count = 0
        
        for value in condition:
            if value:
                count += 1
            else:
                count = 0
            consecutive.append(count)
        
        return consecutive
    
    def _create_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Create target variable (next day positive return)
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with binary target (1 if next day return > 0, 0 otherwise)
        """
        try:
            # Calculate next day return
            next_day_return = data['close'].shift(-1) / data['close'] - 1
            
            # Create binary target
            target = (next_day_return > 0).astype(int)
            
            logger.debug(f"Created target with {target.sum()} positive days out of {len(target)}")
            return target
            
        except Exception as e:
            logger.error(f"Error creating target: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_columns.copy()
    
    def prepare_features_for_training(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Remove rows with NaN target (last row typically)
            clean_data = data.dropna(subset=[self.target_column])
            
            # Separate features and target
            X = clean_data[self.feature_columns].fillna(0)  # Fill NaN features with 0
            y = clean_data[self.target_column]
            
            logger.info(f"Prepared {len(X)} samples with {len(self.feature_columns)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features for training: {str(e)}")
            return pd.DataFrame(), pd.Series()
