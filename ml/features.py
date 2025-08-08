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
                features['rsi'] = df['rsi'].fillna(50)  # Fill with neutral RSI
                features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
                features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
                features['rsi_normalized'] = (df['rsi'].fillna(50) - 50) / 50  # Normalize around 50
            
            # MACD features
            if 'macd_histogram' in df.columns:
                macd_hist = df['macd_histogram'].fillna(0)
                features['macd_histogram'] = macd_hist
                features['macd_positive'] = (macd_hist > 0).astype(int)
                features['macd_momentum'] = macd_hist.diff().fillna(0)
            
            # Volume features
            if 'volume_delta' in df.columns:
                vol_delta = df['volume_delta'].fillna(0)
                # Clip extreme values to prevent infinity
                vol_delta = np.clip(vol_delta, -10, 10)
                features['volume_delta'] = vol_delta
                
                vol_mean = vol_delta.rolling(20, min_periods=1).mean()
                vol_std = vol_delta.rolling(20, min_periods=1).std().fillna(1)
                # Avoid division by zero in std
                vol_std = np.where(vol_std == 0, 1, vol_std)
                features['volume_spike'] = (vol_delta > vol_mean + 2 * vol_std).astype(int)
            
            # Momentum features
            if 'momentum' in df.columns:
                momentum = df['momentum'].fillna(0)
                # Clip extreme momentum values
                momentum = np.clip(momentum, -1, 1)
                features['momentum'] = momentum
                features['momentum_positive'] = (momentum > 0).astype(int)
            
            # SMA slope features
            if 'sma_20_slope' in df.columns:
                sma20_slope = df['sma_20_slope'].fillna(0)
                # Clip extreme slopes
                sma20_slope = np.clip(sma20_slope, -0.1, 0.1)
                features['sma_20_slope'] = sma20_slope
                features['sma_20_rising'] = (sma20_slope > 0).astype(int)
            
            if 'sma_50_slope' in df.columns:
                sma50_slope = df['sma_50_slope'].fillna(0)
                # Clip extreme slopes
                sma50_slope = np.clip(sma50_slope, -0.1, 0.1)
                features['sma_50_slope'] = sma50_slope
                features['sma_50_rising'] = (sma50_slope > 0).astype(int)
            
            # Additional technical features
            self._add_price_features(df, features)
            self._add_volatility_features(df, features)
            self._add_trend_features(df, features)
            self._add_pattern_features(df, features)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(features, index=df.index)
            
            # Clean the feature DataFrame
            feature_df = self._clean_features(feature_df)
            
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
            # Price position within day's range (avoid division by zero)
            range_size = data['high'] - data['low']
            range_size = np.where(range_size == 0, 1e-10, range_size)  # Avoid division by zero
            features['price_position'] = np.clip(
                (data['close'] - data['low']) / range_size, 0, 1
            ).fillna(0.5)
            
            # Daily return (clip extreme values)
            daily_returns = data['close'].pct_change().fillna(0)
            features['daily_return'] = np.clip(daily_returns, -0.5, 0.5)
            
            # Gap from previous close (avoid division by zero and clip)
            prev_close = data['close'].shift(1)
            prev_close = np.where(prev_close == 0, 1e-10, prev_close)
            gap = (data['open'] - prev_close) / prev_close
            features['gap'] = np.clip(gap.fillna(0), -0.5, 0.5)
            
            # Intraday movement (avoid division by zero and clip)
            open_price = np.where(data['open'] == 0, 1e-10, data['open'])
            intraday_ret = (data['close'] - data['open']) / open_price
            features['intraday_return'] = np.clip(intraday_ret.fillna(0), -0.5, 0.5)
            
            # Price vs moving averages (avoid division by zero and clip)
            if 'sma_20' in data.columns:
                sma_20 = np.where(data['sma_20'] == 0, 1e-10, data['sma_20'])
                price_vs_sma20 = (data['close'] - data['sma_20']) / sma_20
                features['price_vs_sma20'] = np.clip(price_vs_sma20.fillna(0), -2, 2)
            
            if 'sma_50' in data.columns:
                sma_50 = np.where(data['sma_50'] == 0, 1e-10, data['sma_50'])
                price_vs_sma50 = (data['close'] - data['sma_50']) / sma_50
                features['price_vs_sma50'] = np.clip(price_vs_sma50.fillna(0), -2, 2)
                
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
            features['true_range'] = true_range.fillna(0)
            
            # Average True Range
            atr = true_range.rolling(14, min_periods=1).mean().fillna(0)
            features['atr'] = np.clip(atr, 0, data['close'].median() * 0.1)  # Cap at 10% of median price
            
            # Volatility (rolling standard deviation of returns)
            returns = data['close'].pct_change().fillna(0)
            volatility = returns.rolling(20, min_periods=1).std().fillna(0)
            features['volatility'] = np.clip(volatility, 0, 0.2)  # Cap at 20% volatility
            
            # High-Low percentage (avoid division by zero and clip)
            close_price = np.where(data['close'] == 0, 1e-10, data['close'])
            hl_pct = (data['high'] - data['low']) / close_price * 100
            features['hl_pct'] = np.clip(hl_pct.fillna(0), 0, 50)  # Cap at 50%
            
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
                        if len(y) > 1 and not np.isnan(y).all():
                            try:
                                slope = np.polyfit(x, y, 1)[0]
                                # Normalize by price and clip extreme values
                                current_price = data['close'].iloc[i]
                                if current_price > 0:
                                    normalized_slope = slope / current_price
                                    normalized_slope = np.clip(normalized_slope, -0.01, 0.01)
                                    slope_values.append(normalized_slope)
                                else:
                                    slope_values.append(0)
                            except (np.linalg.LinAlgError, ValueError):
                                slope_values.append(0)
                        else:
                            slope_values.append(0)
                
                features[f'price_trend_{period}'] = slope_values
            
            # Consecutive up/down days
            returns = data['close'].pct_change().fillna(0)
            features['consecutive_up'] = self._count_consecutive(returns > 0.001)  # Small threshold
            features['consecutive_down'] = self._count_consecutive(returns < -0.001)
            
        except Exception as e:
            logger.error(f"Error adding trend features: {str(e)}")
    
    def _add_pattern_features(self, data: pd.DataFrame, features: dict):
        """Add pattern recognition features"""
        try:
            # Avoid division by zero in pattern calculations
            close_price = np.where(data['close'] == 0, 1e-10, data['close'])
            
            # Doji pattern (open ≈ close)
            body_size = abs(data['close'] - data['open']) / close_price
            features['doji'] = (body_size < 0.01).astype(int)
            
            # Hammer pattern (small body, long lower shadow)
            lower_shadow = (data['open'].combine(data['close'], min) - data['low']) / close_price
            upper_shadow = (data['high'] - data['open'].combine(data['close'], max)) / close_price
            features['hammer'] = ((lower_shadow > 2 * body_size) & 
                                (upper_shadow < body_size)).astype(int)
            
            # Engulfing patterns
            prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
            curr_body = abs(data['close'] - data['open'])
            # Handle NaN and zero values
            engulfing_condition = curr_body > prev_body * 1.5
            features['engulfing'] = engulfing_condition.fillna(False).astype(int)
            
        except Exception as e:
            logger.error(f"Error adding pattern features: {str(e)}")
    
    def _clean_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Clean feature DataFrame by handling infinite and extreme values"""
        try:
            # Replace infinite values with NaN
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with column median or 0
            for col in feature_df.columns:
                if feature_df[col].isna().all():
                    feature_df[col] = 0
                else:
                    median_val = feature_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    feature_df[col] = feature_df[col].fillna(median_val)
            
            # Clip extreme values using percentiles
            for col in feature_df.columns:
                if feature_df[col].dtype in ['float64', 'float32']:
                    q1 = feature_df[col].quantile(0.01)
                    q99 = feature_df[col].quantile(0.99)
                    if not pd.isna(q1) and not pd.isna(q99) and q1 != q99:
                        feature_df[col] = np.clip(feature_df[col], q1, q99)
            
            logger.debug(f"Cleaned features: {feature_df.shape}")
            return feature_df
            
        except Exception as e:
            logger.error(f"Error cleaning features: {str(e)}")
            return feature_df
    
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
