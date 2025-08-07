import yfinance as yf
import pandas as pd
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from config import config

logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Fetches and validates stock data from Yahoo Finance"""
    
    def __init__(self):
        self.stocks = config.NIFTY_50_STOCKS
        self.lookback_period = config.LOOKBACK_PERIOD
        self.interval = config.DATA_INTERVAL
    
    def fetch_stock_data(self, symbol: str, period_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period_days: Number of days to look back (default from config)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            period_days = period_days or self.lookback_period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=self.interval
            )
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Validate data
            if not self._validate_data(data, symbol):
                return None
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data.reset_index(inplace=True)
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols (default from config)
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        symbols = symbols or self.stocks
        stock_data = {}
        
        logger.info(f"Fetching data for {len(symbols)} stocks")
        
        for symbol in symbols:
            data = self.fetch_stock_data(symbol)
            if data is not None:
                stock_data[symbol] = data
            else:
                logger.warning(f"Skipping {symbol} due to data fetch failure")
        
        logger.info(f"Successfully fetched data for {len(stock_data)} out of {len(symbols)} stocks")
        return stock_data
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate the fetched data
        
        Args:
            data: Raw data from yfinance
            symbol: Stock symbol for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if data has required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing columns for {symbol}: {missing_columns}")
            return False
        
        # Check for sufficient data points
        if len(data) < 50:  # Need at least 50 days for indicators
            logger.warning(f"Insufficient data for {symbol}: {len(data)} records")
            return False
        
        # Check for data quality issues
        if data[required_columns].isnull().sum().sum() > len(data) * 0.1:  # More than 10% null values
            logger.warning(f"Too many null values for {symbol}")
            return False
        
        # Check for unrealistic price movements
        daily_returns = data['Close'].pct_change().abs()
        if daily_returns.max() > 0.5:  # More than 50% daily change
            logger.warning(f"Unrealistic price movements detected for {symbol}")
        
        # Check for zero volume days
        zero_volume_days = (data['Volume'] == 0).sum()
        if zero_volume_days > len(data) * 0.05:  # More than 5% zero volume days
            logger.warning(f"Too many zero volume days for {symbol}: {zero_volume_days}")
        
        return True
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest close price or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('previousClose')
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None
