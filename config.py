import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Global configuration class for the trading system"""
    
    # Stock Data Configuration
    NIFTY_50_STOCKS: List[str] = os.getenv('NIFTY_50_STOCKS', 'RELIANCE.NS,TCS.NS,INFY.NS').split(',')
    LOOKBACK_PERIOD: int = int(os.getenv('LOOKBACK_PERIOD', '365'))
    DATA_INTERVAL: str = os.getenv('DATA_INTERVAL', '1d')
    
    # Strategy Configuration
    RSI_PERIOD: int = int(os.getenv('RSI_PERIOD', '14'))
    RSI_OVERSOLD: float = float(os.getenv('RSI_OVERSOLD', '30'))
    RSI_OVERBOUGHT: float = float(os.getenv('RSI_OVERBOUGHT', '70'))
    SMA_SHORT: int = int(os.getenv('SMA_SHORT', '20'))
    SMA_LONG: int = int(os.getenv('SMA_LONG', '50'))
    
    # ML Configuration
    TRAIN_TEST_SPLIT_RATIO: float = float(os.getenv('TRAIN_TEST_SPLIT_RATIO', '0.8'))
    MODEL_SAVE_PATH: str = os.getenv('MODEL_SAVE_PATH', './models/model.pkl')
    
    # Google Sheets Configuration
    GOOGLE_SHEET_ID: str = os.getenv('GOOGLE_SHEET_ID', '')
    GOOGLE_CREDENTIALS_PATH: str = os.getenv('GOOGLE_CREDENTIALS_PATH', './credentials/service_account.json')
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Scheduler Configuration
    TRADING_TIMEZONE: str = os.getenv('TRADING_TIMEZONE', 'Asia/Kolkata')
    TRADING_HOUR: int = int(os.getenv('TRADING_HOUR', '9'))
    TRADING_MINUTE: int = int(os.getenv('TRADING_MINUTE', '30'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE_PATH: str = os.getenv('LOG_FILE_PATH', './logs/run.log')

# Global config instance
config = Config()
