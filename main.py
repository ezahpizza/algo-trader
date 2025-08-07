def main():
    import logging
import os
import json
import pandas as pd
import numpy as np
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback

# Import our modules
from config import config
from data.fetcher import StockDataFetcher
from strategy.indicators import calculate_all_indicators
from strategy.rules import TradingStrategy
from ml.features import FeatureEngineer
from ml.trainer import MLTrainer
from ml.predictor import MLPredictor
from utils.backtester import Backtester
from utils.gsheet_logger import GoogleSheetsLogger
from utils.telegram_alerts import TelegramAlert

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    import logging
    
    os.makedirs(os.path.dirname(config.LOG_FILE_PATH), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )

# Create logger after setup
def get_logger():
    import logging
    return logging.getLogger(__name__)

logger = get_logger()

class AlgoTradingSystem:
    """Main algorithmic trading system"""
    
    def __init__(self):
        """Initialize the trading system"""
        setup_logging()
        logger.info("=== Algorithmic Trading System Started ===")
        
        # Initialize components
        self.data_fetcher = StockDataFetcher()
        self.strategy = TradingStrategy()
        self.feature_engineer = FeatureEngineer()
        self.ml_trainer = MLTrainer()
        self.ml_predictor = MLPredictor(config.MODEL_SAVE_PATH)
        self.backtester = Backtester()
        self.gsheet_logger = GoogleSheetsLogger()
        self.telegram = TelegramAlert()
        
        # System state
        self.last_run = None
        self.performance_metrics = {}
        
        logger.info("All components initialized successfully")
    
    def fetch_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch and prepare data for all stocks"""
        logger.info("Starting data fetch and preparation")
        
        try:
            # Fetch data for all stocks
            stock_data = self.data_fetcher.fetch_multiple_stocks()
            
            if not stock_data:
                raise Exception("No stock data fetched")
            
            # Calculate indicators for each stock
            prepared_data = {}
            for symbol, data in stock_data.items():
                logger.info(f"Calculating indicators for {symbol}")
                
                # Calculate technical indicators
                data_with_indicators = calculate_all_indicators(
                    data, 
                    config.RSI_PERIOD, 
                    config.SMA_SHORT, 
                    config.SMA_LONG
                )
                
                # Generate trading signals
                data_with_signals = self.strategy.generate_signals(data_with_indicators)
                
                # Filter signals to avoid overtrading
                data_with_signals = self.strategy.filter_signals(data_with_signals)
                
                prepared_data[symbol] = data_with_signals
            
            logger.info(f"Data preparation completed for {len(prepared_data)} stocks")
            return prepared_data
            
        except Exception as e:
            error_msg = f"Error in data preparation: {str(e)}"
            logger.error(error_msg)
            self.telegram.send_error_alert("Data Fetch Error", error_msg, "main.py")
            return {}
    
    def train_ml_models(self, stock_data: Dict[str, pd.DataFrame]) -> bool:
        """Train ML models using historical data"""
        logger.info("Starting ML model training")
        
        try:
            # Combine data from all stocks for training
            combined_features = []
            
            for symbol, data in stock_data.items():
                logger.info(f"Engineering features for {symbol}")
                
                # Create features
                features = self.feature_engineer.create_features(data)
                
                if not features.empty:
                    features['symbol'] = symbol
                    combined_features.append(features)
            
            if not combined_features:
                raise Exception("No features created for ML training")
            
            # Combine all features
            all_features = pd.concat(combined_features, ignore_index=True)
            logger.info(f"Combined features shape: {all_features.shape}")
            
            # Prepare for training
            X, y = self.feature_engineer.prepare_features_for_training(all_features)
            
            if X.empty or y.empty:
                raise Exception("No valid data for ML training")
            
            # Train model
            training_results = self.ml_trainer.train_model(X, y)
            
            if training_results:
                # Save model
                if self.ml_trainer.save_model():
                    logger.info("ML model trained and saved successfully")
                    
                    # Reload model in predictor
                    self.ml_predictor = MLPredictor(config.MODEL_SAVE_PATH)
                    
                    # Log training metrics
                    test_accuracy = training_results.get('test_metrics', {}).get('accuracy', 0)
                    logger.info(f"Model test accuracy: {test_accuracy:.3f}")
                    
                    return True
                else:
                    raise Exception("Failed to save trained model")
            else:
                raise Exception("Model training failed")
                
        except Exception as e:
            error_msg = f"Error in ML training: {str(e)}"
            logger.error(error_msg)
            self.telegram.send_error_alert("ML Training Error", error_msg, "main.py")
            return False
    
    def run_backtest(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run backtest on historical data"""
        logger.info("Starting backtest")
        
        try:
            all_results = {}
            
            for symbol, data in stock_data.items():
                logger.info(f"Running backtest for {symbol}")
                
                # Run backtest for this stock
                results = self.backtester.run_backtest(data, data)
                
                if results:
                    all_results[symbol] = results
                    
                    # Log key metrics
                    total_return = results.get('total_return_pct', 0)
                    win_rate = results.get('win_rate_pct', 0)
                    total_trades = results.get('total_trades', 0)
                    
                    logger.info(f"{symbol} backtest: {total_return:.2f}% return, "
                               f"{win_rate:.1f}% win rate, {total_trades} trades")
            
            # Calculate combined metrics
            if all_results:
                combined_metrics = self._combine_backtest_results(all_results)
                self.performance_metrics = combined_metrics
                
                logger.info("Backtest completed successfully")
                return combined_metrics
            else:
                raise Exception("No backtest results generated")
                
        except Exception as e:
            error_msg = f"Error in backtesting: {str(e)}"
            logger.error(error_msg)
            self.telegram.send_error_alert("Backtest Error", error_msg, "main.py")
            return {}
    
    def generate_current_signals(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate current trading signals with ML predictions"""
        logger.info("Generating current signals")
        
        current_signals = []
        
        try:
            for symbol, data in stock_data.items():
                # Get current signal from strategy
                current_signal = self.strategy.get_current_signal(data)
                
                if current_signal and current_signal.get('signal') != 0:
                    # Get ML prediction if model is available
                    ml_prediction = None
                    if self.ml_predictor.is_model_ready():
                        # Create features for prediction
                        features = self.feature_engineer.create_features(data)
                        if not features.empty:
                            ml_prediction = self.ml_predictor.predict_next_day_movement(features)
                    
                    # Combine signal with ML prediction
                    signal_data = {
                        'symbol': symbol,
                        'date': current_signal.get('date'),
                        'signal_type': current_signal.get('signal_type'),
                        'price': current_signal.get('price'),
                        'reason': current_signal.get('signal_reason'),
                        'rsi': current_signal.get('rsi'),
                        'sma_20': current_signal.get('sma_20'),
                        'sma_50': current_signal.get('sma_50'),
                        'ml_prediction': ml_prediction
                    }
                    
                    current_signals.append(signal_data)
                    
                    # Send Telegram alert
                    self.telegram.send_signal_alert(signal_data, ml_prediction)
                    
                    logger.info(f"Signal generated: {signal_data['signal_type']} {symbol}")
            
            return current_signals
            
        except Exception as e:
            error_msg = f"Error generating signals: {str(e)}"
            logger.error(error_msg)
            self.telegram.send_error_alert("Signal Generation Error", error_msg, "main.py")
            return []
    
    def log_to_google_sheets(self, signals: List[Dict[str, Any]], 
                           performance_metrics: Dict[str, Any]) -> bool:
        """Log data to Google Sheets"""
        logger.info("Logging to Google Sheets")
        
        try:
            # Setup sheets if first run
            if not hasattr(self, '_sheets_setup'):
                self.gsheet_logger.setup_sheets()
                self._sheets_setup = True
            
            # Log signals as trades
            for signal in signals:
                trade_data = {
                    'date': signal.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'symbol': signal.get('symbol'),
                    'signal_type': signal.get('signal_type'),
                    'price': signal.get('price', 0),
                    'shares': 100,  # Default position size
                    'value': signal.get('price', 0) * 100,
                    'pnl': 0,  # Will be updated when position is closed
                    'reason': signal.get('reason', '')
                }
                
                ml_prediction = signal.get('ml_prediction')
                
                self.gsheet_logger.log_trade(trade_data, ml_prediction)
            
            # Update summary sheet
            if performance_metrics:
                self.gsheet_logger.update_summary(performance_metrics)
            
            logger.info("Google Sheets logging completed")
            return True
            
        except Exception as e:
            error_msg = f"Error logging to Google Sheets: {str(e)}"
            logger.error(error_msg)
            self.telegram.send_error_alert("Google Sheets Error", error_msg, "main.py")
            return False
    
    def export_performance_data(self) -> bool:
        """Export performance data to local files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export performance metrics as JSON
            if self.performance_metrics:
                metrics_file = f"performance_metrics_{timestamp}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(self.performance_metrics, f, indent=2, default=str)
                logger.info(f"Performance metrics exported to {metrics_file}")
            
            # Export backtest results as Excel
            if hasattr(self.backtester, 'trades') and self.backtester.trades:
                excel_file = f"backtest_results_{timestamp}.xlsx"
                self.backtester.export_results(excel_file)
                logger.info(f"Backtest results exported to {excel_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {str(e)}")
            return False
    
    def run_daily_pipeline(self):
        """Run the complete daily trading pipeline"""
        logger.info("Starting daily trading pipeline")
        self.last_run = datetime.now()
        
        try:
            # Step 1: Fetch and prepare data
            stock_data = self.fetch_and_prepare_data()
            if not stock_data:
                raise Exception("Failed to fetch stock data")
            
            # Step 2: Train ML models (periodically)
            should_train = self._should_retrain_model()
            if should_train:
                logger.info("Retraining ML model")
                self.train_ml_models(stock_data)
            
            # Step 3: Run backtest
            performance_metrics = self.run_backtest(stock_data)
            
            # Step 4: Generate current signals
            current_signals = self.generate_current_signals(stock_data)
            
            # Step 5: Log to Google Sheets
            self.log_to_google_sheets(current_signals, performance_metrics)
            
            # Step 6: Export local data
            self.export_performance_data()
            
            # Step 7: Send performance update
            if performance_metrics:
                self.telegram.send_performance_update(performance_metrics)
            
            # Step 8: Send daily summary
            summary_data = {
                'signals_generated': len(current_signals),
                'trades_executed': len([s for s in current_signals if s.get('signal_type') in ['BUY', 'SELL']]),
                'portfolio_value': performance_metrics.get('final_value', 0),
                'daily_pnl': 0,  # Would need position tracking for this
                'ml_accuracy': self.ml_predictor.get_model_info().get('training_metrics', {}).get('accuracy', 0)
            }
            self.telegram.send_daily_summary(summary_data)
            
            logger.info("Daily pipeline completed successfully")
            
        except Exception as e:
            error_msg = f"Error in daily pipeline: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.telegram.send_error_alert("Pipeline Error", error_msg, "main.py")
    
    def _should_retrain_model(self) -> bool:
        """Determine if model should be retrained"""
        # Retrain weekly (every 7 days)
        if not hasattr(self, '_last_training'):
            return True
        
        days_since_training = (datetime.now() - self._last_training).days
        return days_since_training >= 7
    
    def _combine_backtest_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine backtest results from multiple stocks"""
        try:
            total_trades = sum(result.get('total_trades', 0) for result in results.values())
            total_winning = sum(result.get('winning_trades', 0) for result in results.values())
            total_losing = sum(result.get('losing_trades', 0) for result in results.values())
            
            # Calculate weighted averages
            total_return = np.mean([result.get('total_return_pct', 0) for result in results.values()])
            win_rate = (total_winning / total_trades * 100) if total_trades > 0 else 0
            max_drawdown = np.mean([result.get('max_drawdown_pct', 0) for result in results.values()])
            sharpe_ratio = np.mean([result.get('sharpe_ratio', 0) for result in results.values()])
            
            combined = {
                'total_trades': total_trades,
                'winning_trades': total_winning,
                'losing_trades': total_losing,
                'win_rate_pct': win_rate,
                'total_return_pct': total_return,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'individual_results': results
            }
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining backtest results: {str(e)}")
            return {}
    
    def test_system_components(self):
        """Test all system components"""
        logger.info("Testing system components")
        
        # Test Telegram
        telegram_ok = self.telegram.test_connection()
        
        # Test data fetching
        test_data = self.data_fetcher.fetch_stock_data('RELIANCE.NS')
        data_ok = test_data is not None and not test_data.empty
        
        # Test ML model
        ml_ok = self.ml_predictor.is_model_ready()
        
        # Test Google Sheets
        sheets_ok = self.gsheet_logger.gc is not None
        
        status_data = {
            'data_fetch': 'OK' if data_ok else 'ERROR',
            'ml_model': 'OK' if ml_ok else 'WARNING',
            'google_sheets': 'OK' if sheets_ok else 'WARNING',
            'telegram': 'OK' if telegram_ok else 'ERROR',
            'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.telegram.send_system_status(status_data)
        logger.info(f"System test completed: {status_data}")
    
    def schedule_daily_run(self):
        """Schedule daily runs"""
        logger.info(f"Scheduling daily runs at {config.TRADING_HOUR}:{config.TRADING_MINUTE:02d}")
        
        schedule.every().day.at(f"{config.TRADING_HOUR:02d}:{config.TRADING_MINUTE:02d}").do(
            self.run_daily_pipeline
        )
        
        # Test run on startup
        logger.info("Running initial test")
        self.test_system_components()
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main entry point"""
    try:
        # Initialize trading system
        trading_system = AlgoTradingSystem()
        
        # Check command line arguments
        import sys
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'test':
                trading_system.test_system_components()
            elif command == 'run':
                trading_system.run_daily_pipeline()
            elif command == 'schedule':
                trading_system.schedule_daily_run()
            elif command == 'train':
                stock_data = trading_system.fetch_and_prepare_data()
                if stock_data:
                    trading_system.train_ml_models(stock_data)
            else:
                print("Available commands: test, run, schedule, train")
        else:
            # Default: run scheduler
            trading_system.schedule_daily_run()
            
    except KeyboardInterrupt:
        logger.info("System shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
