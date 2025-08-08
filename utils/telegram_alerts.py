import requests
import logging
import time
from typing import Dict, Any
from config import config


logger = logging.getLogger(__name__)

class TelegramAlert:
    """Telegram bot integration for trading alerts"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Telegram alert system
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
        """
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if self.bot_token and self.chat_id:
            logger.info("Telegram alerts initialized")
        else:
            logger.warning("Telegram credentials not provided, alerts disabled")
    
    def send_message(self, message: str, parse_mode: str = 'HTML', max_retries: int = 3) -> bool:
        """
        Send a message to Telegram with retry logic
        
        Args:
            message: Message text
            parse_mode: Message formatting ('HTML' or 'Markdown')
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if not self.bot_token or not self.chat_id:
                logger.warning("Telegram not configured, message not sent")
                return False
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, json=payload, timeout=15)
                    
                    if response.status_code == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying Telegram message (attempt {attempt + 2}/{max_retries})")
                            continue
                        return False
                        
                except requests.exceptions.ConnectionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection error, retrying (attempt {attempt + 2}/{max_retries}): {str(e)}")
                        continue
                    else:
                        logger.error(f"Failed to send Telegram message after {max_retries} attempts: {str(e)}")
                        return False
                except requests.exceptions.Timeout as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout error, retrying (attempt {attempt + 2}/{max_retries}): {str(e)}")
                        continue
                    else:
                        logger.error(f"Telegram message timeout after {max_retries} attempts: {str(e)}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def send_signal_alert(self, signal_data: Dict[str, Any], 
                         ml_prediction: Dict[str, Any] = None) -> bool:
        """
        Send trading signal alert
        
        Args:
            signal_data: Signal information
            ml_prediction: ML prediction data
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            signal_type = signal_data.get('signal_type', 'UNKNOWN')
            symbol = signal_data.get('symbol', 'Unknown')
            price = signal_data.get('price', 0)
            reason = signal_data.get('signal_reason', 'No reason')
            
            # Create alert message
            message = f"<b>🚨 TRADING SIGNAL ALERT</b>\n\n"
            message += f"<b>Signal:</b> {signal_type}\n"
            message += f"<b>Stock:</b> {symbol}\n"
            message += f"<b>Price:</b> Rs.{price:.2f}\n"
            message += f"<b>Reason:</b> {reason}\n"
            
            if ml_prediction:
                ml_signal = ml_prediction.get('prediction_label', 'Unknown')
                confidence = ml_prediction.get('confidence', 0)
                message += f"\n<b>🤖 ML Prediction:</b> {ml_signal}\n"
                message += f"<b>Confidence:</b> {confidence:.1%}\n"
            
            message += f"\n<b>Time:</b> {signal_data.get('date', 'Unknown')}"
            
            # Add appropriate emoji based on signal
            if signal_type == 'BUY':
                message = "🟢 " + message
            elif signal_type == 'SELL':
                message = "🔴 " + message
            else:
                message = "⚪ " + message
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending signal alert: {str(e)}")
            return False
    
    def send_performance_update(self, performance_data: Dict[str, Any]) -> bool:
        """
        Send performance update alert
        
        Args:
            performance_data: Performance metrics
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"<b>📊 PERFORMANCE UPDATE</b>\n\n"
            
            total_return = performance_data.get('total_return_pct', 0)
            total_trades = performance_data.get('total_trades', 0)
            win_rate = performance_data.get('win_rate_pct', 0)
            
            message += f"<b>Total Return:</b> {total_return:.2f}%\n"
            message += f"<b>Total Trades:</b> {total_trades}\n"
            message += f"<b>Win Rate:</b> {win_rate:.1f}%\n"
            
            if 'max_drawdown_pct' in performance_data:
                drawdown = performance_data['max_drawdown_pct']
                message += f"<b>Max Drawdown:</b> {drawdown:.2f}%\n"
            
            if 'sharpe_ratio' in performance_data:
                sharpe = performance_data['sharpe_ratio']
                message += f"<b>Sharpe Ratio:</b> {sharpe:.2f}\n"
            
            # Add emoji based on performance
            if total_return > 0:
                message = "📈 " + message
            else:
                message = "📉 " + message
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending performance update: {str(e)}")
            return False
    
    def send_error_alert(self, error_type: str, error_message: str, 
                        module_name: str = None) -> bool:
        """
        Send error alert
        
        Args:
            error_type: Type of error
            error_message: Error message
            module_name: Module where error occurred
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"<b>⚠️ SYSTEM ERROR ALERT</b>\n\n"
            message += f"<b>Error Type:</b> {error_type}\n"
            
            if module_name:
                message += f"<b>Module:</b> {module_name}\n"
            
            message += f"<b>Message:</b> {error_message[:200]}...\n"  # Truncate long messages
            message += f"\n<b>Time:</b> {self._get_current_time()}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error alert: {str(e)}")
            return False
    
    def send_ml_prediction_alert(self, prediction_data: Dict[str, Any], 
                                stock_symbol: str) -> bool:
        """
        Send ML prediction alert
        
        Args:
            prediction_data: ML prediction results
            stock_symbol: Stock symbol
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            prediction = prediction_data.get('prediction_label', 'Unknown')
            confidence = prediction_data.get('confidence', 0)
            prob_positive = prediction_data.get('probability_positive', 0)
            
            message = f"<b>🤖 ML PREDICTION</b>\n\n"
            message += f"<b>Stock:</b> {stock_symbol}\n"
            message += f"<b>Prediction:</b> {prediction}\n"
            message += f"<b>Confidence:</b> {confidence:.1%}\n"
            message += f"<b>Upward Probability:</b> {prob_positive:.1%}\n"
            
            # Add confidence level indicator
            if confidence >= 0.8:
                message = "🔥 " + message + "\n<i>High confidence prediction!</i>"
            elif confidence >= 0.6:
                message = "✅ " + message + "\n<i>Medium confidence prediction</i>"
            else:
                message = "⚠️ " + message + "\n<i>Low confidence prediction</i>"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending ML prediction alert: {str(e)}")
            return False
    
    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Send daily summary alert
        
        Args:
            summary_data: Daily summary information
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"<b>📋 DAILY SUMMARY</b>\n\n"
            
            signals_generated = summary_data.get('signals_generated', 0)
            trades_executed = summary_data.get('trades_executed', 0)
            daily_pnl = summary_data.get('daily_pnl', 0)
            
            message += f"<b>Signals Generated:</b> {signals_generated}\n"
            message += f"<b>Trades Executed:</b> {trades_executed}\n"
            message += f"<b>Daily P&L:</b> Rs.{daily_pnl:.2f}\n"
            
            if 'portfolio_value' in summary_data:
                portfolio_value = summary_data['portfolio_value']
                message += f"<b>Portfolio Value:</b> Rs.{portfolio_value:,.2f}\n"
            
            if 'ml_accuracy' in summary_data:
                ml_accuracy = summary_data['ml_accuracy']
                message += f"<b>ML Accuracy:</b> {ml_accuracy:.1%}\n"
            
            message += f"\n<b>Date:</b> {self._get_current_time().split()[0]}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")
            return False
    
    def send_system_status(self, status_data: Dict[str, Any]) -> bool:
        """
        Send system status alert
        
        Args:
            status_data: System status information
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"<b>🔧 SYSTEM STATUS</b>\n\n"
            
            data_fetch_status = status_data.get('data_fetch', 'Unknown')
            ml_model_status = status_data.get('ml_model', 'Unknown')
            sheets_status = status_data.get('google_sheets', 'Unknown')
            
            message += f"<b>Data Fetching:</b> {data_fetch_status}\n"
            message += f"<b>ML Model:</b> {ml_model_status}\n"
            message += f"<b>Google Sheets:</b> {sheets_status}\n"
            
            if 'last_run' in status_data:
                last_run = status_data['last_run']
                message += f"<b>Last Run:</b> {last_run}\n"
            
            # Overall status
            all_statuses = [data_fetch_status, ml_model_status, sheets_status]
            if all(status == 'OK' for status in all_statuses):
                message = "✅ " + message
            elif any(status == 'ERROR' for status in all_statuses):
                message = "❌ " + message
            else:
                message = "⚠️ " + message
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending system status: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test Telegram bot connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.bot_token:
                logger.error("No Telegram bot token provided")
                return False
            
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                bot_name = bot_info.get('result', {}).get('username', 'Unknown')
                logger.info(f"Telegram bot connection successful: @{bot_name}")
                
                # Send test message if chat_id is provided
                if self.chat_id:
                    test_message = "🤖 <b>Algo Trading Bot Connected</b>\n\nConnection test successful!"
                    return self.send_message(test_message)
                
                return True
            else:
                logger.error(f"Telegram bot connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing Telegram connection: {str(e)}")
            return False
    
    def _get_current_time(self) -> str:
        """Get current time formatted as string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
