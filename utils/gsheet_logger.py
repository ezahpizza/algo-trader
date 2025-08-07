import gspread
from google.oauth2 import service_account
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime
import os
from config import config

logger = logging.getLogger(__name__)

class GoogleSheetsLogger:
    """Google Sheets integration for logging trades and performance"""
    
    def __init__(self, credentials_path: str = None, sheet_id: str = None):
        """
        Initialize Google Sheets logger
        
        Args:
            credentials_path: Path to service account JSON file
            sheet_id: Google Sheets ID
        """
        self.credentials_path = credentials_path or config.GOOGLE_CREDENTIALS_PATH
        self.sheet_id = sheet_id or config.GOOGLE_SHEET_ID
        self.gc = None
        self.sheet = None
        
        if self._authenticate():
            logger.info("Google Sheets authentication successful")
        else:
            logger.warning("Google Sheets authentication failed")
    
    def _authenticate(self) -> bool:
        """Authenticate with Google Sheets API"""
        try:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Credentials file not found: {self.credentials_path}")
                return False
            
            if not self.sheet_id:
                logger.error("Google Sheet ID not provided")
                return False
            
            # Define the scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Authenticate
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, 
                scopes=scope
            )
            
            self.gc = gspread.authorize(credentials)
            self.sheet = self.gc.open_by_key(self.sheet_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error authenticating with Google Sheets: {str(e)}")
            return False
    
    def setup_sheets(self) -> bool:
        """Setup the required worksheets with headers"""
        try:
            if not self.sheet:
                logger.error("Google Sheets not authenticated")
                return False
            
            # Setup Trade Log sheet
            self._setup_trade_log_sheet()
            
            # Setup Summary sheet
            self._setup_summary_sheet()
            
            # Setup Daily P&L sheet
            self._setup_daily_pnl_sheet()
            
            logger.info("Google Sheets setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up sheets: {str(e)}")
            return False
    
    def _setup_trade_log_sheet(self):
        """Setup Trade Log worksheet"""
        try:
            # Check if worksheet exists, create if not
            try:
                worksheet = self.sheet.worksheet("Trade Log")
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = self.sheet.add_worksheet(title="Trade Log", rows=1000, cols=10)
            
            # Headers for Trade Log
            headers = [
                "Date", "Ticker", "Signal", "Price", "Shares", "Value", 
                "P&L", "ML Prediction", "ML Confidence", "Actual Movement", "Reason"
            ]
            
            worksheet.insert_row(headers, 1)
            
            # Format headers
            worksheet.format('A1:K1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8}
            })
            
            logger.debug("Trade Log sheet setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Trade Log sheet: {str(e)}")
    
    def _setup_summary_sheet(self):
        """Setup Summary worksheet"""
        try:
            # Check if worksheet exists, create if not
            try:
                worksheet = self.sheet.worksheet("Summary")
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = self.sheet.add_worksheet(title="Summary", rows=100, cols=5)
            
            # Headers for Summary
            headers = ["Metric", "Value", "Date Updated", "", ""]
            worksheet.insert_row(headers, 1)
            
            # Initial metrics
            initial_data = [
                ["Total Trades", "0"],
                ["Winning Trades", "0"],
                ["Losing Trades", "0"],
                ["Win Ratio (%)", "0"],
                ["Total P&L", "0"],
                ["Average Trade P&L", "0"],
                ["Max Drawdown (%)", "0"],
                ["Sharpe Ratio", "0"],
                ["Total Return (%)", "0"],
                ["Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M")]
            ]
            
            for i, row in enumerate(initial_data, start=2):
                worksheet.update(f'A{i}:B{i}', [row])
            
            # Format headers
            worksheet.format('A1:E1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8}
            })
            
            logger.debug("Summary sheet setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Summary sheet: {str(e)}")
    
    def _setup_daily_pnl_sheet(self):
        """Setup Daily P&L worksheet"""
        try:
            # Check if worksheet exists, create if not
            try:
                worksheet = self.sheet.worksheet("Daily P&L")
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = self.sheet.add_worksheet(title="Daily P&L", rows=1000, cols=8)
            
            # Headers for Daily P&L
            headers = [
                "Date", "Portfolio Value", "Daily Return (%)", "Cumulative Return (%)",
                "Position", "Cash", "ML Prediction", "Signal"
            ]
            
            worksheet.insert_row(headers, 1)
            
            # Format headers
            worksheet.format('A1:H1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8}
            })
            
            logger.debug("Daily P&L sheet setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Daily P&L sheet: {str(e)}")
    
    def log_trade(self, trade_data: Dict[str, Any], 
                  ml_prediction: Dict[str, Any] = None,
                  actual_movement: float = None) -> bool:
        """
        Log a trade to the Trade Log sheet
        
        Args:
            trade_data: Dictionary with trade information
            ml_prediction: ML prediction data
            actual_movement: Actual next-day movement
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.sheet:
                logger.warning("Google Sheets not authenticated, skipping trade log")
                return False
            
            worksheet = self.sheet.worksheet("Trade Log")
            
            # Prepare row data
            row_data = [
                trade_data.get('date', ''),
                trade_data.get('symbol', ''),
                trade_data.get('signal_type', ''),
                trade_data.get('price', 0),
                trade_data.get('shares', 0),
                trade_data.get('value', 0),
                trade_data.get('pnl', 0),
                ml_prediction.get('prediction_label', '') if ml_prediction else '',
                ml_prediction.get('confidence', 0) if ml_prediction else 0,
                actual_movement or '',
                trade_data.get('reason', '')
            ]
            
            # Append row
            worksheet.append_row(row_data)
            
            logger.info(f"Trade logged: {trade_data.get('signal_type')} {trade_data.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")
            return False
    
    def update_summary(self, performance_metrics: Dict[str, Any]) -> bool:
        """
        Update the Summary sheet with performance metrics
        
        Args:
            performance_metrics: Dictionary with performance data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.sheet:
                logger.warning("Google Sheets not authenticated, skipping summary update")
                return False
            
            worksheet = self.sheet.worksheet("Summary")
            
            # Update specific cells with metrics
            updates = [
                ("B2", performance_metrics.get('total_trades', 0)),
                ("B3", performance_metrics.get('winning_trades', 0)),
                ("B4", performance_metrics.get('losing_trades', 0)),
                ("B5", round(performance_metrics.get('win_rate_pct', 0), 2)),
                ("B6", round(performance_metrics.get('total_pnl', 0), 2)),
                ("B7", round(performance_metrics.get('avg_trade', 0), 2)),
                ("B8", round(performance_metrics.get('max_drawdown_pct', 0), 2)),
                ("B9", round(performance_metrics.get('sharpe_ratio', 0), 3)),
                ("B10", round(performance_metrics.get('total_return_pct', 0), 2)),
                ("B11", datetime.now().strftime("%Y-%m-%d %H:%M"))
            ]
            
            # Batch update for efficiency
            worksheet.batch_update([
                {'range': cell, 'values': [[value]]} 
                for cell, value in updates
            ])
            
            logger.info("Summary sheet updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating summary: {str(e)}")
            return False
    
    def log_daily_portfolio(self, portfolio_data: List[Dict[str, Any]]) -> bool:
        """
        Log daily portfolio values to Daily P&L sheet
        
        Args:
            portfolio_data: List of daily portfolio data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.sheet:
                logger.warning("Google Sheets not authenticated, skipping daily portfolio log")
                return False
            
            worksheet = self.sheet.worksheet("Daily P&L")
            
            # Clear existing data (keep headers)
            worksheet.delete_rows(2, worksheet.row_count)
            
            # Prepare data rows
            rows_data = []
            initial_value = portfolio_data[0]['total_value'] if portfolio_data else 1
            
            for i, day_data in enumerate(portfolio_data):
                daily_return = 0
                if i > 0:
                    prev_value = portfolio_data[i-1]['total_value']
                    daily_return = ((day_data['total_value'] - prev_value) / prev_value) * 100
                
                cumulative_return = ((day_data['total_value'] - initial_value) / initial_value) * 100
                
                row = [
                    day_data.get('date', ''),
                    round(day_data.get('total_value', 0), 2),
                    round(daily_return, 2),
                    round(cumulative_return, 2),
                    day_data.get('position', 0),
                    round(day_data.get('cash', 0), 2),
                    day_data.get('ml_prediction', ''),
                    day_data.get('signal_type', '')
                ]
                rows_data.append(row)
            
            # Batch insert all rows
            if rows_data:
                worksheet.insert_rows(rows_data, 2)
            
            logger.info(f"Daily portfolio data logged: {len(rows_data)} days")
            return True
            
        except Exception as e:
            logger.error(f"Error logging daily portfolio: {str(e)}")
            return False
    
    def read_trade_log(self) -> pd.DataFrame:
        """Read trade log data from Google Sheets"""
        try:
            if not self.sheet:
                logger.error("Google Sheets not authenticated")
                return pd.DataFrame()
            
            worksheet = self.sheet.worksheet("Trade Log")
            data = worksheet.get_all_records()
            
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Read {len(df)} trade records from Google Sheets")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error reading trade log: {str(e)}")
            return pd.DataFrame()
    
    def create_performance_charts(self) -> bool:
        """Create performance visualization charts in Google Sheets"""
        try:
            # This would require additional setup for chart creation
            # For now, we'll log that the feature is available
            logger.info("Performance charts feature available for future implementation")
            return True
            
        except Exception as e:
            logger.error(f"Error creating performance charts: {str(e)}")
            return False
    
    def get_sheet_url(self) -> str:
        """Get the URL of the Google Sheet"""
        if self.sheet_id:
            return f"https://docs.google.com/spreadsheets/d/{self.sheet_id}"
        return ""
