import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission rate per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.portfolio = []
        
        logger.info(f"Initialized backtester with capital: ₹{initial_capital:,.2f}")
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on historical data with signals
        
        Args:
            data: DataFrame with OHLCV data
            signals: DataFrame with trading signals
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Merge data and signals
            backtest_data = pd.merge(data, signals[['signal', 'signal_type', 'signal_reason']], 
                                   left_index=True, right_index=True, how='left')
            backtest_data['signal'] = backtest_data['signal'].fillna(0)
            backtest_data['signal_type'] = backtest_data['signal_type'].fillna('HOLD')
            
            # Initialize portfolio tracking
            portfolio = {
                'cash': self.initial_capital,
                'position': 0,  # Number of shares
                'position_value': 0,
                'total_value': self.initial_capital,
                'trades': []
            }
            
            # Track daily portfolio values
            daily_portfolio = []
            
            for date, row in backtest_data.iterrows():
                # Update position value with current price
                portfolio['position_value'] = portfolio['position'] * row['close']
                portfolio['total_value'] = portfolio['cash'] + portfolio['position_value']
                
                # Process signal
                if row['signal'] == 1 and portfolio['position'] == 0:  # Buy signal
                    self._execute_buy(portfolio, row, date)
                elif row['signal'] == -1 and portfolio['position'] > 0:  # Sell signal
                    self._execute_sell(portfolio, row, date)
                
                # Record daily portfolio state
                daily_portfolio.append({
                    'date': date,
                    'close_price': row['close'],
                    'cash': portfolio['cash'],
                    'position': portfolio['position'],
                    'position_value': portfolio['position_value'],
                    'total_value': portfolio['total_value'],
                    'signal': row['signal'],
                    'signal_type': row['signal_type']
                })
            
            # Calculate final metrics
            self.trades = portfolio['trades']
            self.portfolio = daily_portfolio
            
            backtest_results = self._calculate_performance_metrics(daily_portfolio)
            backtest_results['trades'] = self.trades
            backtest_results['daily_portfolio'] = daily_portfolio
            
            logger.info(f"Backtest completed: {len(self.trades)} trades, "
                       f"Final value: ₹{backtest_results['final_value']:,.2f}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {}
    
    def _execute_buy(self, portfolio: Dict, row: pd.Series, date: datetime):
        """Execute buy order"""
        try:
            buy_price = row['close']
            max_shares = int(portfolio['cash'] / (buy_price * (1 + self.commission)))
            
            if max_shares > 0:
                shares_bought = max_shares
                total_cost = shares_bought * buy_price * (1 + self.commission)
                
                portfolio['cash'] -= total_cost
                portfolio['position'] += shares_bought
                
                trade = {
                    'date': date,
                    'type': 'BUY',
                    'shares': shares_bought,
                    'price': buy_price,
                    'value': shares_bought * buy_price,
                    'commission': shares_bought * buy_price * self.commission,
                    'total_cost': total_cost,
                    'reason': row.get('signal_reason', ''),
                    'remaining_cash': portfolio['cash']
                }
                
                portfolio['trades'].append(trade)
                logger.debug(f"BUY: {shares_bought} shares at ₹{buy_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error executing buy order: {str(e)}")
    
    def _execute_sell(self, portfolio: Dict, row: pd.Series, date: datetime):
        """Execute sell order"""
        try:
            sell_price = row['close']
            shares_to_sell = portfolio['position']
            
            if shares_to_sell > 0:
                gross_proceeds = shares_to_sell * sell_price
                commission_cost = gross_proceeds * self.commission
                net_proceeds = gross_proceeds - commission_cost
                
                portfolio['cash'] += net_proceeds
                portfolio['position'] = 0
                
                # Find corresponding buy trade to calculate P&L
                buy_trade = None
                for trade in reversed(portfolio['trades']):
                    if trade['type'] == 'BUY' and 'pnl' not in trade:
                        buy_trade = trade
                        break
                
                pnl = 0
                if buy_trade:
                    pnl = net_proceeds - buy_trade['total_cost']
                    buy_trade['pnl'] = pnl
                
                trade = {
                    'date': date,
                    'type': 'SELL',
                    'shares': shares_to_sell,
                    'price': sell_price,
                    'value': gross_proceeds,
                    'commission': commission_cost,
                    'net_proceeds': net_proceeds,
                    'pnl': pnl,
                    'reason': row.get('signal_reason', ''),
                    'remaining_cash': portfolio['cash']
                }
                
                portfolio['trades'].append(trade)
                logger.debug(f"SELL: {shares_to_sell} shares at ₹{sell_price:.2f}, P&L: ₹{pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error executing sell order: {str(e)}")
    
    def _calculate_performance_metrics(self, daily_portfolio: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            df_portfolio = pd.DataFrame(daily_portfolio)
            
            if df_portfolio.empty:
                return {}
            
            # Basic metrics
            initial_value = self.initial_capital
            final_value = df_portfolio['total_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Trade metrics
            completed_trades = [t for t in self.trades if t['type'] == 'SELL']
            total_trades = len(completed_trades)
            
            if total_trades > 0:
                # P&L metrics
                trade_pnls = [t['pnl'] for t in completed_trades]
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
                
                win_rate = len(winning_trades) / total_trades
                total_pnl = sum(trade_pnls)
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                avg_trade = np.mean(trade_pnls)
                
                # Risk metrics
                profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
            else:
                win_rate = 0
                total_pnl = 0
                avg_win = 0
                avg_loss = 0
                avg_trade = 0
                profit_factor = 0
            
            # Drawdown calculation
            df_portfolio['peak'] = df_portfolio['total_value'].cummax()
            df_portfolio['drawdown'] = (df_portfolio['total_value'] - df_portfolio['peak']) / df_portfolio['peak']
            max_drawdown = df_portfolio['drawdown'].min()
            
            # Volatility and Sharpe ratio
            df_portfolio['daily_return'] = df_portfolio['total_value'].pct_change()
            volatility = df_portfolio['daily_return'].std() * np.sqrt(252)  # Annualized
            excess_return = total_return - 0.05  # Assuming 5% risk-free rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Time metrics
            start_date = df_portfolio['date'].iloc[0]
            end_date = df_portfolio['date'].iloc[-1]
            trading_days = len(df_portfolio)
            
            # Buy and hold comparison
            first_price = df_portfolio['close_price'].iloc[0]
            last_price = df_portfolio['close_price'].iloc[-1]
            buy_hold_return = (last_price - first_price) / first_price
            
            metrics = {
                # Return metrics
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'buy_hold_return': buy_hold_return,
                'buy_hold_return_pct': buy_hold_return * 100,
                'excess_return': total_return - buy_hold_return,
                
                # Trade metrics
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'total_pnl': total_pnl,
                'avg_trade': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                
                # Risk metrics
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                
                # Time metrics
                'start_date': start_date,
                'end_date': end_date,
                'trading_days': trading_days,
                'avg_days_per_trade': trading_days / total_trades if total_trades > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades"""
        if not self.trades:
            return pd.DataFrame()
        
        trade_df = pd.DataFrame(self.trades)
        return trade_df
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns"""
        try:
            if not self.portfolio:
                return pd.DataFrame()
            
            df_portfolio = pd.DataFrame(self.portfolio)
            df_portfolio['date'] = pd.to_datetime(df_portfolio['date'])
            df_portfolio.set_index('date', inplace=True)
            
            # Resample to monthly and calculate returns
            monthly_values = df_portfolio['total_value'].resample('M').last()
            monthly_returns = monthly_values.pct_change().dropna()
            
            monthly_summary = pd.DataFrame({
                'month': monthly_returns.index.strftime('%Y-%m'),
                'return': monthly_returns.values,
                'return_pct': monthly_returns.values * 100,
                'cumulative_return': (1 + monthly_returns).cumprod() - 1
            })
            
            return monthly_summary
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {str(e)}")
            return pd.DataFrame()
    
    def export_results(self, filepath: str):
        """Export backtest results to Excel"""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Trade summary
                if self.trades:
                    trade_df = pd.DataFrame(self.trades)
                    trade_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Daily portfolio values
                if self.portfolio:
                    portfolio_df = pd.DataFrame(self.portfolio)
                    portfolio_df.to_excel(writer, sheet_name='Daily_Portfolio', index=False)
                
                # Monthly returns
                monthly_df = self.get_monthly_returns()
                if not monthly_df.empty:
                    monthly_df.to_excel(writer, sheet_name='Monthly_Returns', index=False)
            
            logger.info(f"Backtest results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
