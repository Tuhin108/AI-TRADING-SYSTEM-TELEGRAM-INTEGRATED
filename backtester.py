"""
Backtesting Module
Implements backtesting logic for the trading strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)

@dataclass
class Trade:
    """Data class to represent a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    trade_type: str  # 'BUY' or 'SELL'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_days: Optional[int] = None
    is_open: bool = True

class Backtester:
    """
    Backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001, 
                 position_size: float = 0.1):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital in INR
            commission: Commission rate (0.001 = 0.1%)
            position_size: Position size as fraction of capital (0.1 = 10%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        self.reset()
        
        logger.info(f"Backtester initialized with capital: Rs.{initial_capital:,.2f}, "
                   f"commission: {commission*100:.3f}%, position size: {position_size*100:.1f}%")
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.trades = []
        self.portfolio_value = []
        self.positions = {}  # symbol -> Trade object for open positions
        self.equity_curve = pd.DataFrame()
        
    def calculate_position_size(self, price: float) -> int:
        """
        Calculate position size based on available capital
        
        Args:
            price: Current stock price
            
        Returns:
            Number of shares to trade
        """
        try:
            position_value = self.capital * self.position_size
            quantity = int(position_value / price)
            return max(1, quantity)  # Minimum 1 share
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1
    
    def execute_trade(self, symbol: str, date: datetime, price: float, 
                     signal: int, quantity: Optional[int] = None) -> Optional[Trade]:
        """
        Execute a trade based on signal
        
        Args:
            symbol: Stock symbol
            date: Trade date
            price: Trade price
            signal: 1 for buy, -1 for sell
            quantity: Number of shares (if None, calculated automatically)
            
        Returns:
            Trade object if executed, None otherwise
        """
        try:
            if quantity is None:
                quantity = self.calculate_position_size(price)
            
            commission_cost = price * quantity * self.commission
            
            if signal == 1:  # Buy signal
                total_cost = (price * quantity) + commission_cost
                
                # Check if we have enough capital
                if total_cost > self.capital:
                    logger.warning(f"Insufficient capital for {symbol} buy: need Rs.{total_cost:.2f}, have Rs.{self.capital:.2f}")
                    return None
                
                # Check if we already have a position
                if symbol in self.positions:
                    logger.warning(f"Already have open position in {symbol}")
                    return None
                
                # Execute buy
                self.capital -= total_cost
                
                trade = Trade(
                    symbol=symbol,
                    entry_date=date,
                    exit_date=None,
                    entry_price=price,
                    exit_price=None,
                    quantity=quantity,
                    trade_type='BUY'
                )
                
                self.positions[symbol] = trade
                logger.info(f"BUY {quantity} shares of {symbol} at Rs.{price:.2f} on {date.date()}")
                
                return trade
                
            elif signal == -1:  # Sell signal
                # Check if we have a position to sell
                if symbol not in self.positions:
                    logger.warning(f"No open position in {symbol} to sell")
                    return None
                
                # Close the position
                open_trade = self.positions[symbol]
                total_proceeds = (price * open_trade.quantity) - commission_cost
                
                # Calculate P&L
                total_cost = open_trade.entry_price * open_trade.quantity
                pnl = total_proceeds - total_cost
                pnl_pct = (pnl / total_cost) * 100
                
                # Update capital
                self.capital += total_proceeds
                
                # Complete the trade
                open_trade.exit_date = date
                open_trade.exit_price = price
                open_trade.pnl = pnl
                open_trade.pnl_pct = pnl_pct
                open_trade.duration_days = (date - open_trade.entry_date).days
                open_trade.is_open = False
                
                # Move to completed trades and remove from positions
                self.trades.append(open_trade)
                del self.positions[symbol]
                
                logger.info(f"SELL {open_trade.quantity} shares of {symbol} at Rs.{price:.2f} on {date.date()}, P&L: Rs.{pnl:.2f} ({pnl_pct:.2f}%)")
                
                return open_trade
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return None
    
    def run_backtest(self, df_with_signals: pd.DataFrame, symbol: str) -> Dict:
        """
        Run backtest on a single stock with signals
        
        Args:
            df_with_signals: DataFrame with OHLCV data and signals
            symbol: Stock symbol
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info(f"Running backtest for {symbol}")
            
            portfolio_values = []
            dates = []
            
            for date, row in df_with_signals.iterrows():
                # Execute trades based on signals
                if row['signal'] != 0:
                    self.execute_trade(symbol, date, row['close'], int(row['signal']))
                
                # Calculate current portfolio value
                current_value = self.capital
                
                # Add value of open positions
                for pos_symbol, trade in self.positions.items():
                    if pos_symbol == symbol:  # Only count current symbol's position
                        current_value += trade.quantity * row['close']
                
                portfolio_values.append(current_value)
                dates.append(date)
            
            # Close any remaining open positions at the last price
            if symbol in self.positions:
                last_price = df_with_signals['close'].iloc[-1]
                last_date = df_with_signals.index[-1]
                self.execute_trade(symbol, last_date, last_price, -1)
            
            # Create equity curve
            self.equity_curve = pd.DataFrame({
                'portfolio_value': portfolio_values,
                'returns': pd.Series(portfolio_values).pct_change()
            }, index=dates)
            
            # Calculate performance metrics
            results = self.calculate_performance_metrics(symbol)
            
            logger.info(f"Backtest completed for {symbol}: {len(self.trades)} trades, "
                       f"Final value: Rs.{portfolio_values[-1]:,.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {str(e)}")
            return {}
    
    def calculate_performance_metrics(self, symbol: str) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.trades and not self.equity_curve.empty:
                final_value = self.equity_curve['portfolio_value'].iloc[-1]
            elif self.trades:
                final_value = self.capital + sum(trade.quantity * trade.exit_price 
                                               for trade in self.trades if not trade.is_open)
            else:
                final_value = self.initial_capital
            
            # Basic metrics
            total_return = final_value - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            # Trade statistics
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            total_trades = len(self.trades)
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            # Risk metrics
            if not self.equity_curve.empty and len(self.equity_curve) > 1:
                returns = self.equity_curve['returns'].dropna()
                
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                    
                    # Maximum drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min() * 100
                else:
                    volatility = 0
                    sharpe_ratio = 0
                    max_drawdown = 0
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            metrics = {
                'symbol': symbol,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': round(total_return_pct, 2),
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': round(win_rate, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
                'max_drawdown': round(max_drawdown, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'trades': self.trades.copy()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade log as DataFrame
        
        Returns:
            DataFrame with trade details
        """
        try:
            if not self.trades:
                return pd.DataFrame()
            
            trade_data = []
            for trade in self.trades:
                trade_data.append({
                    'symbol': trade.symbol,
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'trade_type': trade.trade_type,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'duration_days': trade.duration_days
                })
            
            return pd.DataFrame(trade_data)
            
        except Exception as e:
            logger.error(f"Error creating trade log: {str(e)}")
            return pd.DataFrame()

def run_backtest_multiple_stocks(stock_data_with_signals: Dict[str, pd.DataFrame], 
                                initial_capital: float = 100000) -> Dict:
    """
    Run backtest on multiple stocks
    
    Args:
        stock_data_with_signals: Dictionary with stock data and signals
        initial_capital: Initial capital for backtesting
        
    Returns:
        Dictionary with combined results
    """
    try:
        logger.info(f"Running backtest on {len(stock_data_with_signals)} stocks")
        
        all_results = {}
        all_trades = []
        
        # Run backtest for each stock separately
        for symbol, df in stock_data_with_signals.items():
            backtester = Backtester(initial_capital=initial_capital)
            results = backtester.run_backtest(df, symbol)
            
            if results:
                all_results[symbol] = results
                all_trades.extend(results.get('trades', []))
        
        # Calculate combined metrics
        if all_results:
            combined_metrics = calculate_combined_metrics(all_results, initial_capital)
            combined_metrics['individual_results'] = all_results
            combined_metrics['all_trades'] = all_trades
            
            return combined_metrics
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error running multi-stock backtest: {str(e)}")
        return {}

def calculate_combined_metrics(individual_results: Dict, initial_capital: float) -> Dict:
    """
    Calculate combined metrics across all stocks
    
    Args:
        individual_results: Dictionary with individual stock results
        initial_capital: Initial capital
        
    Returns:
        Dictionary with combined metrics
    """
    try:
        total_trades = sum(result['total_trades'] for result in individual_results.values())
        total_winning = sum(result['winning_trades'] for result in individual_results.values())
        total_losing = sum(result['losing_trades'] for result in individual_results.values())
        
        # Calculate average metrics
        avg_return_pct = np.mean([result['total_return_pct'] for result in individual_results.values()])
        avg_win_rate = np.mean([result['win_rate'] for result in individual_results.values()])
        avg_sharpe = np.mean([result['sharpe_ratio'] for result in individual_results.values() 
                             if isinstance(result['sharpe_ratio'], (int, float))])
        
        combined = {
            'total_stocks': len(individual_results),
            'total_trades': total_trades,
            'total_winning_trades': total_winning,
            'total_losing_trades': total_losing,
            'overall_win_rate': round((total_winning / total_trades * 100) if total_trades > 0 else 0, 2),
            'avg_return_pct': round(avg_return_pct, 2),
            'avg_win_rate': round(avg_win_rate, 2),
            'avg_sharpe_ratio': round(avg_sharpe, 3),
            'best_performer': max(individual_results.items(), key=lambda x: x[1]['total_return_pct'])[0],
            'worst_performer': min(individual_results.items(), key=lambda x: x[1]['total_return_pct'])[0]
        }
        
        return combined
        
    except Exception as e:
        logger.error(f"Error calculating combined metrics: {str(e)}")
        return {}

# Example usage and testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from strategy import TradingStrategy, run_strategy_on_multiple_stocks
    
    print("Testing backtester...")
    
    # Fetch data and generate signals
    fetcher = DataFetcher()
    stock_data = fetcher.fetch_multiple_stocks(period="6mo")
    
    if stock_data:
        # Generate signals
        strategy = TradingStrategy()
        stock_data_with_signals = run_strategy_on_multiple_stocks(stock_data, strategy)
        
        # Run backtest
        backtest_results = run_backtest_multiple_stocks(stock_data_with_signals)
        
        if backtest_results:
            print(f"\n=== BACKTEST RESULTS ===")
            print(f"Total Stocks: {backtest_results['total_stocks']}")
            print(f"Total Trades: {backtest_results['total_trades']}")
            print(f"Overall Win Rate: {backtest_results['overall_win_rate']}%")
            print(f"Average Return: {backtest_results['avg_return_pct']}%")
            print(f"Best Performer: {backtest_results['best_performer']}")
            print(f"Worst Performer: {backtest_results['worst_performer']}")
            
            # Show individual results
            for symbol, results in backtest_results['individual_results'].items():
                print(f"\n{symbol}:")
                print(f"  Return: {results['total_return_pct']}%")
                print(f"  Trades: {results['total_trades']}")
                print(f"  Win Rate: {results['win_rate']}%")
                print(f"  Sharpe Ratio: {results['sharpe_ratio']}")
        else:
            print("No backtest results generated")
    else:
        print("No stock data available for testing")
