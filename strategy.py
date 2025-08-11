"""
Trading Strategy Module
Implements the RSI + Moving Average crossover strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from indicators import add_all_indicators
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)

class TradingStrategy:
    """
    Implements RSI + Moving Average crossover strategy
    Buy Signal: RSI < 30 AND 20-DMA crossing above 50-DMA
    Sell Signal: RSI > 70 OR 20-DMA crossing below 50-DMA
    """
    
    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70, 
                 short_ma: int = 20, long_ma: int = 50):
        """
        Initialize strategy parameters
        
        Args:
            rsi_oversold: RSI level for oversold condition (default 30)
            rsi_overbought: RSI level for overbought condition (default 70)
            short_ma: Short moving average period (default 20)
            long_ma: Long moving average period (default 50)
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.short_ma = short_ma
        self.long_ma = long_ma
        
        logger.info(f"Strategy initialized with RSI({rsi_oversold}, {rsi_overbought}), MA({short_ma}, {long_ma})")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on strategy rules
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with added signal columns
        """
        try:
            logger.info("Generating trading signals")
            
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # Ensure we have required indicators
            if 'rsi' not in result_df.columns:
                result_df = add_all_indicators(result_df)
            
            # Initialize signal columns
            result_df['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
            result_df['position'] = 0  # Current position
            result_df['ma_crossover'] = 0  # MA crossover signal
            
            # Calculate MA crossover
            result_df['ma_crossover'] = np.where(
                (result_df['sma_20'] > result_df['sma_50']) & 
                (result_df['sma_20'].shift(1) <= result_df['sma_50'].shift(1)), 1,
                np.where(
                    (result_df['sma_20'] < result_df['sma_50']) & 
                    (result_df['sma_20'].shift(1) >= result_df['sma_50'].shift(1)), -1, 0
                )
            )
            
            # Generate buy signals
            buy_condition = (
                (result_df['rsi'] < self.rsi_oversold) &  # RSI oversold
                (result_df['ma_crossover'] == 1)  # 20-DMA crossing above 50-DMA
            )
            
            # Generate sell signals
            sell_condition = (
                (result_df['rsi'] > self.rsi_overbought) |  # RSI overbought
                (result_df['ma_crossover'] == -1)  # 20-DMA crossing below 50-DMA
            )
            
            # Apply signals
            result_df.loc[buy_condition, 'signal'] = 1
            result_df.loc[sell_condition, 'signal'] = -1
            
            # Calculate positions (forward fill signals)
            position = 0
            positions = []
            
            for i, signal in enumerate(result_df['signal']):
                if signal == 1 and position <= 0:  # Buy signal and not already long
                    position = 1
                elif signal == -1 and position >= 0:  # Sell signal and not already short
                    position = -1
                positions.append(position)
            
            result_df['position'] = positions
            
            # Add signal strength and confidence
            result_df['signal_strength'] = self._calculate_signal_strength(result_df)
            
            signal_count = len(result_df[result_df['signal'] != 0])
            logger.info(f"Generated {signal_count} signals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return df
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength based on multiple factors
        
        Args:
            df: DataFrame with indicators and signals
            
        Returns:
            Series with signal strength values (0-1)
        """
        try:
            strength = pd.Series(0.5, index=df.index)  # Default neutral strength
            
            # RSI contribution
            rsi_strength = np.where(
                df['rsi'] < 20, 1.0,  # Very oversold
                np.where(df['rsi'] < 30, 0.8,  # Oversold
                np.where(df['rsi'] > 80, 0.0,  # Very overbought
                np.where(df['rsi'] > 70, 0.2, 0.5)))  # Overbought
            )
            
            # MACD contribution
            macd_strength = np.where(
                (df['macd'] > df['macd_signal']) & (df['macd'] > 0), 0.8,
                np.where((df['macd'] > df['macd_signal']) & (df['macd'] < 0), 0.6,
                np.where((df['macd'] < df['macd_signal']) & (df['macd'] > 0), 0.4, 0.2))
            )
            
            # Volume contribution (if available)
            if 'volume' in df.columns:
                volume_ma = df['volume'].rolling(20).mean()
                volume_strength = np.where(df['volume'] > volume_ma * 1.5, 0.8, 0.5)
            else:
                volume_strength = 0.5
            
            # Combine strengths
            strength = (rsi_strength * 0.4 + macd_strength * 0.4 + volume_strength * 0.2)
            
            return pd.Series(strength, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return pd.Series(0.5, index=df.index)
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict:
        """
        Get the current trading signal for the latest data point
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Dictionary with current signal information
        """
        try:
            if df.empty:
                return {'signal': 0, 'action': 'HOLD', 'strength': 0.5, 'timestamp': None}
            
            latest = df.iloc[-1]
            
            signal_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
            
            return {
                'signal': int(latest['signal']),
                'action': signal_map.get(int(latest['signal']), 'HOLD'),
                'strength': float(latest.get('signal_strength', 0.5)),
                'rsi': float(latest.get('rsi', 0)),
                'sma_20': float(latest.get('sma_20', 0)),
                'sma_50': float(latest.get('sma_50', 0)),
                'price': float(latest['close']),
                'timestamp': latest.name,
                'position': int(latest.get('position', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting current signal: {str(e)}")
            return {'signal': 0, 'action': 'HOLD', 'strength': 0.5, 'timestamp': None}
    
    def analyze_signals(self, df: pd.DataFrame) -> Dict:
        """
        Analyze generated signals and provide statistics
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Dictionary with signal analysis
        """
        try:
            buy_signals = len(df[df['signal'] == 1])
            sell_signals = len(df[df['signal'] == -1])
            total_signals = buy_signals + sell_signals
            
            # Calculate signal frequency
            if len(df) > 0:
                signal_frequency = total_signals / len(df) * 100
            else:
                signal_frequency = 0
            
            # Average signal strength
            signal_rows = df[df['signal'] != 0]
            avg_strength = signal_rows['signal_strength'].mean() if not signal_rows.empty else 0
            
            analysis = {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'signal_frequency_pct': round(signal_frequency, 2),
                'avg_signal_strength': round(avg_strength, 3),
                'data_points': len(df),
                'date_range': {
                    'start': df.index[0] if not df.empty else None,
                    'end': df.index[-1] if not df.empty else None
                }
            }
            
            logger.info(f"Signal analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing signals: {str(e)}")
            return {}

def run_strategy_on_multiple_stocks(stock_data: Dict[str, pd.DataFrame], 
                                   strategy: TradingStrategy = None) -> Dict[str, pd.DataFrame]:
    """
    Run strategy on multiple stocks
    
    Args:
        stock_data: Dictionary with stock symbol as key and DataFrame as value
        strategy: TradingStrategy instance. If None, uses default parameters
        
    Returns:
        Dictionary with stock symbol as key and DataFrame with signals as value
    """
    if strategy is None:
        strategy = TradingStrategy()
    
    results = {}
    
    for symbol, df in stock_data.items():
        try:
            logger.info(f"Running strategy on {symbol}")
            
            # Add indicators and generate signals
            df_with_indicators = add_all_indicators(df)
            df_with_signals = strategy.generate_signals(df_with_indicators)
            
            results[symbol] = df_with_signals
            
            # Log signal analysis
            analysis = strategy.analyze_signals(df_with_signals)
            logger.info(f"{symbol} analysis: {analysis}")
            
        except Exception as e:
            logger.error(f"Error running strategy on {symbol}: {str(e)}")
            continue
    
    return results

# Example usage and testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    
    print("Testing trading strategy...")
    
    # Fetch sample data
    fetcher = DataFetcher()
    stock_data = fetcher.fetch_multiple_stocks(period="6mo")
    
    if stock_data:
        # Initialize strategy
        strategy = TradingStrategy()
        
        # Run strategy on all stocks
        results = run_strategy_on_multiple_stocks(stock_data, strategy)
        
        # Display results for each stock
        for symbol, df_with_signals in results.items():
            print(f"\n=== {symbol} ===")
            
            # Current signal
            current_signal = strategy.get_current_signal(df_with_signals)
            print(f"Current Signal: {current_signal}")
            
            # Signal analysis
            analysis = strategy.analyze_signals(df_with_signals)
            print(f"Signal Analysis: {analysis}")
            
            # Recent signals
            recent_signals = df_with_signals[df_with_signals['signal'] != 0].tail()
            if not recent_signals.empty:
                print(f"Recent Signals:")
                for idx, row in recent_signals.iterrows():
                    action = 'BUY' if row['signal'] == 1 else 'SELL'
                    print(f"  {idx.date()}: {action} at {row['close']:.2f} (RSI: {row['rsi']:.1f})")
    else:
        print("No stock data available for testing")
