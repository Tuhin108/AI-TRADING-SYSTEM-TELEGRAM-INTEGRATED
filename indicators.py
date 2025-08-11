"""
Technical Indicators Module
Implements RSI, SMA, MACD and other technical indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)

class TechnicalIndicators:
    """Class containing various technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data: Price series (typically closing prices)
            window: Period for RSI calculation (default 14)
            
        Returns:
            Series with RSI values
        """
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            logger.debug(f"Calculated RSI with window {window}")
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            data: Price series
            window: Period for SMA calculation
            
        Returns:
            Series with SMA values
        """
        try:
            sma = data.rolling(window=window).mean()
            logger.debug(f"Calculated SMA with window {window}")
            return sma
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            data: Price series
            window: Period for EMA calculation
            
        Returns:
            Series with EMA values
        """
        try:
            ema = data.ewm(span=window).mean()
            logger.debug(f"Calculated EMA with window {window}")
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(dtype=float)
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        try:
            ema_fast = TechnicalIndicators.calculate_ema(data, fast)
            ema_slow = TechnicalIndicators.calculate_ema(data, slow)
            
            macd_line = ema_fast - ema_slow
            signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            logger.debug(f"Calculated MACD with parameters fast={fast}, slow={slow}, signal={signal}")
            return macd_line, signal_line, histogram
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price series
            window: Period for moving average (default 20)
            num_std: Number of standard deviations (default 2)
            
        Returns:
            Tuple of (Upper band, Middle band/SMA, Lower band)
        """
        try:
            sma = TechnicalIndicators.calculate_sma(data, window)
            std = data.rolling(window=window).std()
            
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            logger.debug(f"Calculated Bollinger Bands with window {window} and {num_std} std")
            return upper_band, sma, lower_band
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_window: %K period (default 14)
            d_window: %D period (default 3)
            
        Returns:
            Tuple of (%K, %D)
        """
        try:
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_window).mean()
            
            logger.debug(f"Calculated Stochastic with K={k_window}, D={d_window}")
            return k_percent, d_percent
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for ATR calculation (default 14)
            
        Returns:
            Series with ATR values
        """
        try:
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            
            logger.debug(f"Calculated ATR with window {window}")
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(dtype=float)

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    try:
        logger.info("Adding all technical indicators to DataFrame")
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # RSI
        result_df['rsi'] = TechnicalIndicators.calculate_rsi(result_df['close'])
        
        # Moving Averages
        result_df['sma_20'] = TechnicalIndicators.calculate_sma(result_df['close'], 20)
        result_df['sma_50'] = TechnicalIndicators.calculate_sma(result_df['close'], 50)
        result_df['ema_12'] = TechnicalIndicators.calculate_ema(result_df['close'], 12)
        result_df['ema_26'] = TechnicalIndicators.calculate_ema(result_df['close'], 26)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.calculate_macd(result_df['close'])
        result_df['macd'] = macd
        result_df['macd_signal'] = signal
        result_df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(result_df['close'])
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle
        result_df['bb_lower'] = bb_lower
        
        # Stochastic
        if all(col in result_df.columns for col in ['high', 'low']):
            stoch_k, stoch_d = TechnicalIndicators.calculate_stochastic(
                result_df['high'], result_df['low'], result_df['close']
            )
            result_df['stoch_k'] = stoch_k
            result_df['stoch_d'] = stoch_d
            
            # ATR
            result_df['atr'] = TechnicalIndicators.calculate_atr(
                result_df['high'], result_df['low'], result_df['close']
            )
        
        logger.info(f"Successfully added indicators. DataFrame shape: {result_df.shape}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error adding indicators: {str(e)}")
        return df

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    volume = np.random.randint(1000, 10000, 100)
    
    sample_df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    print("Testing technical indicators...")
    
    # Test individual indicators
    rsi = TechnicalIndicators.calculate_rsi(sample_df['close'])
    sma_20 = TechnicalIndicators.calculate_sma(sample_df['close'], 20)
    macd, signal, histogram = TechnicalIndicators.calculate_macd(sample_df['close'])
    
    print(f"RSI (last 5): {rsi.tail().values}")
    print(f"SMA 20 (last 5): {sma_20.tail().values}")
    print(f"MACD (last 5): {macd.tail().values}")
    
    # Test adding all indicators
    df_with_indicators = add_all_indicators(sample_df)
    print(f"\nDataFrame with indicators shape: {df_with_indicators.shape}")
    print(f"Columns: {list(df_with_indicators.columns)}")
