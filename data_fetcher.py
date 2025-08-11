"""
Data Fetcher Module
Handles fetching stock data from Yahoo Finance API for NIFTY 50 stocks
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)

class DataFetcher:
    def __init__(self):
        # NIFTY 50 stocks (using NSE symbols with .NS suffix for Yahoo Finance)
        self.nifty_50_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'AXISBANK.NS',
            'LT.NS', 'DMART.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS'
        ]
        
        # Default to first 3 stocks for the assignment
        self.default_stocks = self.nifty_50_stocks[:3]
        
    def fetch_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"Fetching data for {symbol} with period {period} and interval {interval}")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str] = None, period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols. If None, uses default stocks
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        if symbols is None:
            symbols = self.default_stocks
            
        stock_data = {}
        
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, period, interval)
            if data is not None:
                stock_data[symbol] = data
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
                
        logger.info(f"Successfully fetched data for {len(stock_data)} out of {len(symbols)} stocks")
        return stock_data
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest closing price or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1d")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Get basic stock information
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock info or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'INR')
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return None

# Example usage and testing
if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Test fetching data for default stocks
    print("Testing data fetcher...")
    
    # Fetch 6 months of data for backtesting
    stock_data = fetcher.fetch_multiple_stocks(period="6mo")
    
    for symbol, data in stock_data.items():
        print(f"\n{symbol}:")
        print(f"  Records: {len(data)}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Latest close: {data['close'].iloc[-1]:.2f}")
        
        # Get stock info
        info = fetcher.get_stock_info(symbol)
        if info:
            print(f"  Company: {info['name']}")
            print(f"  Sector: {info['sector']}")
