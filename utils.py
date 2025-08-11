"""
Utilities Module
Contains logging setup and helper functions
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
from pathlib import Path

def setup_logging(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set default log file if not provided
    if log_file is None:
        log_file = log_dir / f"algo_trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_file: str = ".env") -> Dict[str, Any]:
    """
    Load configuration from environment file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configuration values
    """
    config = {}
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip().strip('"\'')
        
        # Also load from environment variables
        config.update({
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
            'GOOGLE_CREDENTIALS_FILE': os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json'),
            'SPREADSHEET_NAME': os.getenv('SPREADSHEET_NAME', 'Algo Trading Log'),
            'INITIAL_CAPITAL': float(os.getenv('INITIAL_CAPITAL', '100000')),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
        })
        
    except Exception as e:
        print(f"Error loading config: {str(e)}")
    
    return config

def format_currency(amount: float, currency: str = "INR") -> str:
    """
    Format currency amount for display
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "INR":
        return f"Rs.{amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def create_directory(directory_path: str) -> bool:
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False

def save_json(data: Dict, filepath: str) -> bool:
    """
    Save dictionary as JSON file
    
    Args:
        data: Dictionary to save
        filepath: File path to save to
        
    Returns:
        True if saved successfully
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {str(e)}")
        return False

def load_json(filepath: str) -> Optional[Dict]:
    """
    Load JSON file as dictionary
    
    Args:
        filepath: File path to load from
        
    Returns:
        Dictionary or None if failed
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {str(e)}")
        return None

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid format
    """
    try:
        # Basic validation for NSE symbols
        if not symbol:
            return False
        
        # Should end with .NS for NSE stocks
        if not symbol.endswith('.NS'):
            return False
        
        # Should have at least one character before .NS
        base_symbol = symbol.replace('.NS', '')
        if len(base_symbol) < 1:
            return False
        
        return True
        
    except Exception:
        return False

def get_market_status() -> Dict[str, Any]:
    """
    Get current market status
    
    Returns:
        Dictionary with market status information
    """
    try:
        now = datetime.now()
        
        # Check if it's a weekday
        is_weekday = now.weekday() < 5
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_market_hours = market_open_time <= now <= market_close_time
        is_market_open = is_weekday and is_market_hours
        
        # Calculate time to next market event
        if is_market_open:
            time_to_close = market_close_time - now
            next_event = "Market Close"
            time_to_event = time_to_close
        elif is_weekday and now < market_open_time:
            time_to_open = market_open_time - now
            next_event = "Market Open"
            time_to_event = time_to_open
        else:
            # Calculate next market open
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and now > market_close_time:
                days_until_monday = 1
            
            next_market_day = now + timedelta(days=days_until_monday)
            next_market_open = next_market_day.replace(hour=9, minute=15, second=0, microsecond=0)
            
            time_to_event = next_market_open - now
            next_event = "Market Open"
        
        return {
            'is_market_open': is_market_open,
            'is_weekday': is_weekday,
            'is_market_hours': is_market_hours,
            'current_time': now,
            'market_open_time': market_open_time,
            'market_close_time': market_close_time,
            'next_event': next_event,
            'time_to_next_event': str(time_to_event).split('.')[0],  # Remove microseconds
            'market_status': 'OPEN' if is_market_open else 'CLOSED'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'is_market_open': False,
            'market_status': 'UNKNOWN'
        }

# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(__name__)
    logger.info("Testing logging setup")
    
    # Test config loading
    config = load_config()
    print(f"Config loaded: {len(config)} items")
    
    # Test market status
    market_status = get_market_status()
    print(f"Market Status: {market_status}")
    
    # Test utility functions
    print(f"Currency format: {format_currency(123456.78)}")
    print(f"Percentage change: {calculate_percentage_change(100, 110):.2f}%")
    print(f"Safe divide: {safe_divide(10, 0, -1)}")
    
    # Test symbol validation
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INVALID', '.NS', '']
    for symbol in test_symbols:
        print(f"Symbol {symbol} valid: {validate_stock_symbol(symbol)}")
