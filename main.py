from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import sys
from data_fetcher import DataFetcher
from strategy import TradingStrategy, run_strategy_on_multiple_stocks
from backtester import run_backtest_multiple_stocks
from ml_model import train_models_for_multiple_stocks, MLPredictor
from gsheet_webhook import GoogleSheetsWebhookLogger
from telegram_alerts import create_telegram_alerter
from utils import setup_logging, load_config
from gsheet_webhook import convert_backtest_results_to_sheets_format

# Setup logging
logger = setup_logging(__name__)

class AlgoTradingSystem:
    """
    Main algo-trading system orchestrator
    """
    
    def __init__(self, config_file: str = ".env"):
        """
        Initialize the algo-trading system
        
        Args:
            config_file: Path to configuration file
        """
        self.config = load_config(config_file)
        self.data_fetcher = DataFetcher()
        self.strategy = TradingStrategy()
        self.sheets_logger = GoogleSheetsWebhookLogger()
        self.telegram_alerter = create_telegram_alerter()
        
        # System state
        self.last_run_time = None
        self.stock_data = {}
        self.signals_data = {}
        self.ml_models = {}
        
        logger.info("Algo Trading System initialized")
        
        # Send startup notification
        if self.telegram_alerter:
            self.telegram_alerter.send_system_startup_alert()
    
    def fetch_market_data(self, symbols: List[str] = None, period: str = "6mo") -> Dict:
        """
        Fetch market data for analysis
        
        Args:
            symbols: List of stock symbols. If None, uses default stocks
            period: Data period to fetch
            
        Returns:
            Dictionary with stock data
        """
        try:
            logger.info("Fetching market data...")
            
            self.stock_data = self.data_fetcher.fetch_multiple_stocks(
                symbols=symbols, period=period
            )
            
            if self.stock_data:
                logger.info(f"Successfully fetched data for {len(self.stock_data)} stocks")
                return self.stock_data
            else:
                logger.error("Failed to fetch market data")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            if self.telegram_alerter:
                self.telegram_alerter.send_error_alert(str(e), "Data Fetcher")
            return {}
    
    def generate_trading_signals(self) -> Dict:
        """
        Generate trading signals using the strategy
        
        Returns:
            Dictionary with signals data
        """
        try:
            logger.info("Generating trading signals...")
            
            if not self.stock_data:
                logger.error("No stock data available for signal generation")
                return {}
            
            self.signals_data = run_strategy_on_multiple_stocks(
                self.stock_data, self.strategy
            )
            
            # Send alerts for new signals
            self.send_signal_alerts()
            
            logger.info(f"Generated signals for {len(self.signals_data)} stocks")
            return self.signals_data
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            if self.telegram_alerter:
                self.telegram_alerter.send_error_alert(str(e), "Signal Generator")
            return {}
    
    def send_signal_alerts(self):
        """Send Telegram alerts for new trading signals"""
        try:
            if not self.telegram_alerter:
                return
            
            for symbol, df in self.signals_data.items():
                current_signal = self.strategy.get_current_signal(df)
                
                if current_signal['signal'] != 0:  # Non-zero signal
                    trade_data = {
                        'symbol': symbol,
                        'action': current_signal['action'],
                        'price': current_signal['price'],
                        'rsi': current_signal['rsi'],
                        'signal_strength': current_signal['strength'],
                        'timestamp': current_signal['timestamp']
                    }
                    
                    self.telegram_alerter.send_trade_alert(trade_data)
                    
        except Exception as e:
            logger.error(f"Error sending signal alerts: {str(e)}")
    
    def run_backtest(self, initial_capital: float = 100000) -> Dict:
        """
        Run backtesting on the generated signals
        
        Args:
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info("Running backtest...")
            
            if not self.signals_data:
                logger.error("No signals data available for backtesting")
                return {}
            
            backtest_results = run_backtest_multiple_stocks(
                self.signals_data, initial_capital
            )
            
            if backtest_results:
                logger.info("Backtest completed successfully")
                
                # Log results to Google Sheets
                self.log_backtest_results(backtest_results)
                
                # Send Telegram summary
                if self.telegram_alerter:
                    self.telegram_alerter.send_backtest_summary(backtest_results)
                
                return backtest_results
            else:
                logger.error("Backtest failed")
                return {}
                
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            if self.telegram_alerter:
                self.telegram_alerter.send_error_alert(str(e), "Backtester")
            return {}
    
    def train_ml_models(self) -> Dict:
        """
        Train ML models for prediction
        
        Returns:
            Dictionary with ML training results
        """
        try:
            logger.info("Training ML models...")
            
            if not self.stock_data:
                logger.error("No stock data available for ML training")
                return {}
            
            # Use longer period data for ML training
            ml_data = self.data_fetcher.fetch_multiple_stocks(period="1y")
            
            if ml_data:
                ml_results = train_models_for_multiple_stocks(ml_data)
                self.ml_models = ml_results
                
                # Log ML predictions
                self.log_ml_predictions(ml_results)
                
                logger.info(f"Trained ML models for {len(ml_results)} stocks")
                return ml_results
            else:
                logger.error("Failed to fetch data for ML training")
                return {}
                
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
            if self.telegram_alerter:
                self.telegram_alerter.send_error_alert(str(e), "ML Trainer")
            return {}
    
    def log_backtest_results(self, backtest_results: Dict):
        """
        Log backtest results to Google Sheets
        
        Args:
            backtest_results: Backtest results dictionary
        """
        try:
            # No client check for webhook logger
            
            # Convert results to sheets format
            sheets_data = convert_backtest_results_to_sheets_format(backtest_results)
            
            # Log trades
            if sheets_data['trades']:
                self.sheets_logger.log_multiple_trades(sheets_data['trades'])
            
            # Update summary
            if sheets_data['summary']:
                self.sheets_logger.update_summary(sheets_data['summary'])
            
            # Update win ratio
            if sheets_data['win_ratio']:
                self.sheets_logger.update_win_ratio(sheets_data['win_ratio'])
            
            logger.info("Backtest results logged to Google Sheets")
            
        except Exception as e:
            logger.error(f"Error logging backtest results: {str(e)}")
    
    def log_ml_predictions(self, ml_results: Dict):
        """
        Log ML predictions to Google Sheets
        
        Args:
            ml_results: ML training results dictionary
        """
        try:
            # No client check for webhook logger
            
            for symbol, results in ml_results.items():
                if 'latest_prediction' in results:
                    prediction = results['latest_prediction']
                    
                    prediction_data = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'prediction': prediction.get('prediction_label', ''),
                        'confidence': prediction.get('confidence', 0),
                        'actual_movement': '',  # To be filled later
                        'accuracy': 0,  # To be calculated later
                        'model_type': results.get('model_type', '')
                    }
                    
                    self.sheets_logger.log_ml_prediction(prediction_data)
                    
                    # Send Telegram alert
                    if self.telegram_alerter:
                        pred_alert_data = {
                            'symbol': symbol,
                            'prediction_label': prediction.get('prediction_label', ''),
                            'confidence': prediction.get('confidence', 0),
                            'model_type': results.get('model_type', ''),
                            'timestamp': datetime.now()
                        }
                        self.telegram_alerter.send_ml_prediction_alert(pred_alert_data)
            
            logger.info("ML predictions logged to Google Sheets")
            
        except Exception as e:
            logger.error(f"Error logging ML predictions: {str(e)}")
    
    def run_full_analysis(self, symbols: List[str] = None) -> Dict:
        """
        Run complete analysis pipeline
        
        Args:
            symbols: List of stock symbols to analyze
            
        Returns:
            Dictionary with all analysis results
        """
        try:
            logger.info("Starting full analysis pipeline...")
            
            results = {
                'timestamp': datetime.now(),
                'stock_data': {},
                'signals': {},
                'backtest': {},
                'ml_results': {}
            }
            
            # Step 1: Fetch market data
            results['stock_data'] = self.fetch_market_data(symbols)
            
            if not results['stock_data']:
                logger.error("Cannot proceed without market data")
                return results
            
            # Step 2: Generate trading signals
            results['signals'] = self.generate_trading_signals()
            
            # Step 3: Run backtest
            if results['signals']:
                results['backtest'] = self.run_backtest()
            
            # Step 4: Train ML models
            results['ml_results'] = self.train_ml_models()
            
            # Update last run time
            self.last_run_time = datetime.now()
            
            logger.info("Full analysis pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis pipeline: {str(e)}")
            if self.telegram_alerter:
                self.telegram_alerter.send_error_alert(str(e), "Main Pipeline")
            return {}
    
    def print_summary_report(self, results: Dict):
        """
        Print a summary report of the analysis
        
        Args:
            results: Analysis results dictionary
        """
        try:
            print("\n" + "="*60)
            print("ALGO TRADING SYSTEM - ANALYSIS REPORT")
            print("="*60)
            
            print(f"Analysis Time: {results.get('timestamp', 'N/A')}")
            print(f"Stocks Analyzed: {len(results.get('stock_data', {}))}")
            
            # Stock data summary
            if results.get('stock_data'):
                print(f"\n📊 MARKET DATA:")
                for symbol, df in results['stock_data'].items():
                    latest_price = df['close'].iloc[-1]
                    price_change = df['close'].pct_change().iloc[-1] * 100
                    print(f"  {symbol}: Rs.{latest_price:.2f} ({price_change:+.2f}%)")
            
            # Signals summary
            if results.get('signals'):
                print(f"\n🎯 TRADING SIGNALS:")
                for symbol, df in results['signals'].items():
                    current_signal = self.strategy.get_current_signal(df)
                    print(f"  {symbol}: {current_signal['action']} "
                          f"(Strength: {current_signal['strength']:.2f}, RSI: {current_signal['rsi']:.1f})")
            
            # Backtest summary
            if results.get('backtest'):
                bt = results['backtest']
                print(f"\n📈 BACKTEST RESULTS:")
                print(f"  Total Trades: {bt.get('total_trades', 0)}")
                print(f"  Win Rate: {bt.get('overall_win_rate', 0):.1f}%")
                print(f"  Avg Return: {bt.get('avg_return_pct', 0):.2f}%")
                print(f"  Best Performer: {bt.get('best_performer', 'N/A')}")
            
            # ML results summary
            if results.get('ml_results'):
                print(f"\n🤖 ML PREDICTIONS:")
                for symbol, ml_result in results['ml_results'].items():
                    accuracy = ml_result.get('test_accuracy', 0)
                    if 'latest_prediction' in ml_result:
                        pred = ml_result['latest_prediction']
                        print(f"  {symbol}: {pred.get('prediction_label', 'N/A')} "
                              f"(Confidence: {pred.get('confidence', 0):.1%}, Accuracy: {accuracy:.1%})")
            
            # Google Sheets link
            # No client or get_spreadsheet_url for webhook logger
            
            print("\n" + "="*60)
            
        except Exception as e:
            logger.error(f"Error printing summary report: {str(e)}")

def main():
    """Main function to run the algo-trading system"""
    try:
        print("🚀 Starting Algo Trading System...")
        
        # Initialize system
        system = AlgoTradingSystem()
        
        # Run full analysis
        results = system.run_full_analysis()
        
        # Print summary report
        system.print_summary_report(results)
        
        print("\n✅ Analysis completed successfully!")
        print("Check Google Sheets and Telegram for detailed results.")
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
        logger.info("System stopped by user")
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        logger.error(f"System error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
