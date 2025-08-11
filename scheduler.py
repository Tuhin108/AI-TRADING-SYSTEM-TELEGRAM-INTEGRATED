"""
Scheduler Module
Handles automated scheduling and execution of the trading system
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
import threading
from typing import Optional, Callable
import signal
import sys
from main import AlgoTradingSystem
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)

class TradingScheduler:
    """
    Scheduler for automated trading system execution
    """
    
    def __init__(self):
        """Initialize the trading scheduler"""
        self.system = AlgoTradingSystem()
        self.is_running = False
        self.scheduler_thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Trading Scheduler initialized")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def market_hours_check(self) -> bool:
        """
        Check if current time is within market hours
        NSE trading hours: 9:15 AM to 3:30 PM IST (Monday to Friday)
        
        Returns:
            True if within market hours, False otherwise
        """
        try:
            now = datetime.now()
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Market hours: 9:15 AM to 3:30 PM
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            logger.error(f"Error checking market hours: {str(e)}")
            return False
    
    def run_analysis_job(self):
        """Job function to run the trading analysis"""
        try:
            logger.info("Starting scheduled analysis job...")
            
            # Check if market is open
            if not self.market_hours_check():
                logger.info("Market is closed, skipping analysis")
                return
            
            # Run the analysis
            results = self.system.run_full_analysis()
            
            if results:
                logger.info("Scheduled analysis completed successfully")
                
                # Log daily performance if it's end of day
                now = datetime.now()
                if now.hour >= 15 and now.minute >= 30:  # After market close
                    self.log_daily_performance(results)
            else:
                logger.error("Scheduled analysis failed")
                
        except Exception as e:
            logger.error(f"Error in scheduled analysis job: {str(e)}")
            if self.system.telegram_alerter:
                self.system.telegram_alerter.send_error_alert(str(e), "Scheduler")
    
    def log_daily_performance(self, results: Dict):
        """
        Log daily performance summary
        
        Args:
            results: Analysis results dictionary
        """
        try:
            if not self.system.sheets_logger.client:
                return
            
            # Calculate daily performance metrics
            backtest_results = results.get('backtest', {})
            
            if backtest_results:
                performance_data = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'portfolio_value': 100000,  # This would be calculated from actual portfolio
                    'daily_return_pct': backtest_results.get('avg_return_pct', 0),
                    'cumulative_return_pct': backtest_results.get('avg_return_pct', 0),
                    'active_positions': len(results.get('signals', {})),
                    'cash_balance': 100000  # This would be calculated from actual portfolio
                }
                
                self.system.sheets_logger.log_daily_performance(performance_data)
                
                # Send daily summary via Telegram
                if self.system.telegram_alerter:
                    summary_data = {
                        'portfolio_value': performance_data['portfolio_value'],
                        'daily_return_pct': performance_data['daily_return_pct'],
                        'total_return_pct': performance_data['cumulative_return_pct'],
                        'trades_today': backtest_results.get('total_trades', 0),
                        'active_positions': performance_data['active_positions'],
                        'cash_balance': performance_data['cash_balance'],
                        'date': performance_data['date']
                    }
                    self.system.telegram_alerter.send_daily_summary(summary_data)
                
                logger.info("Daily performance logged successfully")
            
        except Exception as e:
            logger.error(f"Error logging daily performance: {str(e)}")
    
    def setup_schedules(self):
        """Setup all scheduled jobs"""
        try:
            # Clear any existing schedules
            schedule.clear()
            
            # Market opening analysis (9:20 AM)
            schedule.every().monday.at("09:20").do(self.run_analysis_job)
            schedule.every().tuesday.at("09:20").do(self.run_analysis_job)
            schedule.every().wednesday.at("09:20").do(self.run_analysis_job)
            schedule.every().thursday.at("09:20").do(self.run_analysis_job)
            schedule.every().friday.at("09:20").do(self.run_analysis_job)
            
            # Mid-day analysis (12:00 PM)
            schedule.every().monday.at("12:00").do(self.run_analysis_job)
            schedule.every().tuesday.at("12:00").do(self.run_analysis_job)
            schedule.every().wednesday.at("12:00").do(self.run_analysis_job)
            schedule.every().thursday.at("12:00").do(self.run_analysis_job)
            schedule.every().friday.at("12:00").do(self.run_analysis_job)
            
            # End of day analysis (3:25 PM)
            schedule.every().monday.at("15:25").do(self.run_analysis_job)
            schedule.every().tuesday.at("15:25").do(self.run_analysis_job)
            schedule.every().wednesday.at("15:25").do(self.run_analysis_job)
            schedule.every().thursday.at("15:25").do(self.run_analysis_job)
            schedule.every().friday.at("15:25").do(self.run_analysis_job)
            
            # Weekend ML model retraining (Saturday 10:00 AM)
            schedule.every().saturday.at("10:00").do(self.retrain_models_job)
            
            logger.info("Scheduled jobs setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up schedules: {str(e)}")
    
    def retrain_models_job(self):
        """Job to retrain ML models on weekends"""
        try:
            logger.info("Starting weekend ML model retraining...")
            
            # Fetch fresh data and retrain models
            ml_results = self.system.train_ml_models()
            
            if ml_results:
                logger.info("ML models retrained successfully")
                
                # Send summary via Telegram
                if self.system.telegram_alerter:
                    message = f"""
🤖 <b>ML MODELS RETRAINED</b> 🤖

📊 <b>Models Updated:</b> {len(ml_results)}
🕐 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Models are ready for next week's trading.

#MLRetraining #WeekendUpdate
                    """.strip()
                    
                    self.system.telegram_alerter.send_message_sync(message)
            else:
                logger.error("ML model retraining failed")
                
        except Exception as e:
            logger.error(f"Error in ML retraining job: {str(e)}")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        try:
            logger.info("Starting scheduler thread...")
            
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Error in scheduler thread: {str(e)}")
    
    def start(self):
        """Start the scheduler"""
        try:
            if self.is_running:
                logger.warning("Scheduler is already running")
                return
            
            logger.info("Starting Trading Scheduler...")
            
            # Setup schedules
            self.setup_schedules()
            
            # Start scheduler in separate thread
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            # Send startup notification
            if self.system.telegram_alerter:
                message = f"""
⏰ <b>SCHEDULER STARTED</b> ⏰

✅ Automated trading system is now running
📅 <b>Next Jobs:</b>
• Market Open: 9:20 AM (Mon-Fri)
• Mid-day: 12:00 PM (Mon-Fri)  
• End of Day: 3:25 PM (Mon-Fri)
• ML Retraining: 10:00 AM (Saturday)

🕐 <b>Started:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#SchedulerStarted #Automation
                """.strip()
                
                self.system.telegram_alerter.send_message_sync(message)
            
            logger.info("Trading Scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {str(e)}")
    
    def stop(self):
        """Stop the scheduler"""
        try:
            if not self.is_running:
                logger.warning("Scheduler is not running")
                return
            
            logger.info("Stopping Trading Scheduler...")
            
            self.is_running = False
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            # Clear schedules
            schedule.clear()
            
            logger.info("Trading Scheduler stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {str(e)}")
    
    def status(self) -> Dict:
        """
        Get scheduler status
        
        Returns:
            Dictionary with scheduler status information
        """
        try:
            next_jobs = []
            for job in schedule.jobs:
                next_jobs.append({
                    'job': str(job.job_func.__name__),
                    'next_run': job.next_run.strftime('%Y-%m-%d %H:%M:%S') if job.next_run else 'N/A'
                })
            
            status_info = {
                'is_running': self.is_running,
                'market_open': self.market_hours_check(),
                'total_jobs': len(schedule.jobs),
                'next_jobs': next_jobs,
                'last_analysis': self.system.last_run_time.strftime('%Y-%m-%d %H:%M:%S') if self.system.last_run_time else 'Never'
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting scheduler status: {str(e)}")
            return {}

def main():
    """Main function to run the scheduler"""
    try:
        print("⏰ Starting Algo Trading Scheduler...")
        
        # Create and start scheduler
        scheduler = TradingScheduler()
        scheduler.start()
        
        # Print status
        status = scheduler.status()
        print(f"\n📊 SCHEDULER STATUS:")
        print(f"Running: {status.get('is_running', False)}")
        print(f"Market Open: {status.get('market_open', False)}")
        print(f"Total Jobs: {status.get('total_jobs', 0)}")
        print(f"Last Analysis: {status.get('last_analysis', 'Never')}")
        
        if status.get('next_jobs'):
            print(f"\n📅 UPCOMING JOBS:")
            for job in status['next_jobs'][:5]:  # Show next 5 jobs
                print(f"  {job['job']}: {job['next_run']}")
        
        print(f"\n✅ Scheduler is running. Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(3600)  # Sleep for 1 hour
        except KeyboardInterrupt:
            print(f"\n⏹️ Stopping scheduler...")
            scheduler.stop()
            print("✅ Scheduler stopped successfully!")
            
    except Exception as e:
        print(f"❌ Scheduler error: {str(e)}")
        logger.error(f"Scheduler error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
