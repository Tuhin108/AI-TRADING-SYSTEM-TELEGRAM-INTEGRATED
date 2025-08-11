"""
Telegram Alerts Module (Bonus Feature)
Sends trading alerts and notifications via Telegram

This version:
- Attempts to load .env if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are missing
- Handles both async and sync Bot.send_message implementations (pytelegram v20+ vs older)
- Provides a robust sync wrapper even when an event loop is running
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import asyncio
import os
import concurrent.futures
from utils import setup_logging

# Lazy import of telegram to keep import-time errors controllable
try:
    from telegram import Bot
    from telegram.error import TelegramError
except Exception:
    Bot = None
    TelegramError = Exception  # fallback

# Setup logging
logger = setup_logging(__name__)


class TelegramAlerter:
    """
    Telegram bot for sending trading alerts
    """

    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Telegram alerter

        Args:
            bot_token: Telegram bot token (from BotFather)
            chat_id: Telegram chat ID(s) to send messages to (comma-separated for multiple)
        """
        # Try provided args first, then environment
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id_env = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        if chat_id_env:
            self.chat_ids = [cid.strip() for cid in chat_id_env.split(",") if cid.strip()]
        else:
            self.chat_ids = []
        self.bot = None

        # If still missing, attempt to load .env as fallback
        if not self.bot_token or not self.chat_ids:
            try:
                # lazy import to avoid forcing python-dotenv as mandatory
                from dotenv import load_dotenv
                load_dotenv()
                self.bot_token = self.bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
                chat_id_env = os.getenv("TELEGRAM_CHAT_ID")
                if chat_id_env:
                    self.chat_ids = [cid.strip() for cid in chat_id_env.split(",") if cid.strip()]
                logger.debug("Loaded .env inside telegram_alerts as fallback")
            except Exception:
                # dotenv not installed or error loading; continue and log below
                pass

        # Initialize Bot if token + chat id are present
        if self.bot_token and self.chat_ids and Bot is not None:
            try:
                # Create Bot instance. Depending on python-telegram-bot version,
                # this object may expose async or sync send_message.
                self.bot = Bot(token=self.bot_token)
                logger.info("Telegram alerter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram Bot: {e}")
                self.bot = None
        else:
            logger.warning("Telegram bot token or chat ID not provided or telegram package missing. Alerts disabled.")
            if Bot is None:
                logger.warning("python-telegram-bot library not available. Install with `pip install python-telegram-bot`")

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram (async-capable and works with sync Bot APIs)

        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            if not self.bot:
                logger.warning("Telegram bot not initialized")
                return False

            send_func = getattr(self.bot, "send_message", None)
            if send_func is None:
                logger.error("Telegram Bot has no send_message method")
                return False

            # If send_message is an async coroutine function (ptb v20+), await it.
            if asyncio.iscoroutinefunction(send_func):
                await send_func(chat_id=self.chat_id, text=message, parse_mode=parse_mode)
            else:
                # It's a blocking function — run it in the default executor so we don't block the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: send_func(chat_id=self.chat_id, text=message, parse_mode=parse_mode))

            logger.info("Telegram message sent successfully")
            return True

        except TelegramError as e:
            logger.error(f"Telegram error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False

    def send_message_sync(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Synchronous wrapper for sending messages.
        Handles both async and sync Bot.send_message.
        """
        import time
        try:
            if not self.bot:
                logger.warning("Telegram bot not initialized")
                return False

            send_func = getattr(self.bot, "send_message", None)
            if send_func is None:
                logger.error("Telegram Bot has no send_message method")
                return False

            all_success = True
            for chat_id in self.chat_ids:
                try:
                    if asyncio.iscoroutinefunction(send_func):
                        # Use a helper to run async code safely, avoiding 'event loop is closed' errors
                        def run_async(coro):
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # Run in a new thread if already running
                                    import threading
                                    result = [None]
                                    def target():
                                        result[0] = asyncio.new_event_loop().run_until_complete(coro)
                                    t = threading.Thread(target=target)
                                    t.start()
                                    t.join()
                                    return result[0]
                                else:
                                    return loop.run_until_complete(coro)
                            except RuntimeError:
                                # No event loop, create one
                                return asyncio.new_event_loop().run_until_complete(coro)

                        result = run_async(send_func(chat_id=chat_id, text=message, parse_mode=parse_mode))
                    else:
                        # If sync, just call it
                        result = send_func(chat_id=chat_id, text=message, parse_mode=parse_mode)
                    logger.info(f"Telegram message sent successfully (sync direct) to {chat_id}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error sending Telegram message to {chat_id}: {str(e)}")
                    all_success = False
            return all_success

        except TelegramError as e:
            logger.error(f"Telegram error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False

    def send_trade_alert(self, trade_data: Dict) -> bool:
        """
        Send a trading signal alert

        Args:
            trade_data: Dictionary containing trade information

        Returns:
            True if alert sent successfully, False otherwise
        """
        try:
            symbol = trade_data.get("symbol", "Unknown")
            action = trade_data.get("action", "Unknown")
            price = trade_data.get("price", 0)
            rsi = trade_data.get("rsi", 0)
            signal_strength = trade_data.get("signal_strength", 0)
            timestamp = trade_data.get("timestamp", datetime.now())

            # Format message
            message = f"""
🚨 <b>TRADING ALERT</b> 🚨

📈 <b>Symbol:</b> {symbol}
🎯 <b>Action:</b> {action}
💰 <b>Price:</b> Rs.{price:.2f}
📊 <b>RSI:</b> {rsi:.1f}
⚡ <b>Signal Strength:</b> {signal_strength:.2f}
🕐 <b>Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

#TradingAlert #{symbol.replace('.NS', '')} #{action}
            """.strip()

            return self.send_message_sync(message)

        except Exception as e:
            logger.error(f"Error sending trade alert: {str(e)}")
            return False

    def send_backtest_summary(self, backtest_results: Dict) -> bool:
        """
        Send backtest results summary
        """
        try:
            message = f"""
📊 <b>BACKTEST RESULTS SUMMARY</b> 📊

🎯 <b>Total Stocks:</b> {backtest_results.get('total_stocks', 0)}
📈 <b>Total Trades:</b> {backtest_results.get('total_trades', 0)}
🏆 <b>Win Rate:</b> {backtest_results.get('overall_win_rate', 0)}%
💰 <b>Avg Return:</b> {backtest_results.get('avg_return_pct', 0)}%

🥇 <b>Best Performer:</b> {backtest_results.get('best_performer', 'N/A')}
🥉 <b>Worst Performer:</b> {backtest_results.get('worst_performer', 'N/A')}

🕐 <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#BacktestResults #AlgoTrading
            """.strip()

            return self.send_message_sync(message)

        except Exception as e:
            logger.error(f"Error sending backtest summary: {str(e)}")
            return False

    def send_ml_prediction_alert(self, prediction_data: Dict) -> bool:
        """
        Send ML prediction alert
        """
        try:
            symbol = prediction_data.get("symbol", "Unknown")
            prediction = prediction_data.get("prediction_label", "Unknown")
            confidence = prediction_data.get("confidence", 0)
            model_type = prediction_data.get("model_type", "Unknown")
            timestamp = prediction_data.get("timestamp", datetime.now())

            emoji = "📈" if prediction == "UP" else "📉"

            # Format confidence as percentage if it's a fraction (0-1)
            try:
                conf_display = f"{confidence:.1%}" if 0 <= confidence <= 1 else f"{confidence}"
            except Exception:
                conf_display = str(confidence)

            message = f"""
🤖 <b>ML PREDICTION ALERT</b> 🤖

{emoji} <b>Symbol:</b> {symbol}
🎯 <b>Prediction:</b> {prediction}
🎲 <b>Confidence:</b> {conf_display}
🧠 <b>Model:</b> {model_type}
🕐 <b>Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

#MLPrediction #{symbol.replace('.NS', '')} #{prediction}
            """.strip()

            return self.send_message_sync(message)

        except Exception as e:
            logger.error(f"Error sending ML prediction alert: {str(e)}")
            return False

    def send_error_alert(self, error_message: str, component: str = "System") -> bool:
        """
        Send error notification
        """
        try:
            message = f"""
🚨 <b>ERROR ALERT</b> 🚨

⚠️ <b>Component:</b> {component}
❌ <b>Error:</b> {error_message}
🕐 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the system logs for more details.

#ErrorAlert #{component}
            """.strip()

            return self.send_message_sync(message)

        except Exception as e:
            logger.error(f"Error sending error alert: {str(e)}")
            return False

    def send_daily_summary(self, summary_data: Dict) -> bool:
        """
        Send daily trading summary
        """
        try:
            message = f"""
📅 <b>DAILY TRADING SUMMARY</b> 📅

💰 <b>Portfolio Value:</b> Rs.{summary_data.get('portfolio_value', 0):,.2f}
📈 <b>Daily Return:</b> {summary_data.get('daily_return_pct', 0):.2f}%
📊 <b>Total Return:</b> {summary_data.get('total_return_pct', 0):.2f}%

🎯 <b>Trades Today:</b> {summary_data.get('trades_today', 0)}
📍 <b>Active Positions:</b> {summary_data.get('active_positions', 0)}
💵 <b>Cash Balance:</b> Rs.{summary_data.get('cash_balance', 0):,.2f}

🕐 <b>Date:</b> {summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))}

#DailySummary #Portfolio
            """.strip()

            return self.send_message_sync(message)

        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")
            return False

    def send_system_startup_alert(self) -> bool:
        """
        Send system startup notification
        """
        try:
            message = f"""
🚀 <b>ALGO TRADING SYSTEM STARTED</b> 🚀

✅ System initialized successfully
🕐 <b>Started at:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The trading bot is now monitoring markets and ready to execute trades.

#SystemStartup #AlgoTrading
            """.strip()

            return self.send_message_sync(message)

        except Exception as e:
            logger.error(f"Error sending startup alert: {str(e)}")
            return False


def create_telegram_alerter() -> Optional[TelegramAlerter]:
    """
    Create and return a TelegramAlerter instance
    """
    try:
        alerter = TelegramAlerter()
        if alerter.bot:
            return alerter
        else:
            logger.info("Telegram alerts not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
            return None
    except Exception as e:
        logger.error(f"Error creating Telegram alerter: {str(e)}")
        return None


# Example usage and quick test
if __name__ == "__main__":
    print("Testing Telegram alerts...")

    alerter = create_telegram_alerter()

    if alerter:
        print("Telegram alerter is configured and ready.")
        # To manually test, uncomment the following lines:
        # alerter.send_system_startup_alert()
        # alerter.send_trade_alert({...})
        # alerter.send_ml_prediction_alert({...})
    else:
        print("Telegram alerter not configured. Please set environment variables:")
        print("- TELEGRAM_BOT_TOKEN: Get from @BotFather on Telegram")
        print("- TELEGRAM_CHAT_ID: Your chat ID or group chat ID")
