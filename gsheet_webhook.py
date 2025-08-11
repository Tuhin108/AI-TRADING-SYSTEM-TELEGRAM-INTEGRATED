import requests
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class GoogleSheetsWebhookLogger:
    """
    Simple Google Sheets logger using a published Google Apps Script Web App
    """
    def __init__(self, webhook_url: str = None, secret: str = None):
        self.webhook_url = webhook_url or os.getenv("SHEETS_WEBHOOK_URL")
        self.secret = secret or os.getenv("SHEETS_WEBHOOK_SECRET")

        if not self.webhook_url:
            logger.warning("Google Sheets Webhook URL not provided. Logging disabled.")
        else:
            logger.info("Google Sheets Webhook Logger initialized")

    def post_row(self, sheet: str, row: list) -> bool:
        """
        Append a single row to the given sheet.
        """
        if not self.webhook_url:
            return False

        payload = {
            "sheet": sheet,
            "row": row
        }
        if self.secret:
            payload["secret"] = self.secret

        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            if r.status_code == 200:
                resp = r.json()
                if resp.get("ok"):
                    logger.info(f"Row appended to {sheet} via webhook")
                    return True
                else:
                    logger.error(f"Webhook responded but not ok: {resp}")
            else:
                logger.error(f"Webhook POST failed {r.status_code} {r.text}")
        except Exception as e:
            logger.error(f"Webhook post exception: {e}")
        return False

    def log_multiple_trades(self, trades: list):
        for trade in trades:
            self.post_row("trade_log", trade)

    def update_summary(self, summary: list):
        self.post_row("summary", summary)

    def update_win_ratio(self, win_ratio: list):
        self.post_row("win_ratio", win_ratio)

    def log_ml_prediction(self, prediction: dict):
        row = [
            prediction.get("date", ""),
            prediction.get("symbol", ""),
            prediction.get("prediction", ""),
            prediction.get("confidence", ""),
            prediction.get("actual_movement", ""),
            prediction.get("accuracy", ""),
            prediction.get("model_type", "")
        ]
        self.post_row("ml_predictions", row)

def convert_backtest_results_to_sheets_format(results):
    # Convert backtest results to Google Sheets format
    # Expecting results to be a dict with keys like 'individual_results', 'all_trades', etc.
    trades = []
    summary = []
    win_ratio = []

    # Extract trades
    if 'all_trades' in results:
        for trade in results['all_trades']:
            trades.append([
                trade.symbol,
                getattr(trade, 'entry_date', ''),
                getattr(trade, 'exit_date', ''),
                getattr(trade, 'entry_price', ''),
                getattr(trade, 'exit_price', ''),
                getattr(trade, 'quantity', ''),
                getattr(trade, 'trade_type', ''),
                getattr(trade, 'pnl', ''),
                getattr(trade, 'pnl_pct', ''),
                getattr(trade, 'duration_days', '')
            ])

    # Extract summary (one row per symbol)
    if 'individual_results' in results:
        for symbol, metrics in results['individual_results'].items():
            summary.append([
                symbol,
                metrics.get('total_trades', 0),
                metrics.get('win_rate', 0),
                metrics.get('total_return_pct', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('profit_factor', 0)
            ])

    # Extract win ratio (overall)
    if 'overall_win_rate' in results:
        win_ratio.append([
            results.get('overall_win_rate', 0),
            results.get('total_trades', 0)
        ])

    return {
        'trades': trades,
        'summary': summary,
        'win_ratio': win_ratio
    }

