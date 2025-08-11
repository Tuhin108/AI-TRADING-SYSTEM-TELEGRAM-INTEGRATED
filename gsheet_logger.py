import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

from gsheet_webhook import GoogleSheetsWebhookLogger

class GoogleSheetsLogger:
    def __init__(self):
        self.client = None
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")
        self.webhook_logger = None

        # Try GCP credentials first
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if GCP_AVAILABLE and creds_path and os.path.exists(creds_path) and self.sheet_id:
            try:
                creds = service_account.Credentials.from_service_account_file(
                    creds_path,
                    scopes=["https://www.googleapis.com/auth/spreadsheets"]
                )
                self.client = build('sheets', 'v4', credentials=creds)
                logger.info("Google Sheets API client initialized with GCP credentials")
            except Exception as e:
                logger.error(f"GCP Sheets API init failed: {e}")

        # If no GCP, use webhook fallback
        if not self.client:
            self.webhook_logger = GoogleSheetsWebhookLogger()
            if self.webhook_logger.webhook_url:
                logger.info("Using Google Sheets Webhook logger fallback")
            else:
                logger.warning("No Google Sheets logging method configured")

    def log_multiple_trades(self, trades):
        if self.client:
            self._append_rows("trade_log", trades)
        elif self.webhook_logger:
            self.webhook_logger.log_multiple_trades(trades)

    def update_summary(self, summary):
        if self.client:
            self._append_rows("summary", [summary])
        elif self.webhook_logger:
            self.webhook_logger.update_summary(summary)

    def update_win_ratio(self, win_ratio):
        if self.client:
            self._append_rows("win_ratio", [win_ratio])
        elif self.webhook_logger:
            self.webhook_logger.update_win_ratio(win_ratio)

    def log_ml_prediction(self, prediction):
        if self.client:
            row = [
                prediction.get("date", ""),
                prediction.get("symbol", ""),
                prediction.get("prediction", ""),
                prediction.get("confidence", ""),
                prediction.get("actual_movement", ""),
                prediction.get("accuracy", ""),
                prediction.get("model_type", "")
            ]
            self._append_rows("ml_predictions", [row])
        elif self.webhook_logger:
            self.webhook_logger.log_ml_prediction(prediction)

    def _append_rows(self, sheet_name, rows):
        try:
            body = {'values': rows}
            self.client.spreadsheets().values().append(
                spreadsheetId=self.sheet_id,
                range=f"{sheet_name}!A1",
                valueInputOption="USER_ENTERED",
                body=body
            ).execute()
            logger.info(f"Rows appended to {sheet_name} via GCP API")
        except Exception as e:
            logger.error(f"Error appending rows to {sheet_name}: {e}")
