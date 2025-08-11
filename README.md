# 🚀 Algo Trading System with ML & Automation

This is a powerful and comprehensive Python-based algorithmic trading system designed for automated financial market analysis and trade execution. It integrates a rule-based trading strategy with machine learning predictions and automates real-time notifications via Telegram and data logging to Google Sheets. The system is built for robustness, providing detailed performance metrics and error handling.

## You can follow my bot 
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/0208c3a9-010b-40ae-ba79-a35f25aa0748" />


-----

## ✨ Key Features

### Core Trading Functionality

  - **📊 Data Ingestion**: Fetches historical and real-time OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance for a customizable list of stocks.
  - **🧠 Trading Strategy**: Implements a dual-indicator strategy combining RSI (Relative Strength Index) and moving average crossovers (20-DMA vs. 50-DMA) to generate precise buy and sell signals.
  - **📈 Backtesting Engine**: A robust backtesting module simulates the strategy over historical data (6 months by default) to provide a comprehensive analysis of performance metrics.
  - **🤖 Machine Learning Predictions**: Integrates three distinct ML models—Random Forest, Decision Tree, and Logistic Regression—to predict next-day stock movements, automatically selecting the best-performing model for each stock.
  - **⏰ Automated Execution**: An integrated scheduler allows for automated analysis during market hours and model retraining on weekends, ensuring the system remains current.

### Seamless Integrations & Analytics

  - **📋 Google Sheets Logging**: Automatically logs all trading activities, including individual trades, performance summaries by stock, win/loss ratios, and ML predictions, to a Google Sheet.
  - **🔔 Telegram Alerts**: Provides real-time notifications directly to your Telegram chat for a wide range of events:
      - **🚨 Trading Alerts**: Instant buy/sell signals with key metrics like price and RSI.
      - **📊 Backtest Summaries**: Post-backtest reports on total trades, win rates, and average returns.
      - **🤖 ML Predictions**: Alerts for new model predictions with confidence levels.
      - **⚠️ Error Alerts**: Critical notifications for system failures and errors.
  - **📈 Performance Analytics**: The system calculates and logs key performance indicators, including total return percentage, win rate, Sharpe ratio, and maximum drawdown.
  - **✅ Robust Error Handling**: Comprehensive logging and a dedicated Telegram error alert system ensure you are immediately aware of any issues.

-----

## 🛠️ Installation & Setup

### 1\. Clone the Repository

```bash
git clone <repository-url>
cd algo-trading-system
```

### 2\. Install Dependencies

Install all the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

### 3\. Configure External Services

The system relies on external services for logging and alerting. Follow the steps in the `.env.example` file to set up your configurations.

#### Google Sheets

  - Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
  - Enable the Google Sheets API and Google Drive API.
  - Create a Service Account, download the JSON credentials, and save the file as `credentials.json` in the project root.
  - Share your Google Sheet with the service account email.

#### Telegram Bot

  - Create a new bot with [@BotFather](https://t.me/botfather) to get your `TELEGRAM_BOT_TOKEN`.
  - Get your `TELEGRAM_CHAT_ID` by messaging [@userinfobot](https://t.me/userinfobot).

### 4\. Configuration File

Copy the `.env.example` file and rename it to `.env`. Fill in your specific configuration values:

```bash
# Telegram Configuration (Get from @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_1,your_chat_id_2

# Google Sheets Configuration
SHEETS_WEBHOOK_URL=https://your-google-apps-script-webhook-url
SHEETS_WEBHOOK_SECRET=your_webhook_secret

# Trading Configuration
INITIAL_CAPITAL=100000
COMMISSION_RATE=0.001
POSITION_SIZE=0.1

# System Configuration
LOG_LEVEL=INFO

# Stock Selection (Optional - uses default NIFTY 50 stocks if not specified)
# STOCK_SYMBOLS=RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,HINDUNILVR.NS
```

-----

## 📂 Project Structure

```
algo-trading-system/
├── data_fetcher.py          # Fetches stock data from Yahoo Finance
├── indicators.py            # Calculates technical indicators like RSI, SMA, MACD
├── strategy.py              # Implements the trading strategy and signal generation
├── backtester.py            # Backtesting engine for historical simulations
├── ml_model.py              # Machine learning models for price prediction
├── gsheet_webhook.py        # Webhook-based Google Sheets logging
├── gsheet_logger.py         # Google Sheets logging via API or webhook
├── telegram_alerts.py       # Manages sending notifications via Telegram
├── main.py                  # Main system orchestrator for a complete analysis cycle
├── scheduler.py             # Handles automated scheduling of jobs
├── utils.py                 # Utility functions for logging and helpers
├── requirements.txt         # List of Python dependencies
├── .env.example             # Template for configuration file
├── README.md                # Project documentation
├── logs/                    # Directory for system logs
└── models/                  # Directory for saved ML models
```

-----

## 🚀 Usage

### Manual Execution

Run a complete analysis cycle manually:

```bash
python main.py
```

### Automated Scheduling

Start the automated scheduler to run the system at predefined times:

```bash
python scheduler.py
```

The scheduler is configured to run a full analysis on market days at **9:20 AM**, **12:00 PM**, and **3:25 PM**, with ML model retraining scheduled for **Saturday at 10:00 AM**.

-----

## 🧠 Trading Strategy

The system's core trading logic is built on a robust combination of technical indicators to identify potential market opportunities.

### Buy Signal Conditions

A **BUY** signal is generated when:

  - The **RSI is below 30** (indicating an oversold condition).
  - The **20-day Simple Moving Average (SMA) crosses above the 50-day SMA** (confirming bullish momentum).
  - The combined signal strength exceeds a configurable confidence threshold.

### Sell Signal Conditions

A **SELL** signal is generated when:

  - The **RSI is above 70** (indicating an overbought condition).
  - **OR** the **20-day SMA crosses below the 50-day SMA** (confirming bearish momentum).

### Risk Management

  - **Position Size**: Trades are executed with a fixed position size of 10% of the capital per trade.
  - **Commissions**: A commission rate of 0.1% is applied to each trade.

-----

## 🤖 Machine Learning Models

The ML prediction module enhances the system by forecasting next-day stock movements.

### Model Types

The system automatically trains and evaluates multiple models to find the best fit for each stock:

  - **Random Forest Classifier**: An ensemble model known for its high accuracy.
  - **Decision Tree Classifier**: Provides interpretable rules and insights.
  - **Logistic Regression**: A linear model for identifying key relationships.

### Features Used

Models are trained using a rich set of features derived from:

  - RSI, MACD, and Bollinger Bands.
  - Moving averages (20 and 50 period).
  - Volume indicators and volatility measures.
  - Price and volume-based momentum indicators.

-----

## 📈 Performance Metrics

The backtester provides a comprehensive overview of the strategy's performance through key metrics:

### Trading Metrics

  - **Total Return %**: The overall profit or loss percentage.
  - **Win Rate**: The percentage of profitable trades.
  - **Profit Factor**: The ratio of gross profit to gross loss.
  - **Sharpe Ratio**: Measures risk-adjusted returns.
  - **Maximum Drawdown**: The largest drop from a peak to a trough in the portfolio value.
