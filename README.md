# Algo Trading System with ML & Automation

A comprehensive Python-based algorithmic trading system that implements RSI + Moving Average crossover strategy with machine learning predictions and automated Google Sheets logging.

## 🚀 Features

### Core Trading Features
- **Data Ingestion**: Fetches real-time data from Yahoo Finance for NIFTY 50 stocks
- **Trading Strategy**: RSI < 30 + 20-DMA crossing above 50-DMA for buy signals
- **Backtesting**: 6-month historical backtesting with comprehensive metrics
- **ML Predictions**: Decision Tree, Random Forest, and Logistic Regression models
- **Automated Execution**: Scheduled analysis during market hours

### Integration Features
- **Google Sheets Logging**: Automated logging of trades, P&L, and analytics
- **Telegram Alerts**: Real-time notifications for signals and system status
- **Performance Analytics**: Win rates, Sharpe ratio, maximum drawdown
- **Error Handling**: Comprehensive logging and error notifications

## 📋 Requirements

### Python Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### External Services Setup

#### 1. Google Sheets Integration
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Sheets API and Google Drive API
4. Create a Service Account and download the JSON credentials
5. Save the credentials file as `credentials.json` in the project root
6. Share your Google Sheet with the service account email

#### 2. Telegram Bot Setup (Optional)
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Create a new bot with `/newbot` command
3. Get your bot token
4. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
5. Add these to your `.env` file

## 🛠️ Installation & Setup

### 1. Clone and Setup
\`\`\`bash
git clone <repository-url>
cd algo-trading-system
pip install -r requirements.txt
\`\`\`

### 2. Configuration
1. Copy `.env.example` to `.env`
2. Fill in your configuration values:
   \`\`\`bash
   # Telegram Configuration
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   
   # Google Sheets
   GOOGLE_CREDENTIALS_FILE=credentials.json
   SPREADSHEET_NAME=Algo Trading Log
   
   # Trading Parameters
   INITIAL_CAPITAL=100000
   \`\`\`

3. Place your Google service account credentials as `credentials.json`

### 3. Project Structure
\`\`\`
algo-trading-system/
├── data_fetcher.py          # Yahoo Finance API integration
├── indicators.py            # Technical indicators (RSI, SMA, MACD)
├── strategy.py              # Trading strategy implementation
├── backtester.py            # Backtesting engine
├── ml_model.py              # Machine learning models
├── gsheet_logger.py         # Google Sheets integration
├── telegram_alerts.py       # Telegram notifications
├── main.py                  # Main system orchestrator
├── scheduler.py             # Automated scheduling
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── .env                     # Configuration file
├── README.md               # This file
├── credentials.json        # Google service account credentials
├── logs/                   # System logs
└── models/                 # Saved ML models
\`\`\`

## 🚀 Usage

### Manual Execution
Run a complete analysis cycle:
\`\`\`bash
python main.py
\`\`\`

### Automated Scheduling
Start the automated scheduler:
\`\`\`bash
python scheduler.py
\`\`\`

The scheduler runs analysis at:
- **9:20 AM**: Market opening analysis
- **12:00 PM**: Mid-day analysis  
- **3:25 PM**: End-of-day analysis
- **Saturday 10:00 AM**: ML model retraining

### Individual Components
Test individual components:
\`\`\`bash
# Test data fetching
python data_fetcher.py

# Test strategy signals
python strategy.py

# Test backtesting
python backtester.py

# Test ML models
python ml_model.py

# Test Google Sheets logging
python gsheet_logger.py

# Test Telegram alerts
python telegram_alerts.py
\`\`\`

## 📊 Output & Results

### Google Sheets Tabs
The system creates the following sheets:
1. **trade_log**: Individual trade records
2. **summary**: Performance summary by stock
3. **win_ratio**: Win/loss ratios
4. **daily_performance**: Daily portfolio metrics
5. **ml_predictions**: ML model predictions

### Console Output
\`\`\`
🚀 Starting Algo Trading System...

📊 MARKET DATA:
  RELIANCE.NS: Rs.2,485.50 (+1.25%)
  TCS.NS: Rs.3,245.75 (-0.85%)
  HDFCBANK.NS: Rs.1,675.25 (+0.45%)

🎯 TRADING SIGNALS:
  RELIANCE.NS: BUY (Strength: 0.85, RSI: 28.5)
  TCS.NS: HOLD (Strength: 0.45, RSI: 55.2)
  HDFCBANK.NS: SELL (Strength: 0.75, RSI: 72.1)

📈 BACKTEST RESULTS:
  Total Trades: 15
  Win Rate: 66.7%
  Avg Return: 2.45%
  Best Performer: RELIANCE.NS

🤖 ML PREDICTIONS:
  RELIANCE.NS: UP (Confidence: 75.2%, Accuracy: 68.5%)
  TCS.NS: DOWN (Confidence: 82.1%, Accuracy: 71.2%)
  HDFCBANK.NS: UP (Confidence: 69.8%, Accuracy: 65.4%)
\`\`\`

### Telegram Notifications
- 🚨 **Trading Alerts**: Buy/sell signals with RSI and price
- 📊 **Backtest Summaries**: Performance metrics
- 🤖 **ML Predictions**: Model predictions with confidence
- ⚠️ **Error Alerts**: System errors and warnings
- 📅 **Daily Summaries**: End-of-day performance

## 🧠 Trading Strategy

### Buy Signal Conditions
1. **RSI < 30** (Oversold condition)
2. **20-DMA crosses above 50-DMA** (Bullish momentum)
3. **Signal strength > 0.6** (Confidence threshold)

### Sell Signal Conditions
1. **RSI > 70** (Overbought condition) OR
2. **20-DMA crosses below 50-DMA** (Bearish momentum)

### Risk Management
- **Position Size**: 10% of capital per trade
- **Commission**: 0.1% per trade
- **Stop Loss**: Implemented via technical indicators
- **Maximum Drawdown Monitoring**: Real-time tracking

## 🤖 Machine Learning Models

### Model Types
1. **Random Forest**: Best overall performance
2. **Decision Tree**: Interpretable rules
3. **Logistic Regression**: Linear relationships

### Features Used
- RSI, MACD, Bollinger Bands
- Moving averages (20, 50 period)
- Volume indicators
- Price momentum
- Volatility measures

### Model Selection
- Automatic selection of best performing model per stock
- Cross-validation for robust evaluation
- Model retraining every weekend

## 📈 Performance Metrics

### Trading Metrics
- **Total Return %**: Overall portfolio performance
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

### ML Metrics
- **Accuracy**: Correct predictions / Total predictions
- **Precision & Recall**: Classification performance
- **Cross-validation Score**: Model robustness
- **Feature Importance**: Most predictive indicators

## 🔧 Customization

### Adding New Stocks
Edit the stock list in `data_fetcher.py`:
```python
self.nifty_50_stocks = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS',
    'YOUR_STOCK.NS'  # Add your stock here
]
