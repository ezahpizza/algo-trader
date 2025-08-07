# Algorithmic Trading System

A comprehensive, production-ready algorithmic trading system in Python for NIFTY 50 stocks with ML prediction, backtesting, and automated alerts.

## 🚀 Features

### Core Functionality
- **Multi-stock data ingestion** for NIFTY 50 stocks using `yfinance`
- **Rule-based trading strategy** using RSI + Moving Average crossover
- **ML prediction module** for next-day price movement forecasting
- **Comprehensive backtesting** with performance metrics
- **Google Sheets integration** for trade logging and performance tracking
- **Telegram alerts** for signals, predictions, and system status
- **Automated scheduling** for daily execution

### Technical Indicators
- RSI (Relative Strength Index)
- SMA (Simple Moving Averages) - 20 & 50 period
- MACD (Moving Average Convergence Divergence)
- Volume indicators (OBV, Volume SMA)
- Momentum indicators (ROC, Williams %R, Stochastic)
- Bollinger Bands
- Custom technical features

### Machine Learning
- **Models**: Logistic Regression, Decision Tree, Random Forest
- **Features**: RSI, MACD histogram, Volume delta, Momentum, SMA slopes
- **Target**: Binary next-day price movement prediction
- **Validation**: Time-series aware cross-validation
- **Persistence**: Model saving/loading with `joblib`

### Performance Tracking
- Win/Loss ratio and accuracy metrics
- Total P&L and drawdown analysis
- Sharpe ratio and volatility metrics
- Monthly return breakdowns
- Trade-by-trade analysis

## 📁 Project Structure

```
algo-trader/
├── main.py                 # Main application entry point
├── config.py               # Global configuration
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
├── data/
│   ├── __init__.py
│   └── fetcher.py         # Stock data fetching from Yahoo Finance
├── strategy/
│   ├── __init__.py
│   ├── indicators.py      # Technical indicator calculations
│   └── rules.py           # Trading signal generation
├── ml/
│   ├── __init__.py
│   ├── features.py        # Feature engineering
│   ├── trainer.py         # Model training
│   └── predictor.py       # ML predictions
├── utils/
│   ├── __init__.py
│   ├── backtester.py      # Backtesting engine
│   ├── gsheet_logger.py   # Google Sheets integration
│   └── telegram_alerts.py # Telegram notifications
├── models/                # Saved ML models
├── logs/                  # Application logs
└── credentials/           # Google Sheets credentials
```

## 🛠️ Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd algo-trader

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
```

Required environment variables:
```env
# Stock Configuration
NIFTY_50_STOCKS=RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS
LOOKBACK_PERIOD=365

# Google Sheets
GOOGLE_SHEET_ID=your_google_sheet_id_here
GOOGLE_CREDENTIALS_PATH=./credentials/service_account.json

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

### 3. Setup Google Sheets Integration

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Google Sheets API and Google Drive API

2. **Create Service Account**:
   - Go to IAM & Admin > Service Accounts
   - Create new service account
   - Download JSON key file
   - Save as `credentials/service_account.json`

3. **Create Google Sheet**:
   - Create a new Google Sheet
   - Copy the Sheet ID from URL: `https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit`
   - Share sheet with service account email (from JSON file)

### 4. Setup Telegram Bot

1. **Create Bot**:
   - Message [@BotFather]() on Telegram
   - Use `/newbot` command
   - Save the bot token

2. **Get Chat ID**:
   - Start conversation with your bot
   - Send message to: `https://api.telegram.org/bot{BOT_TOKEN}/getUpdates`
   - Find your chat ID in response

## 🎯 Usage

### Command Line Interface

```bash
# Test all system components
python main.py test

# Run single pipeline execution
python main.py run

# Start scheduled daily execution
python main.py schedule

# Train ML model only
python main.py train
```

### Trading Strategy Configuration

The system uses a dual-signal approach:

**Entry Conditions (BUY)**:
- RSI < 30 (oversold condition)
- 20-day SMA crosses above 50-day SMA (bullish crossover)

**Exit Conditions (SELL)**:
- RSI > 70 (overbought condition)
- 20-day SMA crosses below 50-day SMA (bearish crossover)

### Machine Learning Features

The ML module uses these primary features:
- **RSI**: Momentum indicator
- **MACD Histogram**: Trend momentum
- **Volume Delta**: Volume change patterns
- **Momentum**: Price momentum
- **SMA Slopes**: Trend direction indicators

Plus additional engineered features:
- Price position within daily range
- Gap analysis
- Volatility measures
- Pattern recognition

## 📊 Output Examples

### Google Sheets Layout

**Tab 1: Trade Log**
| Date | Ticker | Signal | Price | P&L | ML Prediction | Actual Movement |
|------|--------|--------|-------|-----|---------------|----------------|
| 2024-08-07 | RELIANCE.NS | BUY | 2,456.30 | 0 | UP (85%) | TBD |

**Tab 2: Summary**
| Metric | Value |
|--------|-------|
| Total Trades | 25 |
| Win Ratio (%) | 68.5 |
| Total P&L | ₹15,432 |
| Sharpe Ratio | 1.24 |

**Tab 3: Daily P&L**
| Date | Portfolio Value | Daily Return (%) | Cumulative Return (%) |
|------|----------------|------------------|----------------------|
| 2024-08-07 | ₹1,05,432 | 0.85 | 5.43 |

### Telegram Alerts

```
🟢 TRADING SIGNAL ALERT

Signal: BUY
Stock: RELIANCE.NS
Price: ₹2,456.30
Reason: RSI oversold (28.5); MA bullish crossover

🤖 ML Prediction: UP
Confidence: 85.2%

Time: 2024-08-07 09:30
```

### Performance Metrics Export

```json
{
  "total_return_pct": 12.45,
  "win_rate_pct": 68.5,
  "total_trades": 25,
  "max_drawdown_pct": -3.2,
  "sharpe_ratio": 1.24,
  "profit_factor": 2.1
}
```

## 🔧 Configuration Options

### Strategy Parameters
```python
# RSI settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Moving averages
SMA_SHORT = 20
SMA_LONG = 50
```

### ML Model Options
```python
# Available models
MODEL_TYPES = ['logistic', 'decision_tree', 'random_forest']

# Training parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
```

### Data Configuration
```python
# Stock selection
NIFTY_50_STOCKS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', ...]
LOOKBACK_PERIOD = 365  # Days of historical data
DATA_INTERVAL = '1d'   # Daily data
```

## 📈 Backtesting Results

The system provides comprehensive backtesting with metrics:

- **Return Analysis**: Total returns vs buy-and-hold
- **Risk Metrics**: Maximum drawdown, volatility, Sharpe ratio
- **Trade Analysis**: Win rate, average trade P&L, trade frequency
- **Time Analysis**: Monthly returns, drawdown periods

## 🛡️ Error Handling

The system includes robust error handling:

- **Data Fetch Failures**: Automatic retries and fallbacks
- **API Rate Limits**: Built-in delays and retry logic
- **Network Issues**: Timeout handling and graceful degradation
- **Model Failures**: Fallback to rule-based signals
- **Alert System**: Telegram notifications for all errors

## 🔄 Automated Scheduling

The system runs automatically using the scheduler:

- **Daily Execution**: Configurable time (default: 9:30 AM IST)
- **Model Retraining**: Weekly automatic retraining
- **Health Checks**: System component testing
- **Performance Updates**: Daily summary reports

## 📚 Dependencies

Core libraries:
- `yfinance`: Stock data fetching
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning
- `ta`: Technical analysis
- `gspread`: Google Sheets API
- `requests`: HTTP requests for Telegram
- `schedule`: Task scheduling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves substantial risk of loss. Use at your own risk and always do your own research before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and support:
1. Check the logs in `logs/run.log`
2. Verify environment configuration
3. Test system components: `python main.py test`
4. Check Telegram alerts for error notifications

## 🎯 Roadmap

Future enhancements:
- [ ] Support for more exchanges and asset classes
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Real-time data streaming
- [ ] Portfolio optimization
- [ ] Risk management module
- [ ] Web dashboard interface
- [ ] Advanced charting and visualization
