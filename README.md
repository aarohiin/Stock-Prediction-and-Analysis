# 🚀 Stock Analysis Dashboard

A comprehensive web application built with Streamlit for analyzing Indian stocks (NSE) with features including real-time price tracking, technical analysis, sentiment analysis, and price predictions.

## 📋 Features

- **Overall Market Status**
  - Real-time tracking of major indices (NIFTY, SENSEX, etc.)
  - Live intraday NIFTY chart
  - Gold and Silver prices
  - Dow Jones tracking

- **Stock Analysis Tools**
  - Current Price Tracking
  - Historical Price Analysis
  - Multi-stock Comparison
  - Time Series Analysis
  - Technical Indicators (SMA, RSI, MACD)
  - Fundamental Analysis
  - News Sentiment Analysis
  - Price Predictions using Machine Learning

## 🔧 Prerequisites

```
python 3.x
yfinance
gnews
nltk
numpy
pandas
streamlit
plotly
scikit-learn
ta (Technical Analysis Library)
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/aarohiin/Stock-Prediction-and-Analysis/tree/main
cd stock-analysis-dashboard
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
```

## 🚀 Usage

1. Run the Streamlit app:
```bash
streamlit run Stock_dashboard.py
```

2. Navigate to the provided local URL (typically `http://localhost:8501`)

3. Use the sidebar to select different analysis options:
   - Overall Market Status
   - Current Price
   - Price Between Dates
   - Stock Comparison
   - Time Series Analysis
   - Fundamental Analysis
   - Prediction (Gyaani Baba)
   - Technical Analysis

## 📊 Available Analysis Options

### Current Price
- Real-time stock prices
- Latest news and sentiment analysis

### Price Between Dates
- Historical price data
- Interactive line charts

### Stock Comparison
- Multiple stock comparison
- Comparative price charts

### Time Series Analysis
- One-year historical data
- Trend visualization

### Fundamental Analysis
- Market Cap
- PE Ratio
- Dividend Yield
- EPS
- 52-Week High/Low

### Prediction (Gyaani Baba)
- Machine learning-based price predictions
- Random Forest model
- Performance metrics
- Future price forecasts

### Technical Analysis
- SMA (50 and 200 days)
- RSI Indicator
- MACD
- Interactive charts

## 🔒 Data Sources

- Stock data: Yahoo Finance (yfinance)
- News data: Google News (gnews)
- Technical Indicators: Technical Analysis Library (ta)

## ⚠️ Notes

- The app uses caching to optimize data fetching
- Predictions are based on historical data and should not be used as the sole basis for investment decisions
- News sentiment analysis is performed using NLTK's VADER sentiment analyzer

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

[MIT License](LICENSE)

## 👩‍💻 Author

Created by Aarohi

---

*Disclaimer: This tool is for educational and research purposes only. It should not be considered as financial advice. Always do your own research before making investment decisions.*
