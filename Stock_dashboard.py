import yfinance as yf
from gnews import GNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download and setup NLTK for sentiment analysis
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK data: {e}")

# Set the title of the Streamlit app
st.markdown("<h1 style='text-align: center; color: #0073e6;'>🚀 Stock Analysis Dashboard By Aarohi 🚀</h1>", unsafe_allow_html=True)

# List of NSE stock symbols
stock_symbols = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "WIPRO.NS",
    "ITC.NS", "BAJAJFINSV.NS", "MARUTI.NS", "SBIN.NS", "NESTLE.NS",
    "HINDUNILEVR.NS", "BRITANIA.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    "TATAMOTORS.NS", "POWERGRID.NS", "TECHM.NS", "CIPLA.NS", "UPL.NS"
]

# Sidebar selection for options
st.sidebar.header("Select an Option")
option = st.sidebar.selectbox("Choose Analysis", (
    "Overall Market Status",
    "Current Price",
    "Price Between Dates",
    "Stock Comparison",
    "Time Series Analysis",
    "Fundamental Analysis",
    "Prediction (Gyaani Baba)",
    "Technical Analysis"
))


@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data for a specific symbol within a date range."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No data available for {symbol} between {start_date} and {end_date}.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


def fetch_market_data():
    """Retrieve current data for major market indices."""
    indices = {
        "NIFTY": "^NSEI",
        "SENSEX": "^BSESN",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Dow Jones": "^DJI"
    }
    market_data = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                close_price = data['Close'].iloc[-1]
                change = close_price - data['Open'].iloc[0]
                percent_change = (change / data['Open'].iloc[0]) * 100
                market_data[name] = {
                    "price": close_price,
                    "change": change,
                    "percent_change": percent_change
                }
        except Exception as e:
            st.error(f"Error fetching data for {name}: {str(e)}")
    return market_data


def display_market_interface(market_data):
    """Display overall market data in a grid layout."""
    cols = st.columns(len(market_data))
    for i, (name, data) in enumerate(market_data.items()):
        with cols[i]:
            st.metric(label=name, value=f"{data['price']:.2f}", delta=f"{data['change']:.2f} ({data['percent_change']:.2f}%)")
    
    # Plot NIFTY Intraday Chart
    nifty_data = yf.download('^NSEI', period='1d', interval='5m')
    if not nifty_data.empty:
        fig = go.Figure(data=go.Scatter(x=nifty_data.index, y=nifty_data['Close'], mode='lines'))
        fig.update_layout(title='NIFTY Intraday Chart', xaxis_title='Time', yaxis_title ='Price', template='plotly_dark')
        st.plotly_chart(fig)


def fetch_news_sentiment(symbol):
    """Fetch news and analyze sentiment for a given stock symbol."""
    try:
        gnews = GNews(language='en', country='IN', max_results=10)
        news = gnews.get_news(symbol)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        
        st.write("\nLatest News and Sentiments:")
        st.write("--------------------------------------------------")
        
        for article in news:
            title = article['title']
            sentiment_score = sentiment_analyzer.polarity_scores(title)['compound']
            sentiments.append(sentiment_score)
            sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            st.write(f"Title: {title}\nSentiment: {sentiment_label} (Score: {sentiment_score:.2f})\n")
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        overall_sentiment = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
        
        st.write("--------------------------------------------------")
        st.write(f"Overall Sentiment: {overall_sentiment} (Score: {avg_sentiment:.2f})")
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")


def gyaani_baba_prediction(symbol, days=120):
    """Predict future stock prices using a Random Forest model."""
    try:
        data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=1), datetime.date.today())
        data['SMA_50'] = SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = SMAIndicator(data['Close'], window=200).sma_indicator()
        data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = MACD(data['Close']).macd_diff()
        
        # Drop rows with missing values after creating indicators
        data = data.dropna()
        
        X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']]
        y = data['Close'].shift(-1).dropna()  # Predict next day's close price
        X = X.iloc[:-1]  # Drop last row to align with y
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        train_r2 = r2_score(y_train, train_predictions) * 100
        test_r2 = r2_score(y_test, test_predictions) * 100
        
        # Generate future predictions
        last_data = X.iloc[-1].values.reshape(1, -1)
        predictions = []
        for _ in range(days):
            pred = model.predict(last_data)[0]
            predictions.append(pred)
            last_data = np.roll(last_data, -1)
            last_data[0, -1] = pred
        
        # Display model performance metrics
        st.write("Model Performance Metrics:")
        st.write("--------------------------------------------------")
        st.write(f"Training R-squared Score: {train_r2:.2f}%")
        st.write(f"Testing R-squared Score: {test_r2:.2f}%")
        st.write(f"Training RMSE: ₹{train_rmse:.2f}")
        st.write(f"Testing RMSE: ₹{test_rmse:.2f}")
        st.write(f"Training Accuracy: {100 - train_rmse:.2f}%")
        
        # Display predictions
        future_dates = pd.date_range(datetime.date.today(), periods=days)
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
        st.write("Predicted Prices:")
        st.write(pred_df)
        st.line_chart(pred_df.set_index('Date'))
        
        return predictions
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None


if option == "Overall Market Status":
    market_data = fetch_market_data()
    display_market_interface(market_data)

elif option == "Current Price":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    ticker = yf.Ticker(symbol)
    st.write(f"Current Price of {symbol}: ₹{ticker.info.get('currentPrice', 'N/A')}")
    fetch_news_sentiment(symbol)

elif option == "Price Between Dates":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    data = fetch_stock_data(symbol, start_date, end_date)
    if not data.empty:
        st.write(f"Price of {symbol} between {start_date} and {end_date}:")
        st.write(data)
        st.line_chart(data['Close'])

elif option == "Stock Comparison":
    symbols = st.sidebar.multiselect("Select Stocks for Comparison", stock_symbols)
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    data = {}
    for symbol in symbols:
        data[symbol] = fetch_stock_data(symbol, start_date, end_date)
    if all(not df.empty for df in data.values()):
        st.write("Stock Comparison:")
        for symbol, df in data.items():
            st.write(f"{symbol}:")
            st.write(df)
            st.line_chart(df['Close'])
    else:
        st.error("No data available for the selected stocks and date range.")

elif option == "Time Series Analysis":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=1), datetime.date.today())
    if not data.empty:
        st.write(f"Time Series Analysis of {symbol}:")
        st.write(data)
        st.line_chart(data['Close'])

elif option == "Fundamental Analysis":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    ticker = yf.Ticker(symbol)
    info = ticker.info
    st.write(f"Fundamental Analysis of {symbol}:")
    st.write("--------------------------------------------------")
    st.write(f"Market Cap: ₹{info.get('marketCap', 'N/A')}")
    st.write(f"PE Ratio: {info.get('peRatio', 'N/A')}")
    st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
    st.write(f"EPS: ₹{info.get('eps', 'N/A')}")
    st.write(f"52-Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
    st.write(f"52-Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}")

elif option == "Prediction (Gyaani Baba)":
    symbol = st.sidebar.selectbox("Select Stock for Prediction", stock_symbols)
    days = st.sidebar.slider("Days to Predict", 1, 120)
    predictions = gyaani_baba_prediction(symbol, days)
    fetch_news_sentiment(symbol)

elif option == "Technical Analysis":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=1), datetime.date.today())
    if not data.empty:
        st.write(f"Technical Analysis of {symbol}:")
        st.write("--------------------------------------------------")
        sma_50 = SMAIndicator(data['Close'], window=50).sma_indicator()
        sma_200 = SMAIndicator(data['Close'], window=200).sma_indicator()
        rsi = RSIIndicator(data['Close'], window=14).rsi()
        macd = MACD(data['Close']).macd_diff()
        st.write("SMA (50):")
        st.write(sma_50)
        st.write("SMA (200):")
        st.write(sma_200)
        st.write("RSI:")
        st.write(rsi)
        st.write("MACD:")
        st.write(macd)
        st.line_chart(data['Close'])
        st.line_chart(sma_50)
        st.line_chart(sma_200)
        st.line_chart(rsi)
        st.line_chart(macd)