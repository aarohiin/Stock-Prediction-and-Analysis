Stock Analysis Dashboard
Overview
The Stock Analysis Dashboard is an interactive web application built using Streamlit that allows users to analyze stock market data, perform sentiment analysis on news articles, and make predictions about future stock prices. This application utilizes various libraries including yfinance, GNews, nltk, and scikit-learn to provide a comprehensive analysis of selected stocks.

Features
Overall Market Status: Displays the current status of major market indices (NIFTY, SENSEX, Gold, Silver, Dow Jones) with their prices, changes, and percentage changes.

Current Price: Shows the current price of a selected stock along with recent news and sentiment analysis.

Price Between Dates: Allows users to fetch and visualize stock prices within a specified date range.

Stock Comparison: Enables users to compare the performance of multiple stocks over a selected date range.

Time Series Analysis: Provides a time series analysis of a selected stock over the past year.

Fundamental Analysis: Displays key financial metrics of a selected stock, including market cap, PE ratio, dividend yield, EPS, and 52-week high/low.

Prediction (Gyaani Baba): Utilizes a Random Forest model to predict future stock prices based on historical data and technical indicators.

Technical Analysis: Displays key technical indicators such as SMA (50 and 200), RSI, and MACD for a selected stock.

Requirements
To run this application, you'll need the following Python libraries:

yfinance
GNews
nltk
numpy
pandas
streamlit
datetime
plotly
scikit-learn
ta (Technical Analysis Library)
You can install the required packages using pip:

bash

Verify

Open In Editor
Edit
Copy code
pip install yfinance gnews nltk numpy pandas streamlit plotly scikit-learn ta
Additionally, you'll need to download the VADER lexicon for sentiment analysis using NLTK:

python

Verify

Open In Editor
Edit
Copy code
import nltk
nltk.download('vader_lexicon')
Usage
Clone the repository or download the code files.
Navigate to the directory containing the app.py file.
Run the Streamlit application with the following command:
bash

Verify

Open In Editor
Edit
Copy code
streamlit run app.py
Open your web browser and go to http://localhost:8501 to access the dashboard.
Code Structure
Imports: The necessary libraries are imported at the beginning of the code.
Logging: Basic logging is set up to help track the application's performance and errors.
Functions: Various functions are defined to handle data fetching, sentiment analysis, predictions, and visualizations.
Streamlit Interface: The user interface is created using Streamlit components, allowing users to interact with the application and select various options.
Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Any improvements, bug fixes, or new features are welcome!

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Thanks to the developers of the libraries used in this project for their contributions to the Python ecosystem.
Special thanks to the creators of Streamlit for providing an easy way to build interactive web applications.
