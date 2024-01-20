# Trading Bot

This Python script implements a simple cryptocurrency trading bot using historical price data and machine learning for price prediction. The bot is designed to make trading decisions based on technical indicators and includes features such as buying, selling, and additional trading strategies.

## Overview

The trading bot is implemented in Python and uses the following libraries:

- **yfinance**: For downloading historical cryptocurrency price data.
- **scikit-learn**: For building a machine learning model to predict cryptocurrency prices.
- **joblib**: For saving and loading machine learning models.
- **termcolor**: For colored console output.
- **matplotlib**: For plotting feature importance analysis.
- **datetime**: For handling date and time operations.
- **threading**: For running the trading loop in a separate thread.
- **json**: For saving and loading order book and trading data.

## Features

1. **Machine Learning Model:** The bot uses a Passive Aggressive Regressor model for predicting cryptocurrency prices based on historical data. The model is saved to disk and loaded if available.

2. **Technical Indicators:** The script calculates various technical indicators such as moving averages, Bollinger Bands, MACD, stochastic oscillator, RSI, Aroon, Fibonacci retracement, Ichimoku cloud, volatility, and ATR.

3. **Buy and Sell Logic:** The bot implements buying and selling logic based on a combination of technical indicators. It includes conditions for entering and...

...exiting positions, as well as risk management strategies such as stop-loss and take-profit.

4. **Additional Trading Strategies:** The script includes a method for implementing additional trading strategies, allowing users to customize and extend the bot's behavior.

5. **Backtesting:** A simple backtesting strategy is provided to simulate trading decisions based on historical data. This allows users to evaluate the performance of the bot without executing actual trades.

## Getting Started

### Install Dependencies:

```bash
pip install yfinance scikit-learn joblib termcolor matplotlib
```
### Run the Script:
```bash
python trading_bot.py
```
The script will start the trading loop in a separate thread, downloading real-time data and making trading decisions.

### Configuration
The script can be configured to trade a specific cryptocurrency by initializing the Trader class with the desired stock name (e.g., "BTC-USD"). The trading logic, technical indicators, and risk management parameters can be customized within the script.

### Testing
To run the provided test suite:
```bash
python trading_bot.py
```
The test suite checks various functionalities such as data downloading, model training, buying and selling conditions, and additional trading strategies.

### Disclaimer
Use this script at your own risk. Cryptocurrency trading involves financial risk, and it is essential to thoroughly test any trading strategy in a simulated environment before deploying it in a live setting. The provided script is for educational purposes only and does not constitute financial advice.

### License
This trading bot script is licensed under the MIT License. See the LICENSE file for details.
