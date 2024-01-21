import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from termcolor import colored
import matplotlib.pyplot as plt
import datetime
import os
import threading
import json
from PortfolioManager import PortfolioManager
import time


class Trader:
    def __init__(self, stock_name):
        self.stock_name = stock_name
        self.model_filename = f"{self.stock_name}_price_prediction_model.joblib"
        self.order_book = [] 
        self.trading_data = []
        self.load_order_book() 
        self.load_trading_data()
        self.portfolio = {"cash": self.trading_data["cash"], f"{self.stock_name}_units": self.trading_data["units"][f"{self.stock_name}"]}
        self.initial_cash = self.portfolio["cash"]
        self.in_position = False
        self.buy_price = 0
        self.portfolio_manager = PortfolioManager(self.initial_cash)
        self.bitcoin = self.download()
        self.model = self.load_or_create_model()
        self.run_thread = True
        self.trainModel()

    def load_order_book(self):
        try:
            with open("order_book.json", "r") as f:
                self.order_book = json.load(f)
            print("Order book loaded successfully.")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print("Order book file not found. Starting with an empty order book.")

    def load_trading_data(self):
        try:
            with open("trading_data.json", "r") as f:
                self.trading_data = json.load(f)
            print("Trading data loaded successfully.")
            if 'units' not in self.trading_data:
                self.trading_data['units'] = {}
            if self.stock_name not in self.trading_data['units']:
                self.trading_data['units'][self.stock_name] = 0
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            self.trading_data = {
                "cash": 1.20,
                "units": {
                    self.stock_name: 0
                }
            }


    def save_order_book(self):
        with open("order_book.json", "w") as f:
            json.dump(self.order_book, f)

    def save_trading_data(self):
        with open("trading_data.json", "w") as f:
            json.dump(self.trading_data, f)     
    
    def load_or_create_model(self):
        if os.path.exists(self.model_filename):
            model = joblib.load(self.model_filename)
            print(f"Trained model loaded from {self.model_filename}")
        else:
            self.trainModel()
            model = self.model
            self.save_model(self.model)
        return model

    def download(self):
        # Download historical Bitcoin data for different time periods
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=7)
        bitcoin_today = yf.download("BTC-USD", period="1d", interval="1m")
        bitcoin_recent = self.downloadData(start_date, today)
        bitcoin_previous = self.downloadData(
            start_date - datetime.timedelta(days=7), start_date
        )
        bitcoin_previous1 = self.downloadData(
            start_date - datetime.timedelta(days=14),
            start_date - datetime.timedelta(days=7),
        )
        bitcoin_previous2 = self.downloadData(
            start_date - datetime.timedelta(days=21),
            start_date - datetime.timedelta(days=14),
        )

        # Concatenate the dataframes
        bitcoin = pd.concat(
            [
                bitcoin_previous2,
                bitcoin_previous1,
                bitcoin_previous,
                bitcoin_recent,
                bitcoin_today,
            ]
        )
        return bitcoin

    def create_model(self):
        model = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline(
                                    [
                                        ("poly", PolynomialFeatures(degree=3)),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                self.numerical_features,
                            ),
                        ]
                    ),
                ),
                ("regressor", PassiveAggressiveRegressor()),
            ]
        )
        return model

    def downloadData(self, start, end):
        return yf.download(self.stock_name, start=start, end=end, interval="1m")

    def additional_feature_engineering(self, data, real_time_prediction=None):
        data["MA_10"] = data["Close"].rolling(window=10).mean()
        data["Price_Diff"] = data["Close"].diff()
        data["Upper_Band"], data["Middle_Band"], data["Lower_Band"] = self.calculate_bollinger_bands(data["Close"], window=20, num_std=2)
        data["MACD"], data["Signal_Line"] = self.calculate_macd(data["Close"], short_window=12, long_window=26, signal_window=9)
        data["Stochastic_Oscillator"] = self.calculate_stochastic_oscillator(data["Close"], window=14)
        data["Volume_MA"] = data["Volume"].rolling(window=10).mean()
        
        # Add RSI calculation
        data["RSI"] = self.calculate_rsi(data["Close"], window=14)
        
        # Additional features
        data["Aroon_Up"], data["Aroon_Down"] = self.calculate_aroon(data["Close"], window=14)
        data["Fibonacci_Retracement_0.382"], data["Fibonacci_Retracement_0.618"] = self.calculate_fibonacci_retracement(data["Close"])
        data["Ichimoku_Span_A"], data["Ichimoku_Span_B"], data["Ichimoku_Lagging_Span"] = self.calculate_ichimoku_cloud(data)
        data["Volatility"] = self.calculate_volatility(data["Close"], window=20)
        data["ATR"] = self.calculate_atr(data, window=14)

        
        # If real_time_prediction is provided, set the last row's Prediction value
        if real_time_prediction is not None:
            data.at[data.index[-1], "Prediction"] = real_time_prediction

        return data

    # Add ATR calculation to the Trader class
    def calculate_atr(self, series, window=14):
        high_low = series["High"] - series["Low"]
        high_close = np.abs(series["High"] - series["Close"].shift())
        low_close = np.abs(series["Low"] - series["Close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = tr.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    def calculate_aroon(self, series, window=14):
        aroon_up = series.rolling(window=window).apply(lambda x: np.argmax(x) / window * 100, raw=True)
        aroon_down = series.rolling(window=window).apply(lambda x: np.argmin(x) / window * 100, raw=True)
        return aroon_up, aroon_down

    def calculate_fibonacci_retracement(self, series):
        high, low = series.max(), series.min()
        retracement_0_382 = high - 0.382 * (high - low)
        retracement_0_618 = high - 0.618 * (high - low)
        return retracement_0_382, retracement_0_618

    def calculate_ichimoku_cloud(self, data):
        conversion_line = (data["High"].rolling(window=9).max() + data["Low"].rolling(window=9).min()) / 2
        base_line = (data["High"].rolling(window=26).max() + data["Low"].rolling(window=26).min()) / 2
        span_a = (conversion_line + base_line) / 2
        span_b = (data["High"].rolling(window=52).max() + data["Low"].rolling(window=52).min()) / 2
        lagging_span = data["Close"].shift(-26)  # Shift by 26 periods for lagging span
        return span_a, span_b, lagging_span

    def calculate_volatility(self, series, window=20):
        return series.pct_change().rolling(window=window).std() * np.sqrt(252)  # Annualized volatility

    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        middle_band = series.rolling(window=window).mean()
        upper_band = middle_band + num_std * series.rolling(window=window).std()
        lower_band = middle_band - num_std * series.rolling(window=window).std()
        return upper_band, middle_band, lower_band

    def calculate_macd(self, series, short_window=12, long_window=26, signal_window=9):
        short_ema = series.ewm(span=short_window, adjust=False).mean()
        long_ema = series.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal_line

    def calculate_stochastic_oscillator(self, series, window=14):
        lowest_low = series.rolling(window=window).min()
        highest_high = series.rolling(window=window).max()
        stochastic_oscillator = 100 * (series - lowest_low) / (highest_high - lowest_low)
        return stochastic_oscillator

    def calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def trainModel(self):
        # Check if bitcoin DataFrame is available
        if self.bitcoin is None:
            raise ValueError("Bitcoin data not available. Call download method first.")

        # Prepare data for the model
        bitcoin = self.additional_feature_engineering(self.bitcoin)
        bitcoin["Prediction"] = bitcoin["Close"].shift(-1)
        bitcoin.dropna(inplace=True)
        X = np.array(bitcoin.drop(["Prediction", "Close"], axis=1))
        Y = np.array(bitcoin["Close"])

        if len(X) <= 1:
            raise ValueError("Not enough data for training. Try a larger date range.")

        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        self.numerical_features = np.arange(X.shape[1])
        self.model = self.create_model()
        self.model.fit(x_train, y_train)
        self.feature_importance_analysis(x_test, y_test)

    def feature_importance_analysis(self, x_test, y_test):
        if self.model is None:
            print("Model not trained. Feature importance analysis cannot be performed.")
            return

        result = permutation_importance(self.model, x_test, y_test, n_repeats=10, random_state=42)

        feature_importance = result.importances_mean
        feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]

        # Print and visualize feature importance
        print("Feature Importance:")
        for feature, importance in zip(feature_names, feature_importance):
            print(f"{feature}: {importance}")

        # Plotting feature importance
        #self.plot_feature_importance(feature_names, feature_importance)

    def plot_feature_importance(self, feature_names, feature_importance):
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importance)
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance Analysis')
        plt.show()

    def save_model(self, model):
        joblib.dump(model, self.model_filename)
        print(f"Trained model saved to {self.model_filename}")

    # More advanced buying logic
    def should_buy(self, data, predicted_price, current_price, current_rsi, short_window=10, long_window=50, max_risk_pct=2, roc_threshold=5, stop_loss_pct=1, take_profit_pct=2 , atr_multiplier=1.5):
        short_ma = data["Close"].rolling(window=short_window).mean()
        long_ma = data["Close"].rolling(window=long_window).mean()

        roc = (data["Close"] / data["Close"].shift(roc_threshold) - 1) * 100
        
        volatility_threshold = data["ATR"].iloc[-1] * atr_multiplier
        trend_confirmation = short_ma > long_ma

        # Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(data["Close"], window=20, num_std=2)

        # MACD
        macd, signal_line = self.calculate_macd(data["Close"], short_window=12, long_window=26, signal_window=9)

        # Stochastic Oscillator
        stochastic_oscillator = self.calculate_stochastic_oscillator(data["Close"], window=14)

        # Volume Moving Average
        volume_ma = data["Volume_MA"].iloc[-1]

        # Calculate stop-loss and take-profit levels
        stop_loss_price = current_price.iloc[0] * (1 - stop_loss_pct / 100)
        take_profit_price = current_price.iloc[0] * (1 + take_profit_pct / 100)

         # Calculate position size based on risk tolerance
        max_risk_amount = self.portfolio["cash"] * max_risk_pct / 100
        position_size = max_risk_amount / (current_price.iloc[0] - stop_loss_price)

        # Add conditions for buying with stop-loss and take-profit
        conditions = (
            (short_ma > long_ma) & (roc > 0) & (current_rsi < 30) &
            (current_price.iloc[0] < lower_band.iloc[-1]) & (macd > signal_line) & 
            (stochastic_oscillator < 20) & (data["Volume"].iloc[-1] > volume_ma) &
            (predicted_price > take_profit_price) & (position_size > 0)  # Add take-profit and position sizing conditions
        )

        return conditions.iloc[-1], stop_loss_price  # Return stop-loss price along with the buy condition

    # More advanced selling logic
    def should_sell(self, data, predicted_price, current_price, current_rsi, short_window=10, long_window=50, roc_threshold=5, stop_loss_pct=1, take_profit_pct=2, trailing_stop_pct=1, partial_profit_levels=None):
        short_ma = data["Close"].rolling(window=short_window).mean()
        long_ma = data["Close"].rolling(window=long_window).mean()

        roc = (data["Close"] / data["Close"].shift(roc_threshold) - 1) * 100

        trend_confirmation = short_ma < long_ma

        # Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(data["Close"], window=20, num_std=2)

        # MACD
        macd, signal_line = self.calculate_macd(data["Close"], short_window=12, long_window=26, signal_window=9)

        # Stochastic Oscillator
        stochastic_oscillator = self.calculate_stochastic_oscillator(data["Close"], window=14)

        # Volume Moving Average
        volume_ma = data["Volume_MA"].iloc[-1]

        # Calculate stop-loss and take-profit levels
        stop_loss_price = current_price.iloc[0] * (1 + stop_loss_pct / 100)
        take_profit_price = current_price.iloc[0] * (1 - take_profit_pct / 100)

        # Trailing stop-loss
        trailing_stop_price = current_price.iloc[0] * (1 - trailing_stop_pct / 100)
        trailing_stop_condition = current_price.iloc[0] < trailing_stop_price

         # Calculate multiple profit-taking levels
        if partial_profit_levels is not None:
            take_profit_prices = [current_price.iloc[0] * (1 - level * take_profit_pct / 100) for level in partial_profit_levels]
        else:
            take_profit_prices = []

        # Add conditions for selling with stop-loss, take-profit, and trailing stop-loss
        conditions = (
            (short_ma < long_ma) & (roc < 0) & (current_rsi > 70) &
            (current_price.iloc[0] > upper_band.iloc[-1]) & (macd < signal_line) & 
            (stochastic_oscillator > 80) & (data["Volume"].iloc[-1] > volume_ma) &
            ((predicted_price < take_profit_price) or trailing_stop_condition or (predicted_price < min(take_profit_prices)))  # Add take-profit and trailing stop conditions
        )
        
        return conditions.iloc[-1], stop_loss_price  # Return stop-loss price along with the sell condition



    # New method for additional trading strategies
    def additional_strategies(self, data):
        # Moving Average Crossover Strategy
        data["Short_MA"] = data["Close"].rolling(window=10).mean()
        data["Long_MA"] = data["Close"].rolling(window=50).mean()

        # RSI Divergence
        bullish_divergence = (data["Close"] < data["Close"].shift(1)) & (data["RSI"] > data["RSI"].shift(1))
        bearish_divergence = (data["Close"] > data["Close"].shift(1)) & (data["RSI"] < data["RSI"].shift(1))

        # Volume Analysis
        high_volume = data["Volume"] > data["Volume_MA"]

        # Additional Technical Indicators
        # Add your code for other technical indicators

        # Trend Confirmation
        uptrend = data["Short_MA"] > data["Long_MA"]
        downtrend = data["Short_MA"] < data["Long_MA"]

        # Support and Resistance Levels
        support_level = data["Close"].rolling(window=20).min()
        resistance_level = data["Close"].rolling(window=20).max()

        return bullish_divergence, bearish_divergence, high_volume, uptrend, downtrend, support_level, resistance_level

    def start_trading(self):
        in_position = False

        buy_price = 0
        while self.run_thread:
            try:
            # Download current Bitcoin data for the next minute
                bitcoin_current = yf.download(self.stock_name, period="1d", interval="1m")

                # Make real-time prediction
                real_time_prediction = self.model.predict(
                    np.array(
                        self.additional_feature_engineering(bitcoin_current)
                        .tail(1)
                        .drop(["Close"], axis=1)
                    )
                )[0]

                # Print additional information (e.g., RSI)
                current_rsi = bitcoin_current["RSI"].iloc[-1]
                print(f"Current RSI: {current_rsi}")

                # Add the "Prediction" column to the real-time data
                bitcoin_current["Prediction"] = np.nan
                bitcoin_current.at[
                    bitcoin_current.index[-1], "Prediction"
                ] = real_time_prediction

                # Convert time to IST
                bitcoin_current.index = bitcoin_current.index.tz_convert("Asia/Kolkata")

                # Print the current Bitcoin price and predicted price
                current_price_usd = bitcoin_current["Close"].iloc[-1]
                # Additional Strategies
                bullish_divergence, bearish_divergence, high_volume, uptrend, downtrend, support_level, resistance_level = self.additional_strategies(bitcoin_current)
                # Buying logic
                if (
                    self.should_buy(data=bitcoin_current, predicted_price=real_time_prediction, current_price=bitcoin_current["Close"], current_rsi=current_rsi)
                    and not in_position
                    and bullish_divergence.iloc[-1] 
                    and high_volume.iloc[-1] 
                    and uptrend.iloc[-1]
                ):
                    print(colored(f"Buying {self.stock_name} @ {str(current_price_usd)}", "green"))
                    in_position = True
                    buy_price = current_price_usd
                    btc_units_bought = self.portfolio["cash"] / current_price_usd
                    self.portfolio["cash"] -= btc_units_bought * buy_price
                    self.portfolio[f"{self.stock_name}_units"] += btc_units_bought
                    self.portfolio_manager.buy(self.stock_name, btc_units_bought, buy_price)
                    order = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "buy",
                        "price": buy_price,
                        "stock": self.stock_name,
                        "quantity": btc_units_bought,
                    }
                    
                    self.trading_data["cash"] -= btc_units_bought * buy_price
                    self.trading_data["units"][f"{self.stock_name}"] += btc_units_bought
                    
                    self.order_book.append(order)

                # Selling logic
                if (
                    self.should_sell(data=bitcoin_current, predicted_price=real_time_prediction, current_price=bitcoin_current["Close"], current_rsi=current_rsi)
                    and in_position
                    and bearish_divergence.iloc[-1] 
                    and high_volume.iloc[-1] 
                    and downtrend.iloc[-1]
                ):
                    print("Selling...")
                    print("Bought @ " + str(buy_price))
                    print(colored(f"Selling {self.stock_name}  @  {str(current_price_usd)}", "red"))
                    profit_loss = (
                        current_price_usd * self.portfolio[f"{self.stock_name}_units"] - self.initial_cash
                    )
                    print(f"Profit/Loss: {profit_loss}")
                    quantity = self.portfolio[f"{self.stock_name}_units"]
                    self.portfolio_manager.sell(self.stock_name, quantity, current_price_usd)
                    self.portfolio["cash"] += current_price_usd * quantity
                    self.portfolio[f"{self.stock_name}_units"] = 0
                    in_position = False
                    order = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "sell",
                        "price": current_price_usd,
                        "stock": self.stock_name,
                        "quantity": quantity,
                    }
                    
                    self.trading_data["cash"] += current_price_usd * quantity
                    self.trading_data["units"][f"{self.stock_name}"] -= quantity
                    
                    self.order_book.append(order)

                # Print portfolio status
                print(
                    colored(
                        f"Portfolio: Cash = {self.portfolio['cash']}, {self.stock_name} Units = {self.portfolio[f"{self.stock_name}_units"]}",
                        "yellow",
                    )
                )

                current_time = bitcoin_current.index[-1]
                print(f"Latest Price of {self.stock_name} : {current_price_usd} at {current_time}")
                print(f"Predicted Price for the next minute: {real_time_prediction}")

                # Emit portfolio update through WebSocket
                timestamp = str(bitcoin_current.index[-1])
                profit = self.initial_cash - 10000  # currentcash - initial_cash
                price_update = (
                    {
                        "timestamp": timestamp,
                        "latestPrice": current_price_usd,
                        "predictedPrice": real_time_prediction,
                        "rsi": current_rsi,
                        "inposition": in_position,
                        "profit": profit,
                    },
                )
                
                self.save_order_book()
                self.save_trading_data()
                # Print portfolio status
                self.portfolio_manager.print_portfolio_status()

                time.sleep(10)
            except KeyboardInterrupt:
                print("Trading interrupted by user.")
                self.run_thread = False
                self.save_order_book()
                self.save_trading_data()
                break
            
    def run(self):
        if threading.active_count() == 1:
            thread = threading.Thread(target=self.start_trading)
            thread.start()
            thread.join()

def test_trader_functionalities():
    # Create a Trader instance
    trader = Trader("BTC-USD")

    # Test downloading data
    data = trader.bitcoin
    assert not data.empty

    # Test model training
    trader.trainModel()
    assert trader.model is not None

    data = trader.additional_feature_engineering(data)
    assert "MA_10" in data.columns

    # Test buying and selling conditions
    buy_condition = trader.should_buy(data, predicted_price=5000, current_price=data["Close"].head(1), current_rsi=30)
    assert isinstance(buy_condition, (bool, np.bool_))

    sell_condition = trader.should_sell(data, predicted_price=6000, current_price=data["Close"].head(1), current_rsi=70)
    assert isinstance(sell_condition, (bool, np.bool_))

    # Test additional strategies
    additional_strategies_data = trader.additional_strategies(data.head(5))
    assert all(isinstance(strategy, pd.Series) for strategy in additional_strategies_data)

def backtest_strategy(trader, historical_data):
    for i in range(len(historical_data)):
        current_data = historical_data.iloc[: i + 1]
        buy_condition = trader.should_buy(current_data, predicted_price=5000, current_price=current_data["Close"].head(1), current_rsi=30)
        sell_condition = trader.should_sell(current_data, predicted_price=6000, current_price=current_data["Close"].head(1), current_rsi=70)

        if buy_condition:
            trader.portfolio["cash"] -= current_data["Close"].iloc[-1]
            trader.portfolio[f"{trader.stock_name}_units"] += 1

        if sell_condition:
            trader.portfolio["cash"] += current_data["Close"].iloc[-1] * trader.portfolio[f"{trader.stock_name}_units"]
            trader.portfolio[f"{trader.stock_name}_units"] = 0


if __name__ == "__main__":
    #test_trader_functionalities()
    stock_list = {
        "crypto": ["BTC-INR", "ETH-INR", "XRP-INR", "LTC-INR", "BCH-INR", "ADA-INR", "DOT-INR", "BNB-INR", "LINK-INR", "XLM-INR"],
        "US": ["GOOGL", "AAPL", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "JNJ", "JPM", "V"],
        "IN": ["RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFC.BO", "ITC.BO", "SBIN.BO", "LT.BO", "HINDUNILVR.BO", "AXISBANK.BO", "BHARTIARTL.BO"]
    }
    bitcoin_trader = Trader("BTC-USD")
    bitcoin_trader.run()
    #historical_data = bitcoin_trader.bitcoin
    #backtest_strategy(bitcoin_trader, historical_data)
    #print("Backtesting completed. Portfolio after backtesting:", bitcoin_trader.portfolio)

