# # import pandas as pd
# # import numpy as np
# # import ta
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report
# # from imblearn.over_sampling import SMOTE
# # import matplotlib.pyplot as plt
# # import yfinance as yf
# # import matplotlib.dates as mdates
# # import pandas as pd
# # import matplotlib.dates as mdates
# # import matplotlib.pyplot as plt

# # def filter_data_by_date(df, start_date, end_date):
# #     """
# #     Filter the dataframe to only include data between start_date and end_date.
# #     """
# #     return df[(df.index >= start_date) & (df.index <= end_date)]

# # # 1. Load and Prepare Data
# # def load_data(ticker, start_date, end_date):
# #     """
# #     Fetch historical stock data using yfinance.
# #     :param ticker: stock ticker symbol (e.g., 'SPY' for S&P 500 ETF)
# #     :param start_date: start date in 'YYYY-MM-DD' format
# #     :param end_date: end date in 'YYYY-MM-DD' format
# #     :return: DataFrame containing historical data
# #     """
# #     df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
# #     df = df.reset_index()  # Reset index to have 'Date' as a column
# #     df.set_index('Date', inplace=True)  # Make 'Date' the index again

# #     # Calculate the percentage change (Return) in the 'Close' prices
# #     df['Return'] = df['Close'].pct_change()

# #     return df


# # # 2. Label Flash Crashes
# # def label_flash_crashes(df, drop_threshold=0.05, rebound_threshold=0.03, time_window=60):
# #     """
# #     Labels flash crashes by identifying a sharp drop followed by a rebound in the given time window.
# #     drop_threshold: percentage drop (e.g., 0.05 for 5%)
# #     rebound_threshold: percentage rebound (e.g., 0.03 for 3%)
# #     time_window: number of minutes (assuming minute-level data)
# #     """
# #     df['Flash_Crash'] = 0
# #     for i in range(len(df) - time_window*2):
# #         window_drop = df['Close'].iloc[i:i+time_window].pct_change().min()
# #         if window_drop <= -drop_threshold:
# #             window_rebound = df['Close'].iloc[i+time_window:i+2*time_window].pct_change().max()
# #             if window_rebound >= rebound_threshold:
# #                 df['Flash_Crash'].iloc[i:i+time_window] = 1
# #     return df

# # # 3. Create Volatility Features
# # def create_volatility_features(df):
# #     """
# #     Adds volatility-based technical indicators to the dataframe.
# #     """
# #     df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
# #     df['Bollinger_High'] = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_hband()
# #     df['Bollinger_Low'] = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_lband()
# #     df['Volatility_Zscore'] = (df['ATR'] - df['ATR'].mean()) / df['ATR'].std()
# #     return df

# # # 4. Feature Engineering
# # def feature_engineering(df):
# #     """
# #     Creates additional features for the model.
# #     """
# #     df['Return'] = df['Close'].pct_change()
# #     df['Lag1_Return'] = df['Return'].shift(1)
# #     df['Lag2_Return'] = df['Return'].shift(2)
# #     df['Rolling_Mean_5'] = df['Close'].rolling(window=5).mean()
# #     df['Rolling_STD_5'] = df['Close'].rolling(window=5).std()
# #     df = create_volatility_features(df)
# #     df.dropna(inplace=True)
# #     return df

# # # 5. Prepare Dataset for Modeling
# # def prepare_dataset(df):
# #     """
# #     Prepares feature matrix X and target vector y.
# #     """
# #     feature_cols = ['ATR', 'Bollinger_High', 'Bollinger_Low', 'Volatility_Zscore',
# #                    'Lag1_Return', 'Lag2_Return', 'Rolling_Mean_5', 'Rolling_STD_5']
# #     X = df[feature_cols]
# #     y = df['Flash_Crash']
# #     return X, y

# # # 6. Handle Class Imbalance and Train Model
# # def train_model(X, y):
# #     """
# #     Trains a Random Forest model with SMOTE for handling class imbalance.
# #     """
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    
# #     # Apply SMOTE
# #     smote = SMOTE(sampling_strategy='auto', random_state=42)
# #     X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
# #     # Initialize and train the model
# #     model = RandomForestClassifier(n_estimators=100, random_state=42)
# #     model.fit(X_resampled, y_resampled)
    
# #     # Predictions
# #     y_pred = model.predict(X_test)
    
# #     # Evaluation
# #     print("Classification Report:")
# #     print(classification_report(y_test, y_pred))
    
# #     return model, X_test, y_test, y_pred

# # # 7. Implement Dynamic Hedging Strategy
# # def dynamic_hedging(df, model, X):
# #     """
# #     Implements dynamic hedging based on model predictions.
# #     When a flash crash is predicted, shift part of the portfolio to a hedging asset.
# #     """
# #     df = df.copy()
# #     df['Prediction'] = model.predict(X)
# #     df['Hedged_Return'] = np.where(df['Prediction'] == 1, 
# #                                     df['Return'] * 0.5 + 0.02,  # Example: 50% in risky asset, 50% in hedging asset with 2% return
# #                                     df['Return'])
# #     return df

# # # 8. Backtest the Strategy

# # def backtest_strategy(df):
# #     """
# #     Enhanced backtest visualization with more evaluations.
# #     Plots original vs hedged returns, drawdown, and volatility.
# #     """
# #     df['Cumulative_Return'] = (1 + df['Return']).cumprod()
# #     df['Cumulative_Hedged_Return'] = (1 + df['Hedged_Return']).cumprod()

# #     # Calculate drawdowns
# #     cumulative = df['Cumulative_Hedged_Return']
# #     peak = cumulative.expanding(min_periods=1).max()
# #     df['Drawdown'] = (cumulative - peak) / peak

# #     # Set up the plot with subplots
# #     fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# #     # Plot cumulative returns
# #     ax[0].plot(df.index, df['Cumulative_Return'], label='Original Strategy', color='blue', alpha=0.6)
# #     ax[0].plot(df.index, df['Cumulative_Hedged_Return'], label='Hedged Strategy', color='green', alpha=0.8)
# #     ax[0].set_title('Cumulative Returns')
# #     ax[0].set_ylabel('Cumulative Return')
# #     ax[0].legend()

# #     # Plot drawdowns
# #     ax[1].plot(df.index, df['Drawdown'], label='Drawdown (Hedged)', color='red', alpha=0.6)
# #     ax[1].set_title('Drawdown')
# #     ax[1].set_ylabel('Drawdown Percentage')
# #     ax[1].legend()

# #     # Plot volatility (ATR)
# #     ax[2].plot(df.index, df['ATR'], label='ATR (Volatility)', color='purple', alpha=0.6)
# #     ax[2].set_title('Volatility (ATR)')
# #     ax[2].set_ylabel('ATR Value')
# #     ax[2].legend()

# #     # Fix x-axis labeling for dates
# #     for axis in ax:
# #         axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
# #         axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# #         axis.tick_params(axis='x', rotation=45)

# #     plt.tight_layout()
# #     plt.show()


# # # 9. Calculate Risk-Adjusted Metrics
# # def calculate_risk_adjusted_metrics(df):
# #     """
# #     Calculates Sharpe Ratio, Sortino Ratio, and Maximum Drawdown for the hedged strategy.
# #     """
# #     strategy_returns = df['Hedged_Return']
# #     sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)  # Assuming daily returns
# #     sortino_ratio = (strategy_returns.mean() / strategy_returns[strategy_returns < 0].std()) * np.sqrt(252)
# #     cumulative = (1 + strategy_returns).cumprod()
# #     peak = cumulative.expanding(min_periods=1).max()
# #     drawdown = (cumulative - peak) / peak
# #     max_drawdown = drawdown.min()
    
# #     print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
# #     print(f"Sortino Ratio: {sortino_ratio:.2f}")
# #     print(f"Maximum Drawdown: {max_drawdown:.2%}")

# # # 10. Analyze Performance During Flash Crashes and Surrounding Weeks

# # def plot_flash_crash_performance(df, flash_crash_dates):
# #     """
# #     Plot data from 2008 to current, highlighting flash crashes.
# #     """
# #     # Filter data from 2008 to the present
# #     df = filter_data_by_date(df, '2008-01-01', '2023-01-01')

# #     df['Cumulative_Return'] = (1 + df['Return']).cumprod()
# #     df['Cumulative_Hedged_Return'] = (1 + df['Hedged_Return']).cumprod()
    
# #     # Calculate drawdowns
# #     cumulative = df['Cumulative_Hedged_Return']
# #     peak = cumulative.expanding(min_periods=1).max()
# #     df['Drawdown'] = (cumulative - peak) / peak
    
# #     fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# #     # Plot cumulative returns
# #     ax[0].plot(df.index, df['Cumulative_Return'], label='Original Strategy', color='blue', alpha=0.6)
# #     ax[0].plot(df.index, df['Cumulative_Hedged_Return'], label='Hedged Strategy', color='green', alpha=0.8)
# #     ax[0].set_title('Cumulative Returns (2008-Present)')
# #     ax[0].set_ylabel('Cumulative Return')
# #     ax[0].legend()

# #     # Highlight flash crash events
# #     for crash_date in flash_crash_dates:
# #         ax[0].axvline(crash_date, color='red', linestyle='--', label=f'Flash Crash {crash_date.year}')

# #     # Plot drawdowns
# #     ax[1].plot(df.index, df['Drawdown'], label='Drawdown (Hedged)', color='red', alpha=0.6)
# #     ax[1].set_title('Drawdown')
# #     ax[1].set_ylabel('Drawdown Percentage')
# #     ax[1].legend()

# #     # Plot volatility (ATR)
# #     ax[2].plot(df.index, df['ATR'], label='ATR (Volatility)', color='purple', alpha=0.6)
# #     ax[2].set_title('Volatility (ATR)')
# #     ax[2].set_ylabel('ATR Value')
# #     ax[2].legend()

# #     # Highlight flash crash events in drawdown and volatility plots
# #     for crash_date in flash_crash_dates:
# #         ax[1].axvline(crash_date, color='red', linestyle='--', label=f'Flash Crash {crash_date.year}')
# #         ax[2].axvline(crash_date, color='red', linestyle='--', label=f'Flash Crash {crash_date.year}')

# #     # Fix x-axis labeling for dates
# #     for axis in ax:
# #         axis.xaxis.set_major_locator(mdates.YearLocator())
# #         axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# #         axis.tick_params(axis='x', rotation=45)

# #     plt.tight_layout()
# #     plt.show()
# # # Define known flash crash dates for analysis
# # flash_crash_dates = [
# #     pd.Timestamp('2008-09-29'),  # 2008 Global Financial Crisis
# #     pd.Timestamp('2010-05-06'),  # May 6, 2010 Flash Crash
# #     pd.Timestamp('2020-03-16'),  # COVID-19 Crash
# #     # Add more dates as needed
# # ]

# # def main():
# #     # Define the ticker, start date, and end date for the historical data
# #     ticker = 'SPY'  # Example: SPY is the ticker for the S&P 500 ETF
# #     start_date = '2008-01-01'
# #     end_date = '2023-01-01'

# #     # Fetch data from yfinance and calculate returns
# #     df = load_data(ticker, start_date, end_date)

# #     # Label flash crashes in the dataset
# #     df = label_flash_crashes(df, drop_threshold=0.05, rebound_threshold=0.03, time_window=60)

# #     # Feature engineering (this will create necessary columns for the model)
# #     df = feature_engineering(df)

# #     # Prepare dataset for modeling (X = features, y = target 'Flash_Crash')
# #     X, y = prepare_dataset(df)

# #     # Train model to predict flash crashes
# #     model, X_test, y_test, y_pred = train_model(X, y)

# #     # Apply the dynamic hedging strategy to generate 'Hedged_Return'
# #     df_hedged = dynamic_hedging(df, model, X)

# #     # Define known flash crash dates
# #     flash_crash_dates = [
# #         pd.Timestamp('2008-09-29'),  # 2008 Global Financial Crisis
# #         pd.Timestamp('2010-05-06'),  # May 6, 2010 Flash Crash
# #         pd.Timestamp('2020-03-16'),  # COVID-19 Crash
# #     ]

# #     # Call the function to plot data from 2008 to the present and highlight flash crashes
# #     plot_flash_crash_performance(df_hedged, flash_crash_dates)


# # # Now we only call the main function here
# # if __name__ == "__main__":
# #     main()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import yfinance as yf
# import ta  # Ensure ta-lib (technical analysis) library is installed
# import matplotlib.dates as mdates
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE

# # 1. Load and Prepare Data (assuming you already have the data fetching function)
# def load_data(ticker, start_date, end_date):
#     df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
#     df = df.reset_index()
#     df.set_index('Date', inplace=True)
#     df['Return'] = df['Close'].pct_change()
#     return df

# # 2. Dynamic Hedging Strategy
# def dynamic_hedging(df, model, X):
#     """
#     Implements dynamic hedging based on model predictions.
#     When a flash crash is predicted, shift part of the portfolio to a hedging asset.
#     """
#     df = df.copy()
#     df['Prediction'] = model.predict(X)
#     df['Hedged_Return'] = np.where(df['Prediction'] == 1, 
#                                     df['Return'] * 0.5 + 0.02,  # Example: 50% in risky asset, 50% in hedging asset with 2% return
#                                     df['Return'])
#     return df

# # 3. Inverse ETF Strategy Simulation
# # 3. Inverse ETF Strategy Simulation
# def inverse_etf_strategy(df, crash_dates, inverse_etf_return=0.05, normal_return=0):
#     """
#     Implements the inverse ETF strategy where during flash crashes we assume a positive return (e.g., +5%),
#     and during normal times, we assume no change or a small positive return.
#     """
#     df['Inverse_ETF_Return'] = normal_return  # Apply a small return during non-crash periods
    
#     # During flash crash periods, apply the inverse ETF return
#     for crash in crash_dates:
#         df.loc[crash, 'Inverse_ETF_Return'] = inverse_etf_return
    
#     df['Cumulative_Inverse_ETF_Return'] = (1 + df['Inverse_ETF_Return']).cumprod()
#     return df

# # 4. Tail Risk Hedging Strategy Simulation
# def tail_risk_hedging(df, crash_dates, tail_hedge_return=0.08, normal_return=0):
#     """
#     Implements the tail risk hedging strategy where during flash crashes we assume a positive return (e.g., +8%),
#     and during normal times, we assume no change or a small positive return.
#     """
#     df['Tail_Hedge_Return'] = normal_return  # Apply a small return during non-crash periods
    
#     # During flash crash periods, apply the tail risk hedge return
#     for crash in crash_dates:
#         df.loc[crash, 'Tail_Hedge_Return'] = tail_hedge_return
    
#     df['Cumulative_Tail_Hedge_Return'] = (1 + df['Tail_Hedge_Return']).cumprod()
#     return df

# # 5. Feature Engineering (include ATR calculation here)
# def feature_engineering(df):
#     """
#     Adds volatility-based features, including ATR, to the DataFrame.
#     """
#     # Calculate ATR (Average True Range)
#     df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
#     # Add other features if needed (here we're only focusing on ATR)
#     return df

# def plot_strategies(df):
#     # Calculate cumulative returns
#     df['Cumulative_Return'] = (1 + df['Return']).cumprod()
#     df['Cumulative_Hedged_Return'] = (1 + df['Hedged_Return']).cumprod()
#     df['Cumulative_Inverse_ETF_Return'] = (1 + df['Inverse_ETF_Return']).cumprod()
#     df['Cumulative_Tail_Hedge_Return'] = (1 + df['Tail_Hedge_Return']).cumprod()

#     # Set up the plot with subplots
#     fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

#     # Plot cumulative returns
#     ax[0].plot(df.index, df['Cumulative_Return'], label='Original Strategy', color='blue', alpha=0.6)
#     ax[0].plot(df.index, df['Cumulative_Hedged_Return'], label='Dynamic Hedging', color='green', alpha=0.8)
#     ax[0].plot(df.index, df['Cumulative_Inverse_ETF_Return'], label='Inverse ETF Strategy', color='orange', alpha=0.8)
#     ax[0].plot(df.index, df['Cumulative_Tail_Hedge_Return'], label='Tail Risk Hedging', color='red', alpha=0.8)
#     ax[0].set_title('Cumulative Returns for Different Strategies')
#     ax[0].set_ylabel('Cumulative Return')
#     ax[0].legend()

#     # Plot drawdowns or ATR (Volatility)
#     ax[1].plot(df.index, df['ATR'], label='ATR (Volatility)', color='purple', alpha=0.6)
#     ax[1].set_title('Volatility (ATR)')
#     ax[1].set_ylabel('ATR Value')
#     ax[1].legend()

#     # Fix x-axis labeling for dates
#     for axis in ax:
#         axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
#         axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         axis.tick_params(axis='x', rotation=45)

#     plt.tight_layout()
#     plt.show()

# def main():
#     # Define the ticker, start date, and end date for the historical data
#     ticker = 'SPY'
#     start_date = '2008-01-01'
#     end_date = '2023-01-01'

#     # Load Data
#     df = load_data(ticker, start_date, end_date)

#     # Feature engineering (add ATR)
#     df = feature_engineering(df)

#     # Assume we have a pre-trained model and dataset (You would replace this with real model training)
#     X = np.random.randn(len(df), 5)  # Example features, replace with actual features
#     y = np.random.randint(0, 2, len(df))  # Example binary target, replace with actual target

#     # Train a simple RandomForest model (for demo purposes)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y)

#     # Apply Dynamic Hedging strategy
#     df = dynamic_hedging(df, model, X)

#     # Define flash crash dates (you can modify these)
#     flash_crash_dates = [
#         pd.Timestamp('2008-09-29'),  # 2008 Global Financial Crisis
#         pd.Timestamp('2010-05-06'),  # May 6, 2010 Flash Crash
#         pd.Timestamp('2020-03-16')   # COVID-19 Crash
#     ]

#     # Apply Inverse ETF strategy
#     df = inverse_etf_strategy(df, flash_crash_dates)

#     # Apply Tail Risk Hedging strategy
#     df = tail_risk_hedging(df, flash_crash_dates)

#     # Plot results
#     plot_strategies(df)

# # Call the main function
# if __name__ == "__main__":
#     main()
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load Data
def load_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df = df.reset_index()  # Reset index to have 'Date' as a column
    df.set_index('Date', inplace=True)  # Make 'Date' the index again
    df['Return'] = df['Close'].pct_change()  # Calculate percentage change in 'Close' prices
    return df

# 2. Inverse ETF Strategy
def inverse_etf_strategy(df, crash_dates, inverse_etf_return=0.05, normal_return=0):
    """
    Simulates an inverse ETF strategy during flash crash periods.
    """
    df['Inverse_ETF_Return'] = normal_return  # Normal return during non-crash periods
    for crash in crash_dates:
        df.loc[crash, 'Inverse_ETF_Return'] = inverse_etf_return  # Positive return during crash periods
    df['Cumulative_Inverse_ETF_Return'] = (1 + df['Inverse_ETF_Return']).cumprod()
    return df

# 3. Plot Inverse ETF Strategy
def plot_inverse_etf_strategy(df, flash_crash_dates):
    """
    Plots the behavior of the Inverse ETF strategy during flash crash dates.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot cumulative returns for Inverse ETF strategy
    ax.plot(df.index, df['Cumulative_Inverse_ETF_Return'], label='Inverse ETF Strategy', color='orange')

    # Highlight flash crash dates with vertical lines
    for crash in flash_crash_dates:
        ax.axvline(crash, color='red', linestyle='--', label=f'Flash Crash {crash.year}')

    # Formatting the plot
    ax.set_title('Inverse ETF Strategy Behavior During Flash Crashes')
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Date')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Main Function
def main():
    ticker = 'SPY'  # Example ticker for S&P 500
    start_date = '2008-01-01'
    end_date = '2023-01-01'
    
    # Load historical data for S&P 500
    df = load_data(ticker, start_date, end_date)
    
    # Define flash crash dates
    flash_crash_dates = [
        pd.Timestamp('2008-09-29'),  # Global Financial Crisis
        pd.Timestamp('2010-05-06'),  # May 6, 2010 Flash Crash
        pd.Timestamp('2020-03-16')   # COVID-19 Crash
    ]
    
    # Apply the inverse ETF strategy
    df = inverse_etf_strategy(df, flash_crash_dates)
    
    # Plot the inverse ETF strategy during flash crashes
    plot_inverse_etf_strategy(df, flash_crash_dates)

# Call the main function
if __name__ == "__main__":
    main()
