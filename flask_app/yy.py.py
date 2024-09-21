# Comprehensive Trading Model Script
# ===================================
# Enhanced to include top 50 stocks of India and Japan, additional technical indicators,
# and comprehensive visualization of multiple strategies.
from sklearn.impute import SimpleImputer
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import multiprocessing as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ===================================
# 1. Data Collection and Integration
# ===================================
def download_ticker(ticker, start_date, end_date):
    """
    Downloads data for a single ticker and returns an empty dataframe if data cannot be fetched.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if data.empty:
            logging.warning(f"No data fetched for {ticker}.")
        return ticker, data
    except Exception as e:
        logging.error(f"Error downloading {ticker}: {e}")
        return ticker, pd.DataFrame()
def fetch_market_data_parallel(tickers, start_date, end_date):
    """
    Fetches market data in parallel.
    """
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(download_ticker, args=(ticker, start_date, end_date)) for ticker in tickers]
    pool.close()
    pool.join()
    
    data = {}
    for res in results:
        ticker, ticker_data = res.get()
        if not ticker_data.empty:
            data[ticker] = ticker_data
    return data

def fetch_market_data_all(tickers_dict, start_date, end_date):
    """
    Fetches historical data for all tickers using yfinance in parallel and renames columns accordingly.
    """
    all_tickers = []
    for tickers in tickers_dict.values():
        if isinstance(tickers, list):
            all_tickers.extend(tickers)
        else:
            all_tickers.append(tickers)

    logging.info("Starting parallel data download...")
    data = fetch_market_data_parallel(all_tickers, start_date, end_date)
    logging.info("Data download complete.")

    combined_data = pd.DataFrame()
    for ticker, ticker_data in data.items():
        if ticker_data.empty:
            logging.warning(f"No data available for {ticker}. Skipping this ticker.")
            continue
        # Rename columns to include ticker symbol
        ticker_data = ticker_data.rename(columns=lambda x: f"{ticker}_{x}")
        if combined_data.empty:
            combined_data = ticker_data
        else:
            combined_data = combined_data.join(ticker_data, how='outer')

    # Remove duplicate columns if any
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
    logging.info(f"Combined data shape: {combined_data.shape}")
    return combined_data

# ===================================
# 2. Feature Engineering with Leading Indicators
# ===================================

def add_rsi(df, column, window=14):
    """
    Adds RSI indicator to the DataFrame, ensuring that the column passed is a single series.
    """
    if df[column].ndim > 1:
        df[column] = df[column].iloc[:, 0]
    rsi = ta.momentum.RSIIndicator(close=df[column], window=window)
    df[f'RSI_{column}'] = rsi.rsi()
    return df

def add_price_rate_of_change(df, column, window=12):
    """
    Adds Price Rate of Change indicator to the DataFrame.
    """
    roc = ta.momentum.ROCIndicator(close=df[column], window=window)
    df[f'ROC_{window}_{column}'] = roc.roc()
    return df

def add_bollinger_bands(df, column, window=20, window_dev=2):
    """
    Adds Bollinger Bands to the DataFrame.
    """
    bollinger = ta.volatility.BollingerBands(close=df[column], window=window, window_dev=window_dev)
    df[f'Bollinger_High_{column}'] = bollinger.bollinger_hband()
    df[f'Bollinger_Low_{column}'] = bollinger.bollinger_lband()
    df[f'Bollinger_Middle_{column}'] = bollinger.bollinger_mavg()
    return df

def add_moving_averages(df, column, windows=[20, 50, 200]):
    """
    Adds Simple Moving Averages to the DataFrame.
    """
    for window in windows:
        df[f'SMA_{window}_{column}'] = df[column].rolling(window=window).mean()
    return df

def add_exponential_moving_averages(df, column, windows=[12, 26]):
    """
    Adds Exponential Moving Averages to the DataFrame.
    """
    for window in windows:
        df[f'EMA_{window}_{column}'] = df[column].ewm(span=window, adjust=False).mean()
    return df

def add_macd(df, column):
    """
    Adds MACD indicators to the DataFrame.
    """
    macd = ta.trend.MACD(close=df[column])
    df[f'MACD_{column}'] = macd.macd()
    df[f'MACD_Signal_{column}'] = macd.macd_signal()
    df[f'MACD_Diff_{column}'] = macd.macd_diff()
    return df

def add_stochastic_oscillator(df, column, window=14, smooth_window=3):
    """
    Adds Stochastic Oscillator to the DataFrame.
    """
    base_ticker = column.split('_')[0]
    high_col = f"{base_ticker}_High"
    low_col = f"{base_ticker}_Low"
    if high_col in df.columns and low_col in df.columns:
        stoch = ta.momentum.StochasticOscillator(
            high=df[high_col],
            low=df[low_col],
            close=df[column],
            window=window,
            smooth_window=smooth_window
        )
        df[f'Stoch_%K_{column}'] = stoch.stoch()
        df[f'Stoch_%D_{column}'] = stoch.stoch_signal()
    else:
        logging.warning(f"Stochastic Oscillator cannot be added for {base_ticker} due to missing high/low columns.")
    return df

def add_atr(df, column, high_column, low_column, window=14):
    """
    Adds Average True Range to the DataFrame.
    """
    atr = ta.volatility.AverageTrueRange(
        high=df[high_column], low=df[low_column], close=df[column], window=window
    )
    df[f'ATR_{column}'] = atr.average_true_range()
    return df

def add_cci(df, column, high_column, low_column, window=20):
    """
    Adds Commodity Channel Index to the DataFrame.
    """
    cci = ta.trend.CCIIndicator(
        high=df[high_column], low=df[low_column], close=df[column], window=window
    )
    df[f'CCI_{column}'] = cci.cci()
    return df

def create_lag_features(df, lag=1):
    """
    Creates lag features for the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with original features.
        lag (int): Number of lag days.

    Returns:
        pd.DataFrame: DataFrame with lag features.
    """
    lag_df = df.shift(lag).add_suffix(f'_lag{lag}')
    return pd.concat([df, lag_df], axis=1)
def feature_engineering_extended(df, tickers, nan_threshold=0.3):
    """
    Extends the feature engineering process by adding multiple technical indicators for multiple markets.
    Skips tickers with excessive missing data and only drops rows if missing data is above a given threshold.
    """
    initial_shape = df.shape
    logging.info(f"Initial data shape before feature engineering: {initial_shape}")

    processed_tickers = 0
    skipped_tickers = 0
    max_nan_allowed = nan_threshold  # Threshold for skipping tickers with too many NaNs

    for region, tickers_list in tickers.items():
        if isinstance(tickers_list, list):
            for ticker in tickers_list:
                close_col = f"{ticker}_Close"
                high_col = f"{ticker}_High"
                low_col = f"{ticker}_Low"

                logging.info(f"Processing ticker: {ticker}, close_col: {close_col}, high_col: {high_col}, low_col: {low_col}")

                if close_col not in df.columns or high_col not in df.columns or low_col not in df.columns:
                    logging.warning(f"Skipping {ticker}. Required columns are missing.")
                    skipped_tickers += 1
                    continue

                # Check NaN percentage before applying indicators
                nan_percentage = df[[close_col, high_col, low_col]].isna().mean().mean()
                if nan_percentage > max_nan_allowed:
                    logging.warning(f"Skipping {ticker}. Too many NaNs: {nan_percentage:.2%}")
                    skipped_tickers += 1
                    continue

                # Adding technical indicators
                df = add_rsi(df, column=close_col)
                df = add_price_rate_of_change(df, column=close_col)
                df = add_bollinger_bands(df, column=close_col)
                df = add_moving_averages(df, column=close_col)
                df = add_exponential_moving_averages(df, column=close_col)
                df = add_macd(df, column=close_col)
                df = add_stochastic_oscillator(df, column=close_col)
                df = add_atr(df, column=close_col, high_column=high_col, low_column=low_col)
                processed_tickers += 1

    # Create lag features
    df = create_lag_features(df, lag=1)

    # Logging missing values before handling
    missing_before = df.isna().sum().sum()
    logging.info(f"Total missing values before handling NaNs: {missing_before}")

    # Fill missing values using forward/backward fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Calculate remaining missing values
    missing_after = df.isna().sum().sum()
    logging.info(f"Total missing values after filling: {missing_after}")

    # Apply threshold to drop rows with remaining NaNs (only drop if above threshold)
    row_nan_threshold = nan_threshold  # e.g., drop rows with more than 30% NaNs
    valid_rows = df.isna().mean(axis=1) < row_nan_threshold
    df = df[valid_rows]

    final_shape = df.shape
    logging.info(f"Final data shape after feature engineering: {final_shape}")
    logging.info(f"Processed tickers: {processed_tickers}, Skipped tickers: {skipped_tickers}")

    if df.empty:
        logging.error("No data left after feature engineering.")
        return pd.DataFrame()  # Return empty DataFrame if no data left

    return df

# ===================================
# 3. Target Definition Function
# ===================================

def define_target(df, target_column='SPY_Close', threshold=0.001):
    """
    Defines the target variable based on price movement.

    Parameters:
        df (pd.DataFrame): DataFrame with engineered features.
        target_column (str): Column name for the target variable.
        threshold (float): Threshold for defining a significant price movement.

    Returns:
        pd.DataFrame: DataFrame with the target variable.
    """
    if target_column not in df.columns:
        logging.error(f"Target column {target_column} not found in the DataFrame.")
        return df

    # Calculate percentage change
    df['Pct_Change'] = df[target_column].pct_change().shift(-1)

    # Define target
    df['Target'] = np.where(df['Pct_Change'] > threshold, 1, 0)

    # Drop rows with NaN values resulting from pct_change
    df.dropna(subset=['Target'], inplace=True)

    # Log class distribution
    class_counts = df['Target'].value_counts()
    logging.info(f"Class distribution after target definition:\n{class_counts}")

    return df

# ===================================
# 4. Splitting Data into Training and Testing Sets
# ===================================

def split_data(df, test_size=0.2):
    """
    Splits the data into training and testing sets based on time.

    Parameters:
        df (pd.DataFrame): DataFrame with all data.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing DataFrames.
    """
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    logging.info(f"Training data shape: {train.shape}")
    logging.info(f"Testing data shape: {test.shape}")
    return train, test

# ===================================
# 5. Preparing Features and Targets
# ===================================

def prepare_features_targets(train, test, target_column='Target'):
    """
    Prepares feature matrices and target vectors, ensuring the columns remain consistent after imputation.
    
    Parameters:
        train (pd.DataFrame): Training DataFrame.
        test (pd.DataFrame): Testing DataFrame.
        target_column (str): Column name for the target variable.
        
    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    # Impute any remaining missing values
    imputer = SimpleImputer(strategy='mean')
    
    # Perform imputation
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Create DataFrames with the original column names
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns[:X_train_imputed.shape[1]])
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns[:X_test_imputed.shape[1]])
    
    return X_train_imputed, y_train, X_test_imputed, y_test

# ===================================
# 6. Handling Class Imbalance with SMOTE
# ===================================

def balance_classes(X, y):
    """
    Balances the classes using SMOTE.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        tuple: Resampled X and y.
    """
    smote = SMOTE(random_state=42, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X, y)
    logging.info(f"After SMOTE, X_res shape: {X_res.shape}")
    logging.info(f"After SMOTE, y_res distribution:\n{pd.Series(y_res).value_counts()}")
    return X_res, y_res

# ===================================
# 7. Model Training
# ===================================

def train_random_forest(X, y):
    """
    Trains a Random Forest classifier.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X, y)
    logging.info("Random Forest model trained successfully.")
    return model

# ===================================
# 8. Model Evaluation
# ===================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints classification report and ROC AUC.

    Parameters:
        model (sklearn estimator): Trained model.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Move', 'Move'], yticklabels=['No Move', 'Move'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# ===================================
# 9. Backtesting the Strategy
# ===================================

def backtest_strategy_multiple(df, model, feature_cols, initial_capital=100000):
    """
    Backtests multiple trading strategies based on model predictions.

    Parameters:
        df (pd.DataFrame): Testing DataFrame with features.
        model (sklearn estimator): Trained model.
        feature_cols (list): List of feature column names.
        initial_capital (float): Starting capital for backtesting.

    Returns:
        pd.DataFrame: DataFrame with backtest results for multiple strategies.
    """
    df = df.copy()
    df['Prediction'] = model.predict(df[feature_cols])
    df['Strategy_Return'] = df['Prediction'].shift(1) * df['SPY_Close'].pct_change()
    df['Strategy_Return'].fillna(0, inplace=True)

    # Buy and Hold Strategy
    df['Cumulative_Return'] = (1 + df['SPY_Close'].pct_change()).cumprod() * initial_capital

    # Model-based Strategy
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() * initial_capital

    # Additional Strategy: ROC-based
    if 'ROC_12_US_Close' in df.columns:
        df['ROC_Signal'] = np.where(df['ROC_12_US_Close'] > 0, 1, -1)
        df['ROC_Return'] = df['ROC_Signal'].shift(1) * df['SPY_Close'].pct_change()
        df['ROC_Return'].fillna(0, inplace=True)
        df['Cumulative_ROC_Strategy'] = (1 + df['ROC_Return']).cumprod() * initial_capital
    else:
        logging.warning("ROC indicator for 'US_Close' not found. Skipping ROC-based strategy.")
        df['Cumulative_ROC_Strategy'] = np.nan

    # Additional Strategy: CCI-based (example)
    if 'CCI_US_Close' in df.columns:
        df['CCI_Signal'] = np.where(df['CCI_US_Close'] > 100, -1, 
                                    np.where(df['CCI_US_Close'] < -100, 1, 0))
        df['CCI_Signal'] = df['CCI_Signal'].shift(1).fillna(0)
        df['CCI_Return'] = df['CCI_Signal'] * df['SPY_Close'].pct_change()
        df['CCI_Return'].fillna(0, inplace=True)
        df['Cumulative_CCI_Strategy'] = (1 + df['CCI_Return']).cumprod() * initial_capital
    else:
        logging.warning("CCI indicator for 'US_Close' not found. Skipping CCI-based strategy.")
        df['Cumulative_CCI_Strategy'] = np.nan

    # Compile results
    results = pd.DataFrame({
        'Date': df.index,
        'Buy_and_Hold': df['Cumulative_Return'],
        'Model_Strategy': df['Cumulative_Strategy_Return'],
        'ROC_Strategy': df['Cumulative_ROC_Strategy'],
        'CCI_Strategy': df['Cumulative_CCI_Strategy']
    })

    return results

# ===================================
# 10. Visualization of Backtest Results
# ===================================

def plot_backtest_results(results):
    """
    Plots the backtest results for multiple strategies.

    Parameters:
        results (pd.DataFrame): DataFrame with backtest results.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(results['Date'], results['Buy_and_Hold'], label='Buy and Hold', linewidth=2)
    plt.plot(results['Date'], results['Model_Strategy'], label='Model-Based Strategy', linewidth=2)
    if 'ROC_Strategy' in results.columns and not results['ROC_Strategy'].isna().all():
        plt.plot(results['Date'], results['ROC_Strategy'], label='Price Rate of Change Strategy', linewidth=2)
    if 'CCI_Strategy' in results.columns and not results['CCI_Strategy'].isna().all():
        plt.plot(results['Date'], results['CCI_Strategy'], label='CCI Strategy', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Backtest Results for Multiple Trading Strategies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Additional Visualization: Feature Correlation Heatmap
    # Note: Requires passing the engineered DataFrame with features
    # This can be incorporated separately if needed

# ===================================
# 11. Define Top 50 Stock Tickers for India and Japan
# ===================================

def get_top_tickers():
    """
    Returns dictionaries of top 50 stock tickers for India and Japan.
    Note: Replace the placeholder tickers with actual top 50 tickers from reliable sources.
    """
    india_tickers = [
        # Top 50 Indian Stocks (Replace with actual tickers)
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 'SBIN.NS', 'BAJFINANCE.NS',
        'AXISBANK.NS', 'ASIANPAINT.NS', 'HCLTECH.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS',
        'ITC.NS', 'SUNPHARMA.NS', 'WIPRO.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
        'NESTLEIND.NS', 'POWERGRID.NS', 'BHARTIARTL.NS', 'GRASIM.NS', 'ADANIPORTS.NS',
        'TATASTEEL.NS', 'JSWSTEEL.NS', 'M&M.NS', 'TECHM.NS', 'CIPLA.NS',
        'INDUSINDBK.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'HDFCLIFE.NS', 'SBILIFE.NS',
        'HDFCAMC.NS', 'TATAMOTORS.NS', 'NTPC.NS', 'BAJAJFINSV.NS', 'EICHERMOT.NS',
        'ONGC.NS', 'TATAPOWER.NS', 'VEDL.NS', 'GAIL.NS', 'BPCL.NS',
        'IOC.NS', 'HEROMOTOCO.NS', 'SHREECEM.NS'
        # Add more tickers up to 50 if needed
    ]

    japan_tickers = [
        # Top 50 Japanese Stocks (Replace with actual tickers)
        '7203.T', '6758.T', '9432.T', '9984.T', '6861.T',
        '7751.T', '8035.T', '6954.T', '7974.T', '8316.T',
        '8306.T', '6367.T', '4502.T', '8411.T', '4503.T',
        '6869.T', '9020.T', '8031.T', '7201.T', '6501.T',
        '6902.T', '6903.T', '6503.T', '7267.T', '4901.T',
        '4689.T', '6594.T', '6592.T', '4704.T', '6752.T',
        '6862.T', '6366.T', '6952.T', '7269.T', '9433.T',
        '4755.T', '4543.T', '4684.T', '4902.T', '6591.T',
        '4904.T', '6301.T', '6363.T', '4506.T', '7733.T',
        '8001.T', '6702.T', '8604.T', '4686.T', '2801.T'
        # Add more tickers up to 50 if needed
    ]

    tickers_dict = {
        'India': india_tickers,
        'Japan': japan_tickers,
        'US': ['SPY']  # Assuming 'SPY' is used as a benchmark
    }

    return tickers_dict

# ===================================
# 12. Main Execution Flow
# ===================================
def main():
    """
    Main function to execute the trading model pipeline.
    """
    # Define parameters
    start_date = '2015-01-01'
    end_date = '2024-09-20'

    # Get top tickers
    tickers_dict = get_top_tickers()

    # Fetch market data
    combined_data = fetch_market_data_all(tickers_dict, start_date, end_date)

    if combined_data.empty:
        logging.error("No data fetched. Exiting the program.")
        return

    logging.info("Starting feature engineering...")
    # Feature Engineering
    engineered_data = feature_engineering_extended(combined_data, tickers_dict)

    if engineered_data.empty:
        logging.error("No data left after feature engineering. Exiting the program.")
        return

    logging.info("Defining target variable...")
    # Define Target
    target_data = define_target(engineered_data, target_column='SPY_Close', threshold=0.001)

    if target_data.empty:
        logging.error("No data left after defining target variable. Exiting the program.")
        return

    logging.info("Splitting data into training and testing sets...")
    # Split Data
    train, test = split_data(target_data, test_size=0.2)

    logging.info("Preparing features and targets...")
    # Prepare Features and Targets
    X_train, y_train, X_test, y_test = prepare_features_targets(train, test, target_column='Target')

    if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
        logging.error("No valid training or testing data. Exiting the program.")
        return

    logging.info("Balancing classes using SMOTE...")
    # Handle Class Imbalance
    try:
        X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    except ValueError as e:
        logging.error(f"Error during class balancing: {e}")
        return

    logging.info("Training Random Forest model...")
    # Train Model
    model = train_random_forest(X_train_balanced, y_train_balanced)

    logging.info("Evaluating model performance...")
    # Evaluate Model
    evaluate_model(model, X_test, y_test)

    logging.info("Backtesting trading strategies...")
    # Backtest Strategy
    backtest_results = backtest_strategy_multiple(test, model, feature_cols=X_test.columns.tolist())

    logging.info("Plotting backtest results...")
    # Plot Backtest Results
    plot_backtest_results(backtest_results)

    logging.info("Trading model pipeline executed successfully.")

if __name__ == "__main__":
    freeze_support()  # For Windows support
    main()
