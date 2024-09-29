import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Nifty 500 Stock Screener", layout="wide")

# Title
st.title("Nifty 500 Stock Screener with Enhanced Analysis")

# Sidebar for user inputs
st.sidebar.header("Screening Parameters")

filter_param = st.sidebar.selectbox("Filter By", options=["Volume", "Close Price"], index=0)
threshold = st.sidebar.number_input(f"Threshold for {filter_param}", value=0.0)
short_window = st.sidebar.number_input("Short Moving Average Window (days)", min_value=1, max_value=50, value=9)
long_window = st.sidebar.number_input("Long Moving Average Window (days)", min_value=1, max_value=200, value=21)
signal_filter = st.sidebar.selectbox("Signal Filter", options=["All", "Buy", "Sell"], index=0)
summary_type = st.sidebar.selectbox("Summary Type", options=["Detailed", "Summary"], index=1)
max_symbols = st.sidebar.number_input("Max Symbols to Screen (for performance)", min_value=1, max_value=500, value=100)

# Define Nifty 500 symbols (Add more symbols as needed)

nifty500_symbols = ['360ONE.NS', '3MINDIA.NS']



@st.cache_data(ttl=3600)
def fetch_data(symbol, period="max", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            st.warning(f"No data found for {symbol}.")
        return data
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_moving_averages(df, short_window=9, long_window=21):
    """
    Calculate short-term and long-term Simple Moving Averages (SMA).
    """
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    return df

def calculate_relative_volatility(df):
    """
    Calculate daily returns and rolling volatility.
    """
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=21).std()
    return df

def analyze_trades(df, short_window, long_window):
    """
    Simulate buy/sell trades over the historical data and calculate trade statistics.
    Returns the trades DataFrame.
    """
    df = calculate_moving_averages(df, short_window, long_window)
    df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 'Buy', 'Sell')
    df['Signal_Shifted'] = df['Signal'].shift(1)
    df['Trade'] = np.where(df['Signal'] != df['Signal_Shifted'], df['Signal'], np.nan)
    df.dropna(subset=['Trade'], inplace=True)

    trades = []

    for i in range(len(df) - 1):
        entry = df.iloc[i]
        exit = df.iloc[i + 1]

        entry_date = entry.name
        exit_date = exit.name
        holding_period = (exit_date - entry_date).days

        if entry['Trade'] == 'Buy':
            profit = (exit['Close'] - entry['Close']) / entry['Close']
        else:  # 'Sell' Trade
            profit = (entry['Close'] - exit['Close']) / entry['Close']

        win = 1 if profit > 0 else 0
        trades.append({
            'Trade': entry['Trade'],
            'Profit': profit,
            'Win': win,
            'Holding_Period': holding_period
        })

    trades_df = pd.DataFrame(trades)
    return trades_df

def detect_buy_sell_signals(recent_data):
    """
    Generate current buy/sell signals based on recent data.
    """
    if recent_data.empty:
        return None, None

    latest_data = recent_data.iloc[-1]
    cross_percentage = (latest_data['SMA_short'] - latest_data['SMA_long']) / latest_data['SMA_long'] * 100
    signal = 'Buy' if cross_percentage > 0 else 'Sell'
    return signal, cross_percentage

def analyze_recent_data(recent_data, short_window, long_window):
    """
    Analyze recent data (past 2 weeks) for crossover statistics and identify the last crossover.
    """
    recent_data = calculate_moving_averages(recent_data, short_window, long_window)
    recent_data['Signal'] = np.where(recent_data['SMA_short'] > recent_data['SMA_long'], 'Buy', 'Sell')
    recent_data['Signal_Shifted'] = recent_data['Signal'].shift(1)
    recent_data['Cross'] = np.where(recent_data['Signal'] != recent_data['Signal_Shifted'], recent_data['Signal'], np.nan)
    recent_data.dropna(subset=['Cross'], inplace=True)

    # Filter data for the past 14 days
    today = recent_data.index[-1].normalize()
    two_weeks_ago = today - timedelta(days=14)
    recent_two_weeks = recent_data[recent_data.index >= two_weeks_ago]

    # Find the last crossover in the past two weeks
    if not recent_two_weeks.empty:
        last_crossover = recent_two_weeks.iloc[-1]
        last_crossover_signal = last_crossover['Cross']
        last_crossover_date = last_crossover.name.date()
    else:
        last_crossover_signal = 'N/A'
        last_crossover_date = 'N/A'

    # Count up crosses and down crosses
    up_crosses = (recent_two_weeks['Cross'] == 'Buy').sum()
    down_crosses = (recent_two_weeks['Cross'] == 'Sell').sum()

    # Identify uptrends and downtrends based on consecutive signals
    recent_two_weeks['Trend'] = recent_two_weeks['Signal'].ne(recent_two_weeks['Signal'].shift()).cumsum()
    trend_groups = recent_two_weeks.groupby('Trend')['Signal'].first()

    up_trends = (trend_groups == 'Buy').sum()
    down_trends = (trend_groups == 'Sell').sum()

    # Check if there was at least one crossover in the past two weeks
    has_crossover = (up_crosses + down_crosses) > 0

    return {
        'Up Crosses': up_crosses,
        'Down Crosses': down_crosses,
        'Up Trends': up_trends,
        'Down Trends': down_trends,
        'Last Crossover Signal': last_crossover_signal,
        'Last Crossover Date': last_crossover_date,
        'Has Crossover': has_crossover
    }

def screen_stocks(symbols, filter_param='Volume', threshold=0, short_window=9, long_window=21, 
                  signal_filter='All', summary_type='Detailed', max_symbols=100):
    """
    Screen stocks based on SMA crossover and calculate trade statistics.
    Returns two DataFrames: Trade Statistics and Recent Crossover Analysis.
    """
    trade_data = {}
    summary_data = []
    recent_data_summary = []

    for idx, symbol in enumerate(symbols):
        if idx >= max_symbols:
            break
        try:
            # 1. Fetch historical data (Max period)
            historical_data = fetch_data(symbol, period="max", interval="1d")
            if historical_data.empty:
                st.warning(f"No historical data for {symbol}. Skipping.")
                continue

            # 2. Backtest SMA crossover strategy on historical data
            historical_trades = analyze_trades(historical_data, short_window, long_window)
            if historical_trades.empty:
                win_rate = 0
                num_wins = 0
                num_losses = 0
                avg_win = 0
                avg_loss = 0
                avg_holding_period = 0
                avg_holding_period_win = 0
                avg_holding_period_loss = 0
                risk_reward = np.nan
            else:
                win_rate = historical_trades['Win'].mean() * 100
                num_wins = historical_trades['Win'].sum()
                num_losses = len(historical_trades) - num_wins
                avg_win = historical_trades[historical_trades['Win'] == 1]['Profit'].mean()
                avg_loss = historical_trades[historical_trades['Win'] == 0]['Profit'].mean()
                avg_holding_period = historical_trades['Holding_Period'].mean()
                avg_holding_period_win = historical_trades[historical_trades['Win'] == 1]['Holding_Period'].mean()
                avg_holding_period_loss = historical_trades[historical_trades['Win'] == 0]['Holding_Period'].mean()
                risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

            # 3. Fetch recent data (past month)
            recent_data = fetch_data(symbol, period="1mo", interval="1d")
            if recent_data.empty:
                st.warning(f"No recent data for {symbol}. Skipping.")
                continue

            # 4. Calculate moving averages and relative volatility on recent data
            recent_data = calculate_moving_averages(recent_data, short_window, long_window)
            recent_data = calculate_relative_volatility(recent_data)

            # 5. Generate current buy/sell signal
            current_signal, cross_percentage = detect_buy_sell_signals(recent_data)
            if current_signal is None:
                st.warning(f"Unable to generate signal for {symbol}. Skipping.")
                continue

            # 6. Analyze recent data for additional metrics
            recent_analysis = analyze_recent_data(recent_data, short_window, long_window)

            # **Important:** Only include stocks with at least one crossover in the past two weeks
            if not recent_analysis['Has Crossover']:
                continue  # Skip stocks with no crossovers in the past two weeks

            # 7. Prepare Trade Statistics for display
            latest_data = recent_data.iloc[-1]
            signal_display = 'Buy' if current_signal == 'Buy' else 'Sell'

            if summary_type == 'Detailed':
                latest_summary = {
                    'Symbol': symbol,
                    'Date': latest_data.name.date(),
                    'Open': latest_data['Open'],
                    'Close': latest_data['Close'],
                    'Adj Close': latest_data['Adj Close'],
                    'Volume': latest_data['Volume'],
                    'SMA9': f"{latest_data['SMA_short']:.2f}",
                    'SMA21': f"{latest_data['SMA_long']:.2f}",
                    'Returns': f"{latest_data['Returns']*100:.2f}%",
                    'Volatility': f"{latest_data['Volatility']*100:.2f}%",
                    '% Crossed By': f"{cross_percentage:.2f}%",
                    'Signal': signal_display,
                    'Win Rate (%)': f"{win_rate:.2f}",
                    'Number of Wins': num_wins,
                    'Number of Losses': num_losses,
                    'Avg Win (%)': f"{avg_win*100:.2f}" if not np.isnan(avg_win) else '-',
                    'Avg Loss (%)': f"{avg_loss*100:.2f}" if not np.isnan(avg_loss) else '-',
                    'Avg Holding Period (days)': f"{avg_holding_period:.2f}" if not np.isnan(avg_holding_period) else '-',
                    'Avg Holding Period (Wins)': f"{avg_holding_period_win:.2f}" if not np.isnan(avg_holding_period_win) else '-',
                    'Avg Holding Period (Losses)': f"{avg_holding_period_loss:.2f}" if not np.isnan(avg_holding_period_loss) else '-',
                    'Risk-Reward': f"{risk_reward:.2f}" if not np.isnan(risk_reward) else '-'
                }

                # Apply filters
                if filter_param == 'Volume' and latest_data['Volume'] < threshold:
                    continue
                if filter_param == 'Close Price' and latest_data['Close'] < threshold:
                    continue
                if signal_filter != 'All' and signal_display != signal_filter:
                    continue

                trade_data[symbol] = latest_summary

            else:  # Summary
                summary_row = {
                    'Symbol': symbol,
                    'Buy': '✔️' if signal_display == 'Buy' else '',
                    'Sell': '✔️' if signal_display == 'Sell' else '',
                    'Win Rate (%)': f"{win_rate:.2f}",
                    'Number of Wins': num_wins,
                    'Number of Losses': num_losses,
                    'Avg Holding Period (Wins)': f"{avg_holding_period_win:.2f}" if not np.isnan(avg_holding_period_win) else '-',
                    'Avg Holding Period (Losses)': f"{avg_holding_period_loss:.2f}" if not np.isnan(avg_holding_period_loss) else '-',
                    'Risk-Reward': f"{risk_reward:.2f}" if not np.isnan(risk_reward) else '-'
                }

                # Apply filters
                if filter_param == 'Volume' and latest_data['Volume'] < threshold:
                    continue
                if filter_param == 'Close Price' and latest_data['Close'] < threshold:
                    continue
                if signal_filter != 'All' and signal_display != signal_filter:
                    continue

                summary_data.append(summary_row)

            # 8. Prepare Recent Crossover Analysis for display
            recent_summary = {
                'Symbol': symbol,
                'Last Crossover': recent_analysis['Last Crossover Signal'],
                'Crossover Date': recent_analysis['Last Crossover Date']
            }

            recent_data_summary.append(recent_summary)

            # To avoid hitting API rate limits
            time.sleep(0.5)

        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")

    # Create Trade Statistics DataFrame
    if summary_type == 'Detailed':
        trade_df = pd.DataFrame.from_dict(trade_data, orient='index')
    else:
        trade_df = pd.DataFrame(summary_data)

    # Create Recent Crossover Analysis DataFrame
    recent_df = pd.DataFrame(recent_data_summary)

    return trade_df, recent_df

# Main Execution
if st.button("Run Screening"):
    with st.spinner("Screening stocks... This may take a while..."):
        screened_trade_df, screened_recent_df = screen_stocks(
            nifty500_symbols,
            filter_param=filter_param,
            threshold=threshold,
            short_window=short_window,
            long_window=long_window,
            signal_filter=signal_filter,
            summary_type=summary_type,
            max_symbols=int(max_symbols)
        )

    if not screened_trade_df.empty and not screened_recent_df.empty:
        if summary_type == 'Detailed':
            st.subheader("Trade Statistics")
            st.dataframe(screened_trade_df, use_container_width=True)
            st.download_button(
                "Download Trade Statistics CSV",
                screened_trade_df.to_csv(index=False).encode('utf-8'),
                "trade_statistics_detailed.csv",
                mime='text/csv'
            )

            st.subheader("Recent Crossover Analysis (Past 2 Weeks)")
            # Highlight Last Crossover with colors
            def highlight_last_crossover(row):
                color = ''
                if row['Last Crossover'] == 'Buy':
                    color = 'background-color: green; color: white'
                elif row['Last Crossover'] == 'Sell':
                    color = 'background-color: red; color: white'
                return [color] * len(row)

            styled_recent_df = screened_recent_df.style.apply(highlight_last_crossover, axis=1)
            st.dataframe(styled_recent_df, use_container_width=True)
            st.download_button(
                "Download Recent Crossover Analysis CSV",
                screened_recent_df.to_csv(index=False).encode('utf-8'),
                "recent_crossover_analysis.csv",
                mime='text/csv'
            )

        else:
            st.subheader("Summary Trade Statistics")
            st.dataframe(screened_trade_df, use_container_width=True)
            st.download_button(
                "Download Summary Trade Statistics CSV",
                screened_trade_df.to_csv(index=False).encode('utf-8'),
                "summary_trade_statistics.csv",
                mime='text/csv'
            )

            st.subheader("Recent Crossover Analysis (Past 2 Weeks)")
            # Highlight Last Crossover with colors
            def highlight_last_crossover(row):
                color = ''
                if row['Last Crossover'] == 'Buy':
                    color = 'background-color: green; color: white'
                elif row['Last Crossover'] == 'Sell':
                    color = 'background-color: red; color: white'
                return [color] * len(row)

            styled_recent_df = screened_recent_df.style.apply(highlight_last_crossover, axis=1)
            st.dataframe(styled_recent_df, use_container_width=True)
            st.download_button(
                "Download Recent Crossover Analysis CSV",
                screened_recent_df.to_csv(index=False).encode('utf-8'),
                "recent_crossover_analysis.csv",
                mime='text/csv'
            )
    else:
        st.warning("No stocks matched the screening criteria.")

# Footer
st.markdown("---")
st.markdown("Developed by [Yash](https://www.linkedin.com/in/yashshahh/)")
