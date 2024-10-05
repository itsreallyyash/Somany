import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Set Streamlit page configuration
st.set_page_config(page_title="Nifty 500 Stock Screener", layout="wide")

# Title
st.title("Nifty 500 Stock Screener with Enhanced Analysis")

# Sidebar for Screening Parameters
st.sidebar.header("Screening Parameters")

# Moving Average Type Selection
ma_type = st.sidebar.selectbox("Select Moving Average Type", options=["SMA", "EMA"], index=0)

# Other Screening Parameters
filter_param = st.sidebar.selectbox("Filter By", options=["Volume", "Close Price"], index=0)
threshold = st.sidebar.number_input(f"Threshold for {filter_param}", value=0.0)
short_window = st.sidebar.number_input(
    f"Short {'SMA' if ma_type == 'SMA' else 'EMA'} Window (days)", min_value=1, max_value=200, value=9
)
long_window = st.sidebar.number_input(
    f"Long {'SMA' if ma_type == 'SMA' else 'EMA'} Window (days)", min_value=1, max_value=200, value=21
)
signal_filter = st.sidebar.selectbox("Signal Filter", options=["All", "Buy", "Sell"], index=0)
summary_type = st.sidebar.selectbox("Summary Type", options=["Detailed", "Summary"], index=1)
max_symbols = st.sidebar.number_input(
    "Max Symbols to Screen (for performance)", min_value=1, max_value=500, value=50
)

# Defined Nifty 500 Symbols
nifty500_symbols = [
    'JSWSTEEL.ns', 'NTPC.ns', 'HINDALCO.ns', 'TATASTEEL.ns', 'BRITANNIA.ns', 
    'BPCL.ns', 'ASIANPAINT.ns', 'GRASIM.ns', 'TITAN.ns', 'ADANIENT.ns',
    'ONGC.ns', 'DRREDDY.ns', 'WIPRO.ns', 'HINDUNILVR.ns', 'TATACONSUM.ns', 
    'POWERGRID.ns', 'APOLLOHOSP.ns', 'ADANIPORTS.ns', 'HCLTECH.ns', 'BAJFINANCE.ns', 
    'EICHERMOT.ns', 'LT.ns', 'ITC.ns', 'TCS.ns', 'KOTAKBANK.ns', 
    'INDUSINDBK.ns', 'CIPLA.ns', 'SUNPHARMA.ns', 'COALINDIA.ns', 'HDFCBANK.ns', 
    'SHRIRAMFIN.ns', 'ULTRACEMCO.ns', 'BHARTIARTL.ns', 'INFY.ns', 'HDFCLIFE.ns', 
    'SBIN.ns', 'TATAMOTORS.ns', 'BAJAJFINSV.ns', 'MARUTI.ns', 'TECHM.ns', 
    'NESTLEIND.ns', 'SBILIFE.ns', 'BAJAJ-AUTO.ns', 'ICICIBANK.ns', 'M&M.ns', 
    'BEL.ns', 'AXISBANK.ns', 'RELIANCE.ns', 'TRENT.ns', 'HEROMOTOCO.ns'
]

# Remove duplicates
nifty500_symbols = list(dict.fromkeys(nifty500_symbols))

# Caching fetched data to improve performance
@st.cache_data(ttl=3600)
def fetch_data(symbol, period="5y", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            st.warning(f"No data found for {symbol}.")
            return pd.DataFrame()
        data.reset_index(inplace=True) 
        data.drop_duplicates(inplace=True) 
        data.sort_values('Date', inplace=True)
        return data
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_moving_averages(df, short_window=9, long_window=21, ma_type='SMA'):
    """
    Calculate moving averages (SMA or EMA) for the dataframe.
    """
    if ma_type == 'SMA':
        df['MA_short'] = df['Close'].rolling(window=short_window).mean()
        df['MA_long'] = df['Close'].rolling(window=long_window).mean()
    else:
        df['MA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['MA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    return df

def calculate_relative_volatility(df):
    """
    Calculate relative volatility based on returns.
    """
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=21).std()
    return df

def analyze_trades(df, short_window, long_window, ma_type='SMA'):
    """
    Analyze trades based on moving average crossovers.
    """
    df = calculate_moving_averages(df, short_window, long_window, ma_type)
    df['Signal'] = np.where(df['MA_short'] > df['MA_long'], 'Buy', 'Sell')
    df['Signal_Shifted'] = df['Signal'].shift(1)
    df['Trade'] = np.where(df['Signal'] != df['Signal_Shifted'], df['Signal'], np.nan)
    df.dropna(subset=['Trade'], inplace=True)

    trades = []

    for i in range(len(df) - 1):
        entry = df.iloc[i]
        exit = df.iloc[i + 1]

        entry_date = entry['Date']
        exit_date = exit['Date']
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
    Detect current buy/sell signals based on the latest moving averages.
    """
    if recent_data.empty:
        return None, None

    latest_data = recent_data.iloc[-1]
    if pd.isna(latest_data['MA_short']) or pd.isna(latest_data['MA_long']):
        return None, None

    cross_percentage = (latest_data['MA_short'] - latest_data['MA_long']) / latest_data['MA_long'] * 100
    signal = 'Buy' if cross_percentage > 0 else 'Sell'
    return signal, cross_percentage

def analyze_recent_data(recent_data, short_window, long_window, ma_type='SMA'):
    """
    Analyze recent data for crossovers and trends.
    """
    recent_data = calculate_moving_averages(recent_data, short_window, long_window, ma_type)
    recent_data['Signal'] = np.where(recent_data['MA_short'] > recent_data['MA_long'], 'Buy', 'Sell')
    recent_data['Signal_Shifted'] = recent_data['Signal'].shift(1)
    recent_data['Cross'] = np.where(recent_data['Signal'] != recent_data['Signal_Shifted'], recent_data['Signal'], np.nan)

    latest_date = recent_data['Date'].max()
    two_weeks_ago = latest_date - timedelta(days=14)
    recent_two_weeks = recent_data[recent_data['Date'] >= two_weeks_ago].copy()  

    crossovers = recent_two_weeks[recent_two_weeks['Cross'].notnull()]

    if not crossovers.empty:
        crossover_list = []
        for _, row in crossovers.iterrows():
            crossover_list.append({
                'Date': row['Date'].strftime('%Y-%m-%d'),
                'Signal': row['Cross']
            })
    else:
        crossover_list = []

    up_crosses = (crossovers['Cross'] == 'Buy').sum()
    down_crosses = (crossovers['Cross'] == 'Sell').sum()

    recent_two_weeks['Trend'] = recent_two_weeks['Signal'].ne(recent_two_weeks['Signal'].shift()).cumsum()
    trend_groups = recent_two_weeks.groupby('Trend')['Signal'].first()

    up_trends = (trend_groups == 'Buy').sum()
    down_trends = (trend_groups == 'Sell').sum()

    has_crossover = len(crossover_list) > 0

    return {
        'Up Crosses': up_crosses,
        'Down Crosses': down_crosses,
        'Up Trends': up_trends,
        'Down Trends': down_trends,
        'Crossovers': crossover_list,
        'Has Crossover': has_crossover
    }

def screen_stocks(symbols, filter_param='Volume', threshold=0, short_window=9, long_window=21, 
                 signal_filter='All', summary_type='Detailed', max_symbols=50, ma_type='SMA'):

    # Ensure symbols are unique
    symbols = list(dict.fromkeys(symbols))
    
    trade_data = {}
    summary_data = []
    recent_data_summary = []

    for idx, symbol in enumerate(symbols):
        if idx >= max_symbols:
            break
        try:
            st.write(f"Processing {symbol}...")
            
            # 1. Fetch historical data (Use max period for better analysis)
            historical_data = fetch_data(symbol, period="max", interval="1d")
            if historical_data.empty:
                st.warning(f"No historical data for {symbol}. Skipping.")
                continue

            # 2. Analyze trades using full historical data
            historical_trades = analyze_trades(historical_data, short_window, long_window, ma_type)
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

            # 3. Fetch recent data (past 3 months to ensure enough data points)
            recent_data = fetch_data(symbol, period="3mo", interval="1d")
            if recent_data.empty:
                st.warning(f"No recent data for {symbol}. Skipping.")
                continue

            # 4. Calculate moving averages and relative volatility on recent data
            recent_data = calculate_moving_averages(recent_data, short_window, long_window, ma_type)
            recent_data = calculate_relative_volatility(recent_data)

            # 5. Generate current buy/sell signal based on the latest data
            current_signal, cross_percentage = detect_buy_sell_signals(recent_data)

            # 6. Analyze recent data for additional metrics
            recent_analysis = analyze_recent_data(recent_data, short_window, long_window, ma_type)

            # **Recent Crossover Filtering** for **Recent Crossover Analysis** only
            crossovers = recent_analysis['Crossovers']
            for crossover in crossovers:
                crossover_date_obj = datetime.strptime(crossover['Date'], '%Y-%m-%d')
                signal_effective_date = crossover_date_obj + timedelta(days=1)
                while signal_effective_date.weekday() > 4:  # Skip weekends
                    signal_effective_date += timedelta(days=1)
                signal_effective_date_str = signal_effective_date.strftime('%Y-%m-%d')

                recent_summary = {
                    'Symbol': symbol,
                    'Crossover': crossover['Signal'],
                    'Crossover Date': crossover['Date'],
                    'Signal Effective Date': signal_effective_date_str
                }
                recent_data_summary.append(recent_summary)

            # 7. Prepare Trade Statistics for display (include all stocks)
            latest_data = recent_data.iloc[-1]
            signal_display = current_signal if current_signal else 'N/A'

            if summary_type == 'Detailed':
                latest_summary = {
                    'Symbol': symbol,
                    'Crossover Date': 'N/A',
                    'Signal Effective Date': 'N/A',
                    'Open': latest_data['Open'],
                    'Close': latest_data['Close'],
                    'Adj Close': latest_data['Adj Close'],
                    'Volume': latest_data['Volume'],
                    f'MA{short_window}': f"{latest_data['MA_short']:.2f}",
                    f'MA{long_window}': f"{latest_data['MA_long']:.2f}",
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

                # **Apply user filters here** (for all stocks)
                if filter_param == 'Volume' and latest_data['Volume'] < threshold:
                    continue
                if filter_param == 'Close Price' and latest_data['Close'] < threshold:
                    continue
                if signal_filter != 'All' and signal_display != signal_filter:
                    continue

                trade_data[symbol] = latest_summary

            else:  # Summary view (still includes all stocks)
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

                # **Apply user filters here** (for all stocks)
                if filter_param == 'Volume' and latest_data['Volume'] < threshold:
                    continue
                if filter_param == 'Close Price' and latest_data['Close'] < threshold:
                    continue
                if signal_filter != 'All' and signal_display != signal_filter:
                    continue

                summary_data.append(summary_row)

            # To avoid hitting API rate limits
            time.sleep(0.1)  # Adjust as necessary

        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")

    # Create Trade Statistics DataFrame (include all stocks)
    if summary_type == 'Detailed':
        trade_df = pd.DataFrame.from_dict(trade_data, orient='index')
    else:
        trade_df = pd.DataFrame(summary_data)

    # Create Recent Crossover Analysis DataFrame (still based only on crossover stocks)
    recent_df = pd.DataFrame(recent_data_summary)

    return trade_df, recent_df

def main():
    # Run Screening
    if st.button("Run Screening"):
        with st.spinner("Screening stocks... Please wait..."):
            trade_df, recent_df = screen_stocks(
                nifty500_symbols,
                filter_param=filter_param,
                threshold=threshold,
                short_window=short_window,
                long_window=long_window,
                signal_filter=signal_filter,
                summary_type=summary_type,
                max_symbols=int(max_symbols),
                ma_type=ma_type
            )

        # Display Screening Results
        if not trade_df.empty and not recent_df.empty:
            if summary_type == 'Detailed':
                st.subheader("Trade Statistics")
                st.dataframe(trade_df, use_container_width=True)
                st.download_button(
                    "Download Trade Statistics CSV",
                    trade_df.to_csv(index=False).encode('utf-8'),
                    "trade_statistics_detailed.csv",
                    mime='text/csv'
                )

                st.subheader("Recent Crossover Analysis (Past 2 Weeks)")
                # Highlight Crossover with colors
                def highlight_crossover(row):
                    colors = []
                    for col in row.index:
                        if col == 'Crossover':
                            if row[col] == 'Buy':
                                colors.append('background-color: green; color: white')
                            elif row[col] == 'Sell':
                                colors.append('background-color: red; color: white')
                            else:
                                colors.append('')
                        else:
                            colors.append('')
                    return colors

                styled_recent_df = recent_df.style.apply(highlight_crossover, axis=1)
                st.dataframe(styled_recent_df, use_container_width=True)
                st.download_button(
                    "Download Recent Crossover Analysis CSV",
                    recent_df.to_csv(index=False).encode('utf-8'),
                    "recent_crossover_analysis.csv",
                    mime='text/csv'
                )

            else:
                st.subheader("Summary Trade Statistics")
                st.dataframe(trade_df, use_container_width=True)
                st.download_button(
                    "Download Summary Trade Statistics CSV",
                    trade_df.to_csv(index=False).encode('utf-8'),
                    "summary_trade_statistics.csv",
                    mime='text/csv'
                )

                st.subheader("Recent Crossover Analysis (Past 2 Weeks)")
                # Highlight Crossover with colors
                def highlight_crossover(row):
                    colors = []
                    for col in row.index:
                        if col == 'Crossover':
                            if row[col] == 'Buy':
                                colors.append('background-color: green; color: white')
                            elif row[col] == 'Sell':
                                colors.append('background-color: red; color: white')
                            else:
                                colors.append('')
                        else:
                            colors.append('')
                    return colors

                styled_recent_df = recent_df.style.apply(highlight_crossover, axis=1)
                st.dataframe(styled_recent_df, use_container_width=True)
                st.download_button(
                    "Download Recent Crossover Analysis CSV",
                    recent_df.to_csv(index=False).encode('utf-8'),
                    "recent_crossover_analysis.csv",
                    mime='text/csv'
                )
        else:
            st.warning("No stocks matched the screening criteria.")

    st.markdown("---")
    st.markdown("Developed by [Yash](https://www.linkedin.com/in/yashshahh/)")

if __name__ == "__main__":
    main()
