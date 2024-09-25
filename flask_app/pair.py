import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# --------------------------
# 1. Define Sector-wise Stock Tickers
# --------------------------
sector_tickers = {
    'Financial Services': ['AXISBANK.NS', 'BANDHANBNK.NS', 'CUB.NS', 'FEDERALBNK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
                           'IDFCFIRSTB.NS', 'INDUSINDBK.NS', 'KOTAKBANK.NS', 'RBLBANK.NS', 'AUBANK.NS',
                           'BANKBARODA.NS', 'HDFCAMC.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'LICHSGFIN.NS',
                           'SBICARD.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'BAJFINANCE.NS',
                           'BAJAJFINSV.NS', 'CHOLAFIN.NS', 'PFC.NS', 'RECLTD.NS', 'IEX.NS', 'MCX.NS',
                           'IDFC.NS', 'MUTHOOTFIN.NS', 'MANAPPURAM.NS', 'PIRAMAL.NS', 'POONAWALLA.NS',
                           'IIFL.NS', 'LTF.NS'],
    'Automobile and Auto Components': ['APOLLOTYRE.NS', 'ASHOKLEY.NS', 'BAJAJ-AUTO.NS', 'BALKRISIND.NS',
                                       'BHARATFORG.NS', 'BOSCHLTD.NS', 'EICHERMOT.NS', 'EXIDEIND.NS',
                                       'HEROMOTOCO.NS', 'MRF.NS', 'M&M.NS', 'MARUTI.NS', 'MOTHERSON.NS',
                                       'TVSMOTOR.NS', 'TATAMOTORS.NS'],
    'Fast Moving Consumer Goods (FMCG)': ['BALRAMCHIN.NS', 'BRITANNIA.NS', 'COLPAL.NS', 'DABUR.NS',
                                         'GODREJCP.NS', 'HINDUNILVR.NS', 'ITC.NS', 'MARICO.NS',
                                         'NESTLEIND.NS', 'PGHH.NS', 'RADICO.NS', 'TATACONSUM.NS',
                                         'UBL.NS', 'UNITDSPR.NS', 'VBL.NS'],
    'Healthcare': ['ABBOTINDIA.NS', 'ALKEM.NS', 'AUROPHARMA.NS', 'BIOCON.NS', 'CIPLA.NS',
                   'DIVISLAB.NS', 'DRREDDY.NS', 'GLAND.NS', 'GLENMARK.NS', 'GRANULES.NS',
                   'IPCALAB.NS', 'JBCHEPHARM.NS', 'LAURUSLABS.NS', 'LUPIN.NS', 'MANKIND.NS',
                   'NATCOPHARM.NS', 'SANOFI.NS', 'SUNPHARMA.NS', 'TORNTPHARM.NS', 'ZYDUSLIFE.NS'],
    'Information Technology': ['COFORGE.NS', 'HCLTECH.NS', 'INFY.NS', 'LTTS.NS', 'LTIM.NS',
                                'MPHASIS.NS', 'PERSISTENT.NS', 'TCS.NS', 'TECHM.NS', 'WIPRO.NS'],
    'Oil Gas & Consumable Fuels': ['ADANIENT.NS', 'AEGISLOG.NS', 'BPCL.NS', 'CASTROLIND.NS',
                                   'GAIL.NS', 'GUJGASLTD.NS', 'GSPL.NS', 'HINDPETRO.NS',
                                   'IOC.NS', 'IGL.NS', 'MGL.NS', 'ONGC.NS', 'OIL.NS',
                                   'PETRONET.NS', 'RELIANCE.NS'],
    'Consumer Durables': ['AMBER.NS', 'BATAINDIA.NS', 'BLUESTARCO.NS', 'CENTURYPLY.NS',
                          'CERA.NS', 'CROMPTON.NS', 'DIXON.NS', 'HAVELLS.NS',
                          'KAJARIACER.NS', 'KALYANKJIL.NS', 'RAJESHEXPO.NS',
                          'TITAN.NS', 'VGUARD.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS'],
    'Media Entertainment & Publication': ['DISHTV.NS', 'HATHWAY.NS', 'NAZARA.NS', 'NETWORK18.NS',
                                           'PVRINOX.NS', 'SAREGAMA.NS', 'SUNTV.NS',
                                           'TV18BRDCST.NS', 'TIPSINDLTD.NS', 'ZEEL.NS'],
    'Metals & Mining': ['ADANIENT.NS', 'HINDALCO.NS', 'HINDCOPPER.NS', 'HINDZINC.NS',
                        'JSWSTEEL.NS', 'JSL.NS', 'JINDALSTEL.NS', 'NMDC.NS',
                        'NATIONALUM.NS', 'SAIL.NS', 'TATASTEEL.NS', 'VEDL.NS'],
    'Realty': ['BRIGADE.NS', 'DLF.NS', 'GODREJPROP.NS', 'LODHA.NS', 'MAHLIFE.NS',
               'OBEROIRLTY.NS', 'PHOENIXLTD.NS', 'PRESTIGE.NS', 'SOBHA.NS',
               'SUNTECK.NS']
}

# --------------------------
# 2. Streamlit Dashboard Setup
# --------------------------
st.set_page_config(page_title="📈 Stock Market Analysis Dashboard", layout="wide")
st.title("📈 Stock Market Analysis Dashboard")

# Sidebar for user inputs
st.sidebar.header("🔧 User Inputs")

# Select sectors
selected_sectors = st.sidebar.multiselect(
    'Select Sectors',
    list(sector_tickers.keys()),
    list(sector_tickers.keys())  # Default selected
)

# Time period selection
today = datetime.today()
default_start = today - timedelta(days=31)
start_date = st.sidebar.date_input('Start Date', default_start)
end_date = st.sidebar.date_input('End Date', today)

if start_date > end_date:
    st.sidebar.error('Error: Start date must be before end date.')

# Low movement criteria
low_movement_threshold = st.sidebar.slider(
    'Low Movement Threshold (Average Weekly % Change)',
    min_value=0.0, max_value=10.0, value=0.5, step=0.1  # Reduced default threshold
)

# Correlation thresholds
entry_threshold = st.sidebar.slider(
    'Correlation Entry Threshold (e.g., -0.9)',
    min_value=-1.0, max_value=0.0, value=-0.99, step=0.05
)

exit_threshold = st.sidebar.slider(
    'Correlation Exit Threshold (e.g., -0.3)',
    min_value=-1.0, max_value=1.0, value=-0.95, step=0.05
)

# Minimum holding period to reduce overtrading
min_holding_days = st.sidebar.slider(
    'Minimum Holding Period (Days)',
    min_value=7, max_value=30, value=14, step=1
)

# --------------------------
# 3. Data Fetching Functions
# --------------------------

@st.cache_data(show_spinner=False)
def fetch_adj_close(tickers, start, end):
    """
    Fetches the adjusted close prices for given tickers between start and end dates.
    """
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

@st.cache_data(show_spinner=False)
def fetch_market_cap(ticker):
    """
    Fetches the market capitalization for a given ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', np.nan)
    except:
        market_cap = np.nan
    return market_cap

def get_market_cap_df(tickers):
    """
    Retrieves market capitalization for a list of tickers.
    """
    market_caps = {}
    for ticker in tickers:
        cap = fetch_market_cap(ticker)
        market_caps[ticker] = cap
    return pd.Series(market_caps, name='Market Cap')

# --------------------------
# 4. Sector Movement Calculation
# --------------------------

def calculate_sector_movement(sector, tickers, data):
    """
    Calculates the sector's weekly movement based on the average percentage change of its stocks.
    """
    # Resample to weekly frequency, taking the last available price in the week
    weekly_data = data.resample('W').last()
    # Calculate weekly returns
    weekly_returns = weekly_data.pct_change().dropna()
    # Average weekly change across all stocks in the sector
    avg_weekly_change = weekly_returns.mean(axis=1) * 100  # Convert to percentage
    return avg_weekly_change

# --------------------------
# 5. Stock Pair Correlation Analysis
# --------------------------
def find_negatively_correlated_pairs(tickers, data, cap_similarity=0.2):
    """
    Identifies stock pairs with inverse returns within the same week.
    
    Parameters:
    - tickers: List of stock tickers.
    - data: DataFrame of adjusted close prices.
    - cap_similarity: Allowed percentage difference in market cap (e.g., 0.2 for 20%).
    
    Returns:
    - DataFrame with columns: Week, Stock 1, Stock 2, Return 1 (%), Return 2 (%)
    """
    # Calculate weekly returns
    weekly_data = data.resample('W').last()  # Resample data to weekly frequency
    weekly_returns = weekly_data.pct_change().dropna()  # Calculate weekly returns
    
    # Fetch market cap data
    market_cap_series = get_market_cap_df(tickers)
    
    # Initialize list to store negatively correlated pairs
    neg_pairs = []
    
    # Iterate over each week
    for week, returns in weekly_returns.iterrows():
        # Identify pairs where one stock went up and the other went down
        for stock1, stock2 in combinations(tickers, 2):
            ret1 = returns.get(stock1, 0)
            ret2 = returns.get(stock2, 0)
            if np.isnan(ret1) or np.isnan(ret2):
                continue
            # Define a minimum return threshold to ensure significant moves
            min_return_threshold = 3.0  # in percentage
            if ((ret1 > (min_return_threshold/100) and ret2 < -(min_return_threshold/100)) or 
                (ret1 < -(min_return_threshold/100) and ret2 > (min_return_threshold/100))):
                # Check market cap similarity
                cap1 = market_cap_series.get(stock1, np.nan)
                cap2 = market_cap_series.get(stock2, np.nan)
                if pd.notna(cap1) and pd.notna(cap2):
                    lower_bound = cap1 * (1 - cap_similarity)
                    upper_bound = cap1 * (1 + cap_similarity)
                    if lower_bound <= cap2 <= upper_bound:
                        neg_pairs.append({
                            'Week': week,  # Add week here
                            'Stock 1': stock1,
                            'Stock 2': stock2,
                            'Return 1 (%)': ret1 * 100,
                            'Return 2 (%)': ret2 * 100
                        })
    
    # Convert neg_pairs list to DataFrame
    neg_pairs_df = pd.DataFrame(neg_pairs)
    
    if neg_pairs_df.empty:
        # If the DataFrame is empty, return an empty DataFrame
        return pd.DataFrame()

    # Debugging print statements to check DataFrame
    print(neg_pairs_df.columns)  # Check if 'Week' column exists
    print(neg_pairs_df.head())  # Check if the 'Week' data is populated
    
    return neg_pairs_df

    # Filtering the DataFrame based on the 'Week' column and 'low_movement_weeks'
  # Return an empty DataFrame if no valid pairs

# Example call for testing the function
# Assuming `low_movement_weeks` and `data` have been defined
# neg_pairs_filtered = find_negatively_correlated_pairs(tickers, data, cap_similarity=0.2)



# --------------------------
# 6. Hedging Strategy Backtesting
# --------------------------

def backtest_hedge(stock1, stock2, data, min_holding_days=14):
    """
    Simulates a hedging strategy by buying a call on one stock and a put on the other.
    
    Parameters:
    - stock1: Ticker of the first stock.
    - stock2: Ticker of the second stock.
    - data: DataFrame of adjusted close prices for both stocks.
    - min_holding_days: Minimum number of days to hold the trade to reduce overtrading.
    
    Returns:
    - List of hedge records with details.
    """
    hedge_records = []
    df = data[[stock1, stock2]].dropna()
    weekly_data = df.resample('W').last()
    weekly_returns = weekly_data.pct_change().dropna()
    
    # Calculate historical volatility (standard deviation of daily returns)
    daily_returns = df.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    # Calculate option premium as a function of volatility (e.g., 1% of price per unit volatility)
    base_premium = 0.5  # Fixed base premium, in percent (2% of stock price is reasonable)
    volatility_multiplier = 1.0  # Adjust based on the sensitivity to volatility
    annual_volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

# Calculate option premium as a percentage of the stock price
    option_premium_percent = 3.0

    
    # Initialize position state
    in_position = False
    entry_date = None
    exit_date = None
    entry_price1 = None
    entry_price2 = None
    call_premium = None
    put_premium = None
    theta = 0.1  # Daily time decay rate (10% per week as an example)
    
    for week, returns in weekly_returns.iterrows():
        ret1 = returns.get(stock1, 0)
        ret2 = returns.get(stock2, 0)
        
        if not in_position:
            # Enter hedge if one stock went up and the other went down with significant returns
            if ret1 > 0 and ret2 < 0:
                # Buy Call on stock1 and Buy Put on stock2
                entry_date = week
                entry_price1 = weekly_data.loc[week, stock1]
                entry_price2 = weekly_data.loc[week, stock2]
                call_premium = entry_price1 * (option_premium_percent / 100)
                put_premium = entry_price2 * (option_premium_percent / 100)
                in_position = True
            elif ret1 < 0 and ret2 > 0:
                # Buy Call on stock2 and Buy Put on stock1
                entry_date = week
                entry_price1 = weekly_data.loc[week, stock1]
                entry_price2 = weekly_data.loc[week, stock2]
                call_premium = entry_price2 * (option_premium_percent / 100)
                put_premium = entry_price1 * (option_premium_percent / 100)
                in_position = True
        else:
            # Calculate holding duration
            duration = (week - entry_date).days
            if duration >= min_holding_days:
                # Exit hedge at the end of the holding period
                exit_date = week
                exit_price1 = weekly_data.loc[exit_date, stock1]
                exit_price2 = weekly_data.loc[exit_date, stock2]
                
                # Determine which stock was bought as Call and which as Put
                if ret1 > 0 and ret2 < 0:
                    call_payoff = max(0, exit_price1 - entry_price1)
                    put_payoff = max(0, entry_price2 - exit_price2)
                else:
                    call_payoff = max(0, exit_price2 - entry_price2)
                    put_payoff = max(0, entry_price1 - exit_price1)
                
                # Calculate total premium paid
                total_premium = call_premium + put_premium
                # Calculate total time decay
                total_time_decay = theta * (duration / 7)  # Assuming theta is weekly decay
                # Adjust premium for time decay
                adjusted_premium = total_premium * (1 - total_time_decay)
                # Ensure premium doesn't go negative
                adjusted_premium = max(adjusted_premium, 0)
                # Calculate total payoff
                total_payoff = call_payoff + put_payoff
                # Calculate net profit
                net_profit = total_payoff - adjusted_premium
                # Calculate ROI
                roi_percent = (net_profit / adjusted_premium) * 100 if adjusted_premium != 0 else 0
                
                hedge_records.append({
                    'Stock 1': stock1,
                    'Stock 2': stock2,
                    'Entry Date': entry_date.date(),
                    'Exit Date': exit_date.date(),
                    'Duration (Days)': duration,
                    'ROI (%)': roi_percent
                })
                
                # Reset position
                in_position = False
                entry_date = None
                exit_date = None
                entry_price1 = None
                entry_price2 = None
                call_premium = None
                put_premium = None
    
    # If position is still open at the end
    if in_position:
        exit_date = weekly_data.index[-1]
        exit_price1 = weekly_data.loc[exit_date, stock1]
        exit_price2 = weekly_data.loc[exit_date, stock2]
        
        duration = (exit_date - entry_date).days
        if duration >= min_holding_days:
            if ret1 > 0 and ret2 < 0:
                call_payoff = max(0, exit_price1 - entry_price1)
                put_payoff = max(0, entry_price2 - exit_price2)
            else:
                call_payoff = max(0, exit_price2 - entry_price2)
                put_payoff = max(0, entry_price1 - exit_price1)
            
            total_premium = call_premium + put_premium
            total_time_decay = theta * (duration / 7)
            adjusted_premium = total_premium * (1 - total_time_decay)
            adjusted_premium = max(adjusted_premium, 0)
            total_payoff = call_payoff + put_payoff
            net_profit = total_payoff - adjusted_premium
            roi_percent = (net_profit / adjusted_premium) * 100 if adjusted_premium != 0 else 0
            
            hedge_records.append({
                'Stock 1': stock1,
                'Stock 2': stock2,
                'Entry Date': entry_date.date(),
                'Exit Date': exit_date.date(),
                'Duration (Days)': duration,
                'ROI (%)': roi_percent
            })
    
    return hedge_records

# --------------------------
# 7. Main Dashboard Logic
# --------------------------

def main():
    # Fetch data for all selected sectors
    all_selected_tickers = []
    
    for sector in selected_sectors:
        all_selected_tickers.extend(sector_tickers[sector])
    all_selected_tickers = list(set(all_selected_tickers))  # Remove duplicates

    with st.spinner('Fetching stock data...'):
        adj_close_data = fetch_adj_close(all_selected_tickers, start_date, end_date)

    # Remove tickers with all NaN data
    adj_close_data.dropna(axis=1, how='all', inplace=True)
    available_tickers = adj_close_data.columns.tolist()

    if not available_tickers:
        st.error("No data available for the selected tickers and date range.")
        return

    # Fetch market cap data
    with st.spinner('Fetching market capitalization data...'):
        market_cap_series = get_market_cap_df(available_tickers)

    # Initialize containers for results
    hedge_opportunities = []
    backtest_results = []

    # Analyze each selected sector
    for sector in selected_sectors:
        st.subheader(f"📊 Sector Analysis: {sector}")
        tickers = sector_tickers[sector]
        available_sector_tickers = [ticker for ticker in tickers if ticker in adj_close_data.columns]
        if not available_sector_tickers:
            st.warning(f"No data available for sector: {sector}")
            continue

        sector_data = adj_close_data[available_sector_tickers].dropna()
        if sector_data.empty:
            st.warning(f"No complete data for sector: {sector}")
            continue

        # Calculate sector movement per week
        sector_movement = calculate_sector_movement(sector, available_sector_tickers, sector_data)

        # Plot sector movement
        fig, ax = plt.subplots(figsize=(10, 4))
        sector_movement.plot(ax=ax)
        ax.axhline(y=low_movement_threshold, color='red', linestyle='--', label=f'±{low_movement_threshold}% Threshold')
        ax.axhline(y=-low_movement_threshold, color='red', linestyle='--')
        ax.set_title(f"Weekly Sector Movement - {sector}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Weekly % Change (%)")
        ax.legend()
        st.pyplot(fig)

        # Identify weeks with low movement
        low_movement_weeks = sector_movement[(sector_movement >= -low_movement_threshold) & 
                                            (sector_movement <= low_movement_threshold)].index

        st.markdown(f"**Number of weeks with low movement (±{low_movement_threshold}%):** {len(low_movement_weeks)}")

        if len(low_movement_weeks) == 0:
            st.info(f"No weeks with low movement found for sector: {sector}")
            continue

        # Find negatively correlated pairs in low movement weeks
        neg_pairs_df = find_negatively_correlated_pairs(
        available_sector_tickers,  # Only tickers from the current sector
        adj_close_data[available_sector_tickers],  # Data for the current sector
        cap_similarity=0.2  # 20% similarity in market cap
)

        # Filter pairs that occurred during low movement weeks
        if 'Week' in neg_pairs_df.columns:
            neg_pairs_filtered = neg_pairs_df[neg_pairs_df['Week'].isin(low_movement_weeks)]
        else:
            st.write("No 'Week' column found in the dataframe.")


        # Display detected pairs
        if not neg_pairs_filtered.empty:
            st.markdown(f"**Detected Negatively Correlated Pairs during Low Movement Weeks in {sector}:**")
            st.dataframe(neg_pairs_filtered[['Week', 'Stock 1', 'Stock 2', 'Return 1 (%)', 'Return 2 (%)']])

            # Add to hedging opportunities
            for _, row in neg_pairs_filtered.iterrows():
                # Determine which stock to buy call and which to buy put
                if row['Return 1 (%)'] > 0 and row['Return 2 (%)'] < 0:
                    buy_call = row['Stock 1']
                    buy_put = row['Stock 2']
                elif row['Return 1 (%)'] < 0 and row['Return 2 (%)'] > 0:
                    buy_call = row['Stock 2']
                    buy_put = row['Stock 1']
                else:
                    continue  # Skip if returns are not strictly inverse

                hedge_opportunities.append({
                    'Sector': sector,
                    'Week': row['Week'],
                    'Buy Call': buy_call,
                    'Buy Put': buy_put
                })
        else:
            st.markdown(f"**No negatively correlated pairs found during low movement weeks in {sector} based on the criteria.**")

    # Hedging Strategy Backtesting
    if hedge_opportunities:
        st.markdown("---")
        st.header("📈 Hedging Strategy Backtesting & ROI Analysis")
        with st.spinner('Running backtests...'):
            # Remove duplicate pairs for backtesting
            unique_pairs = pd.DataFrame(hedge_opportunities)[['Buy Call', 'Buy Put']].drop_duplicates()
            for _, row in unique_pairs.iterrows():
                buy_call = row['Buy Call']
                buy_put = row['Buy Put']
                pair_data = adj_close_data[[buy_call, buy_put]].dropna()
                if pair_data.empty:
                    continue
                hedge_records = backtest_hedge(buy_call, buy_put, pair_data, min_holding_days=min_holding_days)
                backtest_results.extend(hedge_records)

        if backtest_results:
            backtest_df = pd.DataFrame(backtest_results)
            # Format dates
            backtest_df['Entry Date'] = pd.to_datetime(backtest_df['Entry Date'])
            backtest_df['Exit Date'] = pd.to_datetime(backtest_df['Exit Date'])
            # Display backtest results
            st.subheader("🔹 Individual Hedging Opportunities")
            st.dataframe(backtest_df)

            # Calculate overall ROI
            total_roi = backtest_df['ROI (%)'].sum()
            avg_roi = backtest_df['ROI (%)'].mean()
            st.markdown(f"**Total ROI from Hedging Strategy:** {total_roi:.2f}%")
            st.markdown(f"**Average ROI per Trade:** {avg_roi:.2f}%")

            # ROI Distribution
            st.subheader("📊 ROI Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(backtest_df['ROI (%)'], bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of ROI from Hedging Trades")
            ax.set_xlabel("ROI (%)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            # Duration vs ROI
            st.subheader("⏳ Duration vs ROI")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.scatterplot(data=backtest_df, x='Duration (Days)', y='ROI (%)', hue='Stock 1', ax=ax)
            ax.set_title("Hedging Trade Duration vs ROI")
            ax.set_xlabel("Duration (Days)")
            ax.set_ylabel("ROI (%)")
            st.pyplot(fig)
        else:
            st.info("No hedging opportunities yielded any trades based on the backtesting criteria.")
    else:
        st.info("No hedging opportunities detected to backtest.")

    # --------------------------
    # 8. Dashboard Insights
    # --------------------------
    st.markdown("---")
    st.header("📋 Dashboard Insights")

    # Sector Performance Trends
    st.subheader("🔸 Sector Performance Trends")
    if selected_sectors:
        fig, ax = plt.subplots(figsize=(12, 6))
        for sector in selected_sectors:
            tickers = sector_tickers[sector]
            available_sector_tickers = [ticker for ticker in tickers if ticker in adj_close_data.columns]
            if not available_sector_tickers:
                continue
            sector_data = adj_close_data[available_sector_tickers].dropna()
            if sector_data.empty:
                continue
            sector_movement = calculate_sector_movement(sector, available_sector_tickers, sector_data)
            ax.plot(sector_movement.index, sector_movement.values, label=sector)
        ax.axhline(y=low_movement_threshold, color='red', linestyle='--', label=f'±{low_movement_threshold}% Threshold')
        ax.axhline(y=-low_movement_threshold, color='red', linestyle='--')
        ax.set_title("Average Weekly % Change Across Selected Sectors")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Weekly % Change (%)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No sector performance data available.")

    # Detected Stock Pairs with High Negative Correlations
    st.subheader("🔸 Detected Stock Pairs with High Negative Correlations")
    if hedge_opportunities:
        pairs_summary = pd.DataFrame(hedge_opportunities)
        st.dataframe(pairs_summary)
    else:
        st.write("No negatively correlated stock pairs detected based on the current criteria.")
    neg_pairs_df = find_negatively_correlated_pairs(tickers, adj_close_data, cap_similarity=0.2)

    if neg_pairs_df is not None and not neg_pairs_df.empty and 'Week' in neg_pairs_df.columns:
        # Proceed with filtering only if neg_pairs_df is valid and contains the 'Week' column
        neg_pairs_filtered = neg_pairs_df[neg_pairs_df['Week'].isin(low_movement_weeks)]
        st.write(neg_pairs_filtered)
    else:
        st.write("No valid pairs or 'Week' column missing.")
    # Timeline of Detected Opportunities
    if backtest_results:
        st.subheader("🔸 Timeline of Hedging Opportunities")
        timeline_df = pd.DataFrame(backtest_results)
        timeline_df['Entry Date'] = pd.to_datetime(timeline_df['Entry Date'])
        timeline_df['Exit Date'] = pd.to_datetime(timeline_df['Exit Date'])
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, row in timeline_df.iterrows():
            ax.plot([row['Entry Date'], row['Exit Date']], [idx, idx], marker='o')
        ax.set_title("Timeline of Hedging Opportunities")
        ax.set_xlabel("Date")
        ax.set_ylabel("Hedging Opportunities")
        st.pyplot(fig)

    # Overall ROI
    if backtest_results:
        st.subheader("🔸 Overall ROI for the Hedging Strategy")
        st.markdown(f"**Total ROI:** {total_roi:.2f}%")
        st.markdown(f"**Average ROI per Trade:** {avg_roi:.2f}%")
    else:
        st.write("No ROI data available.")
#gnn logic is equal false
#temporal analysis 

    # --------------------------
    # 9. Additional Features
    # --------------------------
    st.markdown("---")
    st.header("ℹ️ Additional Information")
    st.markdown("""
    - **Data Source:** [Yahoo Finance](https://finance.yahoo.com/)
    - **Note:** This dashboard is for educational purposes and should not be considered as financial advice. Always consult with a financial professional before making investment decisions.
    - **Disclaimer:** The hedging strategy assumes the ability to buy options on stocks, which may involve additional costs and risks.
    """)

# --------------------------
# 10. Run the App
# --------------------------
if __name__ == "__main__":
    main()

