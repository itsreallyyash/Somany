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
max_symbols = st.sidebar.number_input("Max Symbols to Screen (for performance)", min_value=1, max_value=500, value=50)  # Adjust as needed

# Define Nifty 500 symbols (Sample Nifty 50 for demonstration; expand as needed)
nifty500_symbols = ['360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS','AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'ARE&M.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 'AXISBANK.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS','CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FDC.NS', 'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 'IRFC.NS', 'INDIGOPNTS.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 'KPITTECH.NS', 'KRBL.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KOTAKBANK.NS', 'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'MTARTECH.NS', 'LODHA.NS', 'MGL.NS', 'MAHSEAMLES.NS', 'M&MFIN.NS', 'M&M.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MSUMI.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 'NAZARA.NS', 'NFL.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NILKAMAL.NS', 'NOCIL.NS', 'NOVOCO.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OLECTRA.NS', 'ONE97.NS', 'ORIENTELEC.NS', 'PAYTM.NS', 'PNB.NS', 'PFS.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PVR.NS', 'PATELENG.NS', 'PIRAMALENT.NS', 'PRECISION.NS', 'PRISMJOINTS.NS', 'PROCTER.NS', 'PRISM.NS', 'PRIMO.NS', 'PROFINS.NS', 'RBLBANK.NS', 'RBL.NS', 'RECLTD.NS', 'RELINFRA.NS', 'ROHL.NS', 'RANBAXY.NS', 'RIL.NS', 'RPG.NS', 'RUPA.NS', 'RELIANCE.NS', 'RTNINDIA.NS', 'SEQUENT.NS', 'SFL.NS', 'SILVER.NS', 'SOMANYCERA.NS', 'SHAREINDIA.NS', 'SANDHAR.NS', 'SUNDARAMFAST.NS', 'SUNDARAMFIN.NS', 'SUSHIL.NS', 'SUNPHARMA.NS', 'SYNDICATE.NS', 'SYNGENE.NS', 'SYNPHARM.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATAMETALIK.NS', 'TATACHEM.NS', 'TATASTEEL.NS', 'TCNSBRANDS.NS', 'TECHM.NS', 'TEJASNET.NS', 'TEN.NS', 'TERASOFT.NS', 'THYROCARE.NS', 'TITAN.NS', 'TOUCHWOOD.NS', 'TROYTECH.NS', 'TRITON.NS', 'UCOBANK.NS', 'UPL.NS', 'UJJIVAN.NS', 'UNIPOS.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS', 'VBL.NS', 'VIRINCHI.NS', 'VIPIND.NS', 'VST.NS', 'VSTIND.NS', 'VBL.NS', 'VTL.NS', 'V2RETAIL.NS', 'VIGIL.NS', 'VSTLTD.NS', 'VIRINCHI.NS', 'VIRINCHI.NS', 'YESBANK.NS', 'YUMBRANDS.NS', 'ZEE.NS', 'ZOMATO.NS']


# Remove duplicates just in case
nifty500_symbols = list(dict.fromkeys(nifty500_symbols))

@st.cache_data(ttl=3600)
def fetch_data(symbol, period="3mo", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            st.warning(f"No data found for {symbol}.")
            return pd.DataFrame()
        
        # Data Cleaning and Formatting
        data.reset_index(inplace=True)  # Reset index to access 'Date' as a column
        data.drop_duplicates(inplace=True)  # Remove duplicate entries
        
        # Ensure dates are sorted
        data.sort_values('Date', inplace=True)
        
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
    Generate current buy/sell signals based on recent data.
    """
    if recent_data.empty:
        return None, None

    latest_data = recent_data.iloc[-1]
    if pd.isna(latest_data['SMA_short']) or pd.isna(latest_data['SMA_long']):
        return None, None

    cross_percentage = (latest_data['SMA_short'] - latest_data['SMA_long']) / latest_data['SMA_long'] * 100
    signal = 'Buy' if cross_percentage > 0 else 'Sell'
    return signal, cross_percentage

def analyze_recent_data(recent_data, short_window, long_window):
    """
    Analyze recent data (past 14 days) for crossover statistics and identify all crossovers.
    """
    recent_data = calculate_moving_averages(recent_data, short_window, long_window)
    recent_data['Signal'] = np.where(recent_data['SMA_short'] > recent_data['SMA_long'], 'Buy', 'Sell')
    recent_data['Signal_Shifted'] = recent_data['Signal'].shift(1)
    recent_data['Cross'] = np.where(recent_data['Signal'] != recent_data['Signal_Shifted'], recent_data['Signal'], np.nan)

    # Define "recent" as within the past 14 days from the latest date in the dataset
    latest_date = recent_data['Date'].max()
    two_weeks_ago = latest_date - timedelta(days=14)
    recent_two_weeks = recent_data[recent_data['Date'] >= two_weeks_ago].copy()  # Create a copy to avoid SettingWithCopyWarning

    # Get all crossovers in the past 14 days
    crossovers = recent_two_weeks[recent_two_weeks['Cross'].notnull()]

    # Collect crossovers with their dates and signals
    if not crossovers.empty:
        crossover_list = []
        for _, row in crossovers.iterrows():
            crossover_list.append({
                'Date': row['Date'].strftime('%Y-%m-%d'),
                'Signal': row['Cross']
            })
    else:
        crossover_list = []

    # Count up crosses and down crosses
    up_crosses = (crossovers['Cross'] == 'Buy').sum()
    down_crosses = (crossovers['Cross'] == 'Sell').sum()

    # Identify uptrends and downtrends based on consecutive signals
    recent_two_weeks['Trend'] = recent_two_weeks['Signal'].ne(recent_two_weeks['Signal'].shift()).cumsum()
    trend_groups = recent_two_weeks.groupby('Trend')['Signal'].first()

    up_trends = (trend_groups == 'Buy').sum()
    down_trends = (trend_groups == 'Sell').sum()

    # Check if there was at least one crossover in the past 14 days
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
                 signal_filter='All', summary_type='Detailed', max_symbols=50):
    """
    Screen stocks based on SMA crossover and calculate trade statistics.
    Returns two DataFrames: Trade Statistics and Recent Crossover Analysis.
    """
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
            # 1. Fetch historical data
            historical_data = fetch_data(symbol, period="3mo", interval="1d")
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

            # 3. Fetch recent data (past 3 months to ensure enough data points)
            recent_data = fetch_data(symbol, period="3mo", interval="1d")
            if recent_data.empty:
                st.warning(f"No recent data for {symbol}. Skipping.")
                continue

            # 4. Calculate moving averages and relative volatility on recent data
            recent_data = calculate_moving_averages(recent_data, short_window, long_window)
            recent_data = calculate_relative_volatility(recent_data)

            # 5. Generate current buy/sell signal based on the previous day's data
            current_signal, cross_percentage = detect_buy_sell_signals(recent_data)

            # 6. Analyze recent data for additional metrics
            recent_analysis = analyze_recent_data(recent_data, short_window, long_window)

            # **Important:** Only include stocks with at least one crossover in the past two weeks
            if not recent_analysis['Has Crossover']:
                continue  # Skip stocks with no crossovers in the past two weeks

            # 7. Prepare Trade Statistics for display
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
            crossovers = recent_analysis['Crossovers']

            for crossover in crossovers:
                # Assign the signal to the next trading day
                crossover_date_obj = datetime.strptime(crossover['Date'], '%Y-%m-%d')
                signal_effective_date = crossover_date_obj + timedelta(days=1)
                # Adjust for weekends
                while signal_effective_date.weekday() > 4:  # 5=Saturday, 6=Sunday
                    signal_effective_date += timedelta(days=1)
                signal_effective_date_str = signal_effective_date.strftime('%Y-%m-%d')

                recent_summary = {
                    'Symbol': symbol,
                    'Crossover': crossover['Signal'],
                    'Crossover Date': crossover['Date'],
                    'Signal Effective Date': signal_effective_date_str
                }
                recent_data_summary.append(recent_summary)

            # To avoid hitting API rate limits
            time.sleep(0.1)  # Adjust as necessary

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
# Main Execution
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
            max_symbols=int(max_symbols)
        )

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

# Footer
st.markdown("---")
st.markdown("Developed by [Yash](https://www.linkedin.com/in/yashshahh/)")
