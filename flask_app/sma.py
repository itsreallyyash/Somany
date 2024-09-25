import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Nifty 500 Stock Screener", layout="wide")

# Title
st.title("Nifty 500 Stock Screener with 10-Year Win Rate Analysis")

# Sidebar for user inputs
st.sidebar.header("Screening Parameters")

filter_param = st.sidebar.selectbox("Filter By", options=["Volume", "Close Price"], index=0)
threshold = st.sidebar.number_input(f"Threshold for {filter_param}", value=0.0)
short_window = st.sidebar.number_input("Short Moving Average Window (days)", min_value=1, max_value=50, value=9)
long_window = st.sidebar.number_input("Long Moving Average Window (days)", min_value=1, max_value=200, value=21)
vcp_threshold = st.sidebar.number_input("VCP Contraction Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
signal_filter = st.sidebar.selectbox("Signal Filter", options=["All", "Buy", "Sell"], index=0)
summary_type = st.sidebar.selectbox("Summary Type", options=["Detailed", "Summary"], index=1)
max_symbols = st.sidebar.number_input("Max Symbols to Screen (for performance)", min_value=1, max_value=500, value=100)

# Define Nifty 500 symbols
nifty500_symbols = [
    '360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS','AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 
    'AFFLE.NS', 'AJANTPHARM.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'ARE&M.NS', 
    'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 
    'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 
    'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 'AXISBANK.NS', 
    'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 
    'BALAMINES.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 
    'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 
    'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 
    'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 
    'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS','CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 
    'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 
    'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 
    'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 
    'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 
    'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 
    'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 
    'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 
    'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 
    'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 
    'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 
    'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 
    'FDC.NS', 'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 
    'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 'GMMPFAUDLR.NS', 
    'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 
    'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 
    'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 
    'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 
    'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 
    'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 
    'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 
    'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 
    'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 
    'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDIACEM.NS', 
    'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 
    'IRFC.NS', 'INDIGOPNTS.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 
    'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 
    'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS', 
    'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 
    'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 
    'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 
    'KPITTECH.NS', 'KRBL.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 
    'KANSAINER.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KOTAKBANK.NS', 
    'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 
    'LAURUSLABS.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 
    'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'MTARTECH.NS', 'LODHA.NS', 'MGL.NS', 'MAHSEAMLES.NS', 
    'M&MFIN.NS', 'M&M.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MANKIND.NS', 
    'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 
    'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MSUMI.NS', 'MOTILALOFS.NS', 
    'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 
    'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 
    'NAZARA.NS', 'NFL.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NILKAMAL.NS', 'NOCIL.NS', 
    'NOVOCO.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OLECTRA.NS', 
    'ONE97.NS', 'ORIENTELEC.NS', 'PAYTM.NS', 'PNB.NS', 'PFS.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 
    'PIDILITIND.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PVR.NS', 'PATELENG.NS', 'PIRAMALENT.NS', 
    'PRECISION.NS', 'PRISMJOINTS.NS', 'PROCTER.NS', 'PRISM.NS', 'PRIMO.NS', 'PROFINS.NS', 
    'RBLBANK.NS', 'RBL.NS', 'RECLTD.NS', 'RELINFRA.NS', 'ROHL.NS', 'RANBAXY.NS', 'RIL.NS', 
    'RPG.NS', 'RUPA.NS', 'RELIANCE.NS', 'RTNINDIA.NS', 'SEQUENT.NS', 'SFL.NS', 'SILVER.NS', 
    'SOMANYCERA.NS', 'SHAREINDIA.NS', 'SANDHAR.NS', 'SUNDARAMFAST.NS', 'SUNDARAMFIN.NS', 
    'SUSHIL.NS', 'SUNPHARMA.NS', 'SYNDICATE.NS', 'SYNGENE.NS', 'SYNPHARM.NS', 'TATACONSUM.NS', 
    'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATAMETALIK.NS', 'TATACHEM.NS', 'TATASTEEL.NS', 
    'TCNSBRANDS.NS', 'TECHM.NS', 'TEJASNET.NS', 'TEN.NS', 'TERASOFT.NS', 'THYROCARE.NS', 
    'TITAN.NS', 'TOUCHWOOD.NS', 'TROYTECH.NS', 'TRITON.NS', 'UCOBANK.NS', 'UPL.NS', 
    'UJJIVAN.NS', 'UNIPOS.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS', 'VBL.NS', 
    'VIRINCHI.NS', 'VIPIND.NS', 'VST.NS', 'VSTIND.NS', 'VBL.NS', 'VTL.NS', 'V2RETAIL.NS', 
    'VIGIL.NS', 'VSTLTD.NS', 'VIRINCHI.NS', 'VIRINCHI.NS', 'YESBANK.NS', 'YUMBRANDS.NS', 
    'ZEE.NS', 'ZOMATO.NS'
]

@st.cache_data(ttl=3600)
def fetch_data(symbol, period="10y", interval="1d"):
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

def detect_vcp_pattern(df, contraction_threshold=0.1):
    """
    Detect Volatility Contraction Pattern (VCP) in the stock data.
    """
    if len(df) < 4:
        return False

    df['High_to_Low'] = df['High'] - df['Low']
    df['Previous_High_to_Low'] = df['High'].shift(1) - df['Low'].shift(1)
    df['Contraction'] = df['High_to_Low'] / df['Previous_High_to_Low']

    contractions = df['Contraction'].dropna().values[-3:]
    if len(contractions) < 3:
        return False
    if all(c < contraction_threshold for c in contractions):
        return True

    return False

def analyze_trades(df, short_window, long_window):
    """
    Simulate buy/sell trades over the historical data and calculate win rate.
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

        if entry['Trade'] == 'Buy':
            profit = (exit['Close'] - entry['Close']) / entry['Close']
        else:  # 'Sell' Trade
            profit = (entry['Close'] - exit['Close']) / entry['Close']

        win = 1 if profit > 0 else 0
        trades.append({'Trade': entry['Trade'], 'Profit': profit, 'Win': win})

    return pd.DataFrame(trades)

def detect_buy_sell_signals(recent_data):
    """
    Generate current buy/sell signals based on recent data.
    """
    if recent_data.empty:
        return None

    latest_data = recent_data.iloc[-1]
    cross_percentage = (latest_data['SMA_short'] - latest_data['SMA_long']) / latest_data['SMA_long'] * 100
    signal = 'Buy' if cross_percentage > 0 else 'Sell'
    return signal, cross_percentage

def screen_stocks(symbols, filter_param='Volume', threshold=0, short_window=9, long_window=21, 
                 vcp_threshold=0.1, signal_filter='All', summary_type='Detailed', max_symbols=100):
    """
    Screen stocks based on SMA crossover and VCP pattern, calculate win rate.
    """
    data = {}
    summary_data = []

    for idx, symbol in enumerate(symbols):
        if idx >= max_symbols:
            break
        try:
            # 1. Fetch 10 years of historical data
            historical_data = fetch_data(symbol, period="10y", interval="1d")
            if historical_data.empty:
                st.warning(f"No historical data for {symbol}. Skipping.")
                continue

            # 2. Backtest SMA crossover strategy on historical data
            historical_trades = analyze_trades(historical_data, short_window, long_window)
            if historical_trades.empty:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                risk_reward = np.nan
            else:
                win_rate = historical_trades['Win'].mean() * 100
                avg_win = historical_trades[historical_trades['Win'] == 1]['Profit'].mean()
                avg_loss = historical_trades[historical_trades['Win'] == 0]['Profit'].mean()
                risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

            # 3. Fetch 2-week recent data for current signals
            recent_data = fetch_data(symbol, period="2wk", interval="1d")
            if recent_data.empty:
                st.warning(f"No recent data for {symbol}. Skipping.")
                continue

            # 4. Calculate moving averages and detect VCP pattern on recent data
            recent_data = calculate_moving_averages(recent_data, short_window, long_window)
            recent_data = calculate_relative_volatility(recent_data)
            has_vcp = detect_vcp_pattern(recent_data, vcp_threshold)

            # 5. Generate current buy/sell signal
            current_signal, cross_percentage = detect_buy_sell_signals(recent_data)
            if current_signal is None:
                st.warning(f"Unable to generate signal for {symbol}. Skipping.")
                continue

            # 6. Prepare data for display
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
                    'VCP Pattern': 'Yes' if has_vcp else 'No',
                    'Win Rate (%)': f"{win_rate:.2f}",
                    'Avg Win (%)': f"{avg_win*100:.2f}" if not np.isnan(avg_win) else '-',
                    'Avg Loss (%)': f"{avg_loss*100:.2f}" if not np.isnan(avg_loss) else '-',
                    'Risk-Reward': f"{risk_reward:.2f}" if not np.isnan(risk_reward) else '-'
                }

                # Apply filters
                if filter_param == 'Volume' and latest_data['Volume'] < threshold:
                    continue
                if filter_param == 'Close Price' and latest_data['Close'] < threshold:
                    continue
                if signal_filter != 'All' and signal_display != signal_filter:
                    continue

                data[symbol] = latest_summary

            else:  # Summary
                summary_row = {
                    'Symbol': symbol,
                    'Buy': '✔️' if signal_display == 'Buy' else '',
                    'Sell': '✔️' if signal_display == 'Sell' else '',
                    'VCP Pattern': '✔️' if has_vcp else '',
                    'Win Rate (%)': f"{win_rate:.2f}",
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

            # To avoid hitting API rate limits
            time.sleep(0.5)

        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")

    if summary_type == 'Detailed':
        df = pd.DataFrame.from_dict(data, orient='index')
        return df
    else:
        summary_df = pd.DataFrame(summary_data)
        return summary_df

# Main Execution
if st.button("Run Screening"):
    with st.spinner("Screening stocks... This may take a while..."):
        screened_df = screen_stocks(
            nifty500_symbols,
            filter_param=filter_param,
            threshold=threshold,
            short_window=short_window,
            long_window=long_window,
            vcp_threshold=vcp_threshold,
            signal_filter=signal_filter,
            summary_type=summary_type,
            max_symbols=int(max_symbols)
        )

    if not screened_df.empty:
        if summary_type == 'Detailed':
            st.subheader("Detailed Results")

            # Highlight Cross Percentage
            def highlight_cross(x):
                try:
                    value = float(x.strip('%'))
                    color = 'green' if value > 0 else 'red'
                    return f'color: {color}'
                except:
                    return 'color: black'

            styled_df = screened_df.style.applymap(highlight_cross, subset=['% Crossed By'])
            st.dataframe(styled_df, use_container_width=True)
            st.download_button("Download CSV", screened_df.to_csv(index=False).encode('utf-8'), "screened_stocks.csv", mime='text/csv')
        else:
            st.subheader("Summary Results")
            st.dataframe(screened_df, use_container_width=True)
            st.download_button("Download CSV", screened_df.to_csv(index=False).encode('utf-8'), "summary_screened_stocks.csv", mime='text/csv')
    else:
        st.warning("No stocks matched the screening criteria.")

# Footer
st.markdown("---")
st.markdown("Developed by [Yash](https://yourwebsite.com)")
