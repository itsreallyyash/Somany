from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import time

app = Flask(__name__)
nifty500_symbols = ['360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS','AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'ARE&M.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 'AXISBANK.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS','CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FDC.NS', 'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 'IRFC.NS', 'INDIGOPNTS.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 'KPITTECH.NS', 'KRBL.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KOTAKBANK.NS', 'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'MTARTECH.NS', 'LODHA.NS', 'MGL.NS', 'MAHSEAMLES.NS', 'M&MFIN.NS', 'M&M.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MSUMI.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 'NAZARA.NS', 'NFL.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NILKAMAL.NS', 'NOCIL.NS', 'NOVOCO.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OLECTRA.NS', 'ONE97.NS', 'ORIENTELEC.NS', 'PAYTM.NS', 'PNB.NS', 'PFS.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PVR.NS', 'PATELENG.NS', 'PIRAMALENT.NS', 'PRECISION.NS', 'PRISMJOINTS.NS', 'PROCTER.NS', 'PRISM.NS', 'PRIMO.NS', 'PROFINS.NS', 'RBLBANK.NS', 'RBL.NS', 'RECLTD.NS', 'RELINFRA.NS', 'ROHL.NS', 'RANBAXY.NS', 'RIL.NS', 'RPG.NS', 'RUPA.NS', 'RELIANCE.NS', 'RTNINDIA.NS', 'SEQUENT.NS', 'SFL.NS', 'SILVER.NS', 'SOMANYCERA.NS', 'SHAREINDIA.NS', 'SANDHAR.NS', 'SUNDARAMFAST.NS', 'SUNDARAMFIN.NS', 'SUSHIL.NS', 'SUNPHARMA.NS', 'SYNDICATE.NS', 'SYNGENE.NS', 'SYNPHARM.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATAMETALIK.NS', 'TATACHEM.NS', 'TATASTEEL.NS', 'TCNSBRANDS.NS', 'TECHM.NS', 'TEJASNET.NS', 'TEN.NS', 'TERASOFT.NS', 'THYROCARE.NS', 'TITAN.NS', 'TOUCHWOOD.NS', 'TROYTECH.NS', 'TRITON.NS', 'UCOBANK.NS', 'UPL.NS', 'UJJIVAN.NS', 'UNIPOS.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS', 'VBL.NS', 'VIRINCHI.NS', 'VIPIND.NS', 'VST.NS', 'VSTIND.NS', 'VBL.NS', 'VTL.NS', 'V2RETAIL.NS', 'VIGIL.NS', 'VSTLTD.NS', 'VIRINCHI.NS', 'VIRINCHI.NS', 'YESBANK.NS', 'YUMBRANDS.NS', 'ZEE.NS', 'ZOMATO.NS']
def calculate_moving_averages(df, short_window=9, long_window=21):
    df['SMA9'] = df['Close'].rolling(window=short_window).mean()
    df['SMA21'] = df['Close'].rolling(window=long_window).mean()
    return df 

def calculate_relative_volatility(df):
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=21).std()
    return df

def detect_vcp_pattern(df, contraction_threshold=0.1):
    if len(df) < 3:
        return False

    df['High_to_Low'] = df['High'] - df['Low']
    df['Previous_High_to_Low'] = df['High'].shift(1) - df['Low'].shift(1)
    df['Contraction'] = df['High_to_Low'] / df['Previous_High_to_Low']

    contractions = df['Contraction'].dropna().values[-3:]
    if all(c < contraction_threshold for c in contractions):
        return True
    
    return False

def screen_stocks(symbols, filter_param='volume', threshold=0, short_window=9, long_window=21, vcp_threshold=0.1, signal_filter='', summary_type='detailed'):
    data = {}
    summary_data = []

    for symbol in symbols:
        try:
            stock_data = yf.download(symbol, period="1y", interval="1wk")
            if stock_data.empty:
                print(f"No data for {symbol}")
                continue
            
            stock_data = calculate_moving_averages(stock_data, short_window, long_window)

            
            latest_data = stock_data.iloc[-1]
            cross_percentage = (latest_data['SMA9'] - latest_data['SMA21']) / latest_data['SMA21'] * 100
            signal = 'Buy' if cross_percentage > 0 else 'Sell'
            
            if summary_type == 'detailed':
                latest_data = pd.Series({
                    'Symbol': symbol,
                    'Date': latest_data.name,
                    'Open': latest_data['Open'],
                    'Close': latest_data['Close'],
                    'Adj Close': latest_data['Adj Close'],
                    'Volume': latest_data['Volume'],
                    'SMA9': latest_data['SMA9'],
                    'SMA21': latest_data['SMA21'],
                    'Returns': latest_data['Returns'],
                    'Volatility': latest_data['Volatility'],
                    '% Crossed By': cross_percentage,
                    'Signal': signal
                })
                if filter_param == 'volume' and latest_data['Volume'] > threshold:
                    data[symbol] = latest_data
                elif filter_param == 'close' and latest_data['Close'] > threshold:
                    data[symbol] = latest_data
                
                if signal_filter and signal != signal_filter:
                    continue
            else:
                summary_row = {
                    'Symbol': symbol,
                    'Buy': '✔️' if signal == 'Buy' else '-',
                    'Sell': '✔️' if signal == 'Sell' else '-',
                    'Neutral': '-' if signal in ['Buy', 'Sell'] else '✔️'
                }
                summary_data.append(summary_row)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        
        time.sleep(1)  
    
    if summary_type == 'detailed':
        df = pd.DataFrame(data.values())
        return df
    else:
        summary_df = pd.DataFrame(summary_data)
        return summary_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        filter_param = request.form.get('filter_param', 'volume')
        threshold = float(request.form.get('threshold', 0))
        short_window = int(request.form.get('short_window', 9))
        long_window = int(request.form.get('long_window', 21))
        vcp_threshold = float(request.form.get('vcp_threshold', 0.1))
        signal_filter = request.form.get('signal_filter', '')
        summary_type = request.form.get('summary_type', 'detailed')

        combined_df = screen_stocks(nifty500_symbols, filter_param, threshold, short_window, long_window, vcp_threshold, signal_filter, summary_type)
        
        if summary_type == 'detailed' and not combined_df.empty:
            combined_df['% Crossed By'] = combined_df['% Crossed By'].apply(
                lambda x: f"<span style='color: {'green' if x > 0 else 'red'};'>{x:.2f}%</span>"
            )
        
        combined_df.to_csv('screened_stocks.csv', index=False)
        
        table_html = combined_df.to_html(classes='data', index=False, escape=False)
        
        return render_template('results.html', table_html=table_html)
    
    return render_template('index1.html')


if __name__ == '__main__':
    app.run(debug=True)
