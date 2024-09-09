from flask import Flask, render_template
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Define time periods
time_periods = {
    '1wk': 7,
    '1m': 30,
    '3mo': 90,
    '6mo': 180,
    '1yr': 365
}

# List of Nifty 500 tickers
nifty500_symbols = ['360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS','AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'ARE&M.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 'AXISBANK.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS','CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FDC.NS', 'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 'IRFC.NS', 'INDIGOPNTS.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 'KPITTECH.NS', 'KRBL.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KOTAKBANK.NS', 'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'MTARTECH.NS', 'LODHA.NS', 'MGL.NS', 'MAHSEAMLES.NS', 'M&MFIN.NS', 'M&M.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MSUMI.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 'NAZARA.NS', 'NFL.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NILKAMAL.NS', 'NOCIL.NS', 'NOVOCO.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OLECTRA.NS', 'ONE97.NS', 'ORIENTELEC.NS', 'PAYTM.NS', 'PNB.NS', 'PFS.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PVR.NS', 'PATELENG.NS', 'PIRAMALENT.NS', 'PRECISION.NS', 'PRISMJOINTS.NS', 'PROCTER.NS', 'PRISM.NS', 'PRIMO.NS', 'PROFINS.NS', 'RBLBANK.NS', 'RBL.NS', 'RECLTD.NS', 'RELINFRA.NS', 'ROHL.NS', 'RANBAXY.NS', 'RIL.NS', 'RPG.NS', 'RUPA.NS', 'RELIANCE.NS', 'RTNINDIA.NS', 'SEQUENT.NS', 'SFL.NS', 'SILVER.NS', 'SOMANYCERA.NS', 'SHAREINDIA.NS', 'SANDHAR.NS', 'SUNDARAMFAST.NS', 'SUNDARAMFIN.NS', 'SUSHIL.NS', 'SUNPHARMA.NS', 'SYNDICATE.NS', 'SYNGENE.NS', 'SYNPHARM.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATAMETALIK.NS', 'TATACHEM.NS', 'TATASTEEL.NS', 'TCNSBRANDS.NS', 'TECHM.NS', 'TEJASNET.NS', 'TEN.NS', 'TERASOFT.NS', 'THYROCARE.NS', 'TITAN.NS', 'TOUCHWOOD.NS', 'TROYTECH.NS', 'TRITON.NS', 'UCOBANK.NS', 'UPL.NS', 'UJJIVAN.NS', 'UNIPOS.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS', 'VBL.NS', 'VIRINCHI.NS', 'VIPIND.NS', 'VST.NS', 'VSTIND.NS', 'VBL.NS', 'VTL.NS', 'V2RETAIL.NS', 'VIGIL.NS', 'VSTLTD.NS', 'VIRINCHI.NS', 'VIRINCHI.NS', 'YESBANK.NS', 'YUMBRANDS.NS', 'ZEE.NS', 'ZOMATO.NS']

# Fetch industry information
def get_stock_industry(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        industry = info.get('industry')
        if industry is None:
            industry = info.get('industryDisp', 'Unknown')
        
        return industry
    except Exception as e:
        print(f"Error fetching industry for {symbol}: {e}")
        return 'Unknown'
def generate_table_html(industry_performance):
    html = '<table>'
    html += '<thead><tr><th>Industry</th>'
    for period_name in time_periods.keys():
        html += f'<th>{period_name}</th>'
    html += '</tr></thead><tbody>'
    
    for industry, performance in industry_performance.items():
        html += f'<tr><td>{industry}</td>'
        for period_name in time_periods.keys():
            change = performance.get(period_name, None)
            if change is not None:
                html += f'<td>{change:.2f}%</td>'
            else:
                html += '<td>Data Not Available</td>'
        html += '</tr>'
    
    html += '</tbody></table>'
    return html

# Fetch stock data
def fetch_stock_data(symbol):
    try:
        stock_data = yf.download(symbol, start="2023-01-01", end="2024-09-02", progress=False)
        if not stock_data.empty:
            return stock_data['Adj Close'].dropna()  # Remove any NaN values
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    return None

# Calculate percentage change for a given period
def calculate_percentage_change(data, days):
    if len(data) >= days:
        return ((data.iloc[-1] - data.iloc[-days]) / data.iloc[-days]) * 100
    return None

# Main function
def stock_screener(symbols):
    industry_data = {}
    
    for symbol in symbols:
        industry = get_stock_industry(symbol)
        stock_data = fetch_stock_data(symbol)
        
        if stock_data is not None:
            if industry not in industry_data:
                industry_data[industry] = []

            industry_data[industry].append(stock_data)

    # Calculate average performance by industry
    industry_performance = {}
    for industry, stocks in industry_data.items():
        industry_performance[industry] = {}
        for period_name, days in time_periods.items():
            changes = [calculate_percentage_change(stock, days) for stock in stocks]
            changes = [change for change in changes if change is not None and -100 < change < 100]  # Filter extreme changes
            if changes:
                industry_performance[industry][period_name] = sum(changes) / len(changes)
            else:
                industry_performance[industry][period_name] = None
    
    return industry_performance

@app.route('/')
def index():
    industry_performance = stock_screener(nifty500_symbols)
    print("Industry Performance Data:", industry_performance)  # Debug print
    table_html = generate_table_html(industry_performance)
    return render_template('inds.html', table_html=table_html)

# Test fetching stock data
print(fetch_stock_data('RELIANCE.NS'))

# Test industry retrieval
print(get_stock_industry('RELIANCE.NS'))

# Test percentage change calculation
test_data = pd.Series([100, 105, 110, 115])
print(calculate_percentage_change(test_data, 3))

if __name__ == '__main__':
    app.run(port=5008)

