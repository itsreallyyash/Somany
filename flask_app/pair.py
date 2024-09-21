# from flask import Flask, render_template, jsonify
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.stattools import coint
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# # Placeholder list of Nifty 500 tickers (replace with actual tickers)
# NIFTY_500_TICKERS = [
# '360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS','AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'ARE&M.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 'AXISBANK.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS','CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FDC.NS', 'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 'IRFC.NS', 'INDIGOPNTS.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 'KPITTECH.NS', 'KRBL.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KOTAKBANK.NS', 'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'MTARTECH.NS', 'LODHA.NS', 'MGL.NS', 'MAHSEAMLES.NS', 'M&MFIN.NS', 'M&M.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MSUMI.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 'NAZARA.NS', 'NFL.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NILKAMAL.NS', 'NOCIL.NS', 'NOVOCO.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OLECTRA.NS', 'ONE97.NS', 'ORIENTELEC.NS', 'PAYTM.NS', 'PNB.NS', 'PFS.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PVR.NS', 'PATELENG.NS', 'PIRAMALENT.NS', 'PRECISION.NS', 'PRISMJOINTS.NS', 'PROCTER.NS', 'PRISM.NS', 'PRIMO.NS', 'PROFINS.NS', 'RBLBANK.NS', 'RBL.NS', 'RECLTD.NS', 'RELINFRA.NS', 'ROHL.NS', 'RANBAXY.NS', 'RIL.NS', 'RPG.NS', 'RUPA.NS', 'RELIANCE.NS', 'RTNINDIA.NS', 'SEQUENT.NS', 'SFL.NS', 'SILVER.NS', 'SOMANYCERA.NS', 'SHAREINDIA.NS', 'SANDHAR.NS', 'SUNDARAMFAST.NS', 'SUNDARAMFIN.NS', 'SUSHIL.NS', 'SUNPHARMA.NS', 'SYNDICATE.NS', 'SYNGENE.NS', 'SYNPHARM.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATAMETALIK.NS', 'TATACHEM.NS', 'TATASTEEL.NS', 'TCNSBRANDS.NS', 'TECHM.NS', 'TEJASNET.NS', 'TEN.NS', 'TERASOFT.NS', 'THYROCARE.NS', 'TITAN.NS', 'TOUCHWOOD.NS', 'TROYTECH.NS', 'TRITON.NS', 'UCOBANK.NS', 'UPL.NS', 'UJJIVAN.NS', 'UNIPOS.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS', 'VBL.NS', 'VIRINCHI.NS', 'VIPIND.NS', 'VST.NS', 'VSTIND.NS', 'VBL.NS', 'VTL.NS', 'V2RETAIL.NS', 'VIGIL.NS', 'VSTLTD.NS', 'VIRINCHI.NS', 'VIRINCHI.NS', 'YESBANK.NS', 'YUMBRANDS.NS', 'ZEE.NS', 'ZOMATO.NS'
# ]
# # Function to get historical data
# def get_stock_data(tickers, start_date, end_date):
#     data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
#     return data

# # Function to identify pairs
# def identify_pairs(data):
#     tickers = data.columns
#     pairs = []
#     for i in range(len(tickers)):
#         for j in range(i+1, len(tickers)):
#             stock1 = tickers[i]
#             stock2 = tickers[j]
#             prices1 = data[stock1].dropna()
#             prices2 = data[stock2].dropna()
#             if len(prices1) > 0 and len(prices2) > 0:
#                 score, p_value, _ = coint(prices1, prices2)
#                 if p_value < 0.05:  # Example threshold for cointegration
#                     pairs.append((stock1, stock2, p_value))
#     return sorted(pairs, key=lambda x: x[2])  # Sort by p-value

# # Function to analyze sectors
# def analyze_sectors(pairs):
#     # Placeholder sector data (replace with actual sector data)
#     sectors = {
    # 'DISHTV': 'Media Entertainment & Publication',
    # 'HATHWAY': 'Media Entertainment & Publication',
    # 'NAZARA': 'Media Entertainment & Publication',
    # 'NETWORK18': 'Media Entertainment & Publication',
    # 'PVRINOX': 'Media Entertainment & Publication',
    # 'SAREGAMA': 'Media Entertainment & Publication',
    # 'SUNTV': 'Media Entertainment & Publication',
    # 'TV18BRDCST': 'Media Entertainment & Publication',
    # 'TIPSINDLTD': 'Media Entertainment & Publication',
    # 'ZEEL': 'Media Entertainment & Publication',
    
    # # Capital Goods
    # 'APLAPOLLO': 'Capital Goods',
    # 'RATNAMANI': 'Capital Goods',
    # 'WELCORP': 'Capital Goods',
    
    # # Metals & Mining
    # 'ADANIENT': 'Metals & Mining',
    # 'HINDALCO': 'Metals & Mining',
    # 'HINDCOPPER': 'Metals & Mining',
    # 'HINDZINC': 'Metals & Mining',
    # 'JSWSTEEL': 'Metals & Mining',
    # 'JSL': 'Metals & Mining',
    # 'JINDALSTEL': 'Metals & Mining',
    # 'NMDC': 'Metals & Mining',
    # 'NATIONALUM': 'Metals & Mining',
    # 'SAIL': 'Metals & Mining',
    # 'TATASTEEL': 'Metals & Mining',
    # 'VEDL': 'Metals & Mining',
    
    # # Financial Services (with additions)
    # 'AXISBANK': 'Financial Services',
    # 'BAJFINANCE': 'Financial Services',
    # 'BAJAJFINSV': 'Financial Services',
    # 'CHOLAFIN': 'Financial Services',
    # 'HDFCAMC': 'Financial Services',
    # 'HDFCBANK': 'Financial Services',
    # 'HDFCLIFE': 'Financial Services',
    # 'ICICIBANK': 'Financial Services',
    # 'ICICIGI': 'Financial Services',
    # 'ICICIPRULI': 'Financial Services',
    # 'IDFC': 'Financial Services',
    # 'KOTAKBANK': 'Financial Services',
    # 'LICHSGFIN': 'Financial Services',
    # 'MUTHOOTFIN': 'Financial Services',
    # 'PFC': 'Financial Services',
    # 'RECLTD': 'Financial Services',
    # 'SBICARD': 'Financial Services',
    # 'SBILIFE': 'Financial Services',
    # 'SHRIRAMFIN': 'Financial Services',
    # 'SBIN': 'Financial Services',
    # 'AUBANK': 'Financial Services',
    # 'ABCAPITAL': 'Financial Services',
    # 'BSE': 'Financial Services',
    # 'BANDHANBNK': 'Financial Services',
    # 'CANFINHOME': 'Financial Services',
    # 'CDSL': 'Financial Services',
    # 'CHOLAHLDNG': 'Financial Services',
    # 'CUB': 'Financial Services',
    # 'CAMS': 'Financial Services',
    # 'FEDERALBNK': 'Financial Services',
    # 'IDFCFIRSTB': 'Financial Services',
    # 'IIFL': 'Financial Services',
    # 'IEX': 'Financial Services',
    # 'LTF': 'Financial Services',
    # 'M&MFIN': 'Financial Services',
    # 'MANAPPURAM': 'Financial Services',
    # 'MFSL': 'Financial Services',
    # 'MCX': 'Financial Services',
    # 'PAYTM': 'Financial Services',
    # 'POLICYBZR': 'Financial Services',
    # 'PEL': 'Financial Services',
    # 'POONAWALLA': 'Financial Services',
    # 'RBLBANK': 'Financial Services',
    # 'SUNDARMFIN': 'Financial Services',
    # 'UNIONBANK': 'Financial Services',
    # 'YESBANK': 'Financial Services',
    # 'BANKBARODA': 'Financial Services',
    # 'BANKINDIA': 'Financial Services',
    # 'MAHABANK': 'Financial Services',
    # 'CANBK': 'Financial Services',
    # 'CENTRALBK': 'Financial Services',
    # 'INDIANB': 'Financial Services',
    # 'IOB': 'Financial Services',
    # 'PSB': 'Financial Services',
    # 'PNB': 'Financial Services',
    # 'UCOBANK': 'Financial Services',
    
    # # Fast Moving Consumer Goods
    # 'BALRAMCHIN': 'Fast Moving Consumer Goods',
    # 'BRITANNIA': 'Fast Moving Consumer Goods',
    # 'COLPAL': 'Fast Moving Consumer Goods',
    # 'DABUR': 'Fast Moving Consumer Goods',
    # 'GODREJCP': 'Fast Moving Consumer Goods',
    # 'HINDUNILVR': 'Fast Moving Consumer Goods',
    # 'ITC': 'Fast Moving Consumer Goods',
    # 'MARICO': 'Fast Moving Consumer Goods',
    # 'NESTLEIND': 'Fast Moving Consumer Goods',
    # 'PGHH': 'Fast Moving Consumer Goods',
    # 'RADICO': 'Fast Moving Consumer Goods',
    # 'TATACONSUM': 'Fast Moving Consumer Goods',
    # 'UBL': 'Fast Moving Consumer Goods',
    # 'UNITDSPR': 'Fast Moving Consumer Goods',
    # 'VBL': 'Fast Moving Consumer Goods',
    
    # # Healthcare (with additions)
    # 'ABBOTINDIA': 'Healthcare',
    # 'ALKEM': 'Healthcare',
    # 'APOLLOHOSP': 'Healthcare',
    # 'AUROPHARMA': 'Healthcare',
    # 'BIOCON': 'Healthcare',
    # 'CIPLA': 'Healthcare',
    # 'DIVISLAB': 'Healthcare',
    # 'LALPATHLAB': 'Healthcare',
    # 'DRREDDY': 'Healthcare',
    # 'GLENMARK': 'Healthcare',
    # 'GRANULES': 'Healthcare',
    # 'IPCALAB': 'Healthcare',
    # 'LAURUSLABS': 'Healthcare',
    # 'LUPIN': 'Healthcare',
    # 'MAXHEALTH': 'Healthcare',
    # 'METROPOLIS': 'Healthcare',
    # 'SUNPHARMA': 'Healthcare',
    # 'SYNGENE': 'Healthcare',
    # 'TORNTPHARM': 'Healthcare',
    # 'ZYDUSLIFE': 'Healthcare',
    # 'GLAND': 'Healthcare',
    # 'JBCHEPHARM': 'Healthcare',
    # 'MANKIND': 'Healthcare',
    # 'NATCOPHARM': 'Healthcare',
    # 'SANOFI': 'Healthcare',
    # 'DUMMYSANOF': 'Healthcare',
    
    # # Information Technology
    # 'COFORGE': 'Information Technology',
    # 'HCLTECH': 'Information Technology',
    # 'INFY': 'Information Technology',
    # 'LTTS': 'Information Technology',
    # 'LTIM': 'Information Technology',
    # 'MPHASIS': 'Information Technology',
    # 'PERSISTENT': 'Information Technology',
    # 'TCS': 'Information Technology',
    # 'TECHM': 'Information Technology',
    # 'WIPRO': 'Information Technology',

    # # Oil Gas & Consumable Fuels (with additions)
    # 'ATGL': 'Oil Gas & Consumable Fuels',
    # 'AEGISLOG': 'Oil Gas & Consumable Fuels',
    # 'BPCL': 'Oil Gas & Consumable Fuels',
    # 'CASTROLIND': 'Oil Gas & Consumable Fuels',
    # 'GAIL': 'Oil Gas & Consumable Fuels',
    # 'GUJGASLTD': 'Oil Gas & Consumable Fuels',
    # 'GSPL': 'Oil Gas & Consumable Fuels',
    # 'HINDPETRO': 'Oil Gas & Consumable Fuels',
    # 'IOC': 'Oil Gas & Consumable Fuels',
    # 'IGL': 'Oil Gas & Consumable Fuels',
    # 'MGL': 'Oil Gas & Consumable Fuels',
    # 'ONGC': 'Oil Gas & Consumable Fuels',
    # 'OIL': 'Oil Gas & Consumable Fuels',
    # 'PETRONET': 'Oil Gas & Consumable Fuels',
    # 'RELIANCE': 'Oil Gas & Consumable Fuels',
#     }

#     sector_pairs = {}
#     for stock1, stock2, p_value in pairs:
#         sector1 = sectors.get(stock1)
#         sector2 = sectors.get(stock2)
#         if sector1 and sector2 and sector1 == sector2:
#             if sector1 not in sector_pairs:
#                 sector_pairs[sector1] = []
#             sector_pairs[sector1].append((stock1, stock2, p_value))
    
#     # Filter out sectors with pairs where one stock is bullish and another is bearish
#     stagnant_sectors = {}
#     for sector, sector_pairs in sector_pairs.items():
#         bullish_bearish_pairs = [pair for pair in sector_pairs]
#         if bullish_bearish_pairs:
#             stagnant_sectors[sector] = bullish_bearish_pairs

#     return stagnant_sectors

# @app.route('/')
# def index():
#     return render_template('index3.html')

# @app.route('/fetch_pairs', methods=['GET'])
# def fetch_pairs():
#     start_date = '2023-01-01'  # Example start date
#     end_date = '2024-01-01'    # Example end date
    
#     data = get_stock_data(NIFTY_500_TICKERS, start_date, end_date)
#     pairs = identify_pairs(data)
#     stagnant_sectors = analyze_sectors(pairs)
    
#     return jsonify(stagnant_sectors)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

app = Flask(__name__)

# Placeholder list of Nifty 500 tickers (replace with actual tickers)
NIFTY_500_TICKERS = [
'360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS','AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'ARE&M.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 'AXISBANK.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS','CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FDC.NS', 'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 'IRFC.NS', 'INDIGOPNTS.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 'KPITTECH.NS', 'KRBL.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KOTAKBANK.NS', 'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'MTARTECH.NS', 'LODHA.NS', 'MGL.NS', 'MAHSEAMLES.NS', 'M&MFIN.NS', 'M&M.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MSUMI.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 'NAZARA.NS', 'NFL.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NILKAMAL.NS', 'NOCIL.NS', 'NOVOCO.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OLECTRA.NS', 'ONE97.NS', 'ORIENTELEC.NS', 'PAYTM.NS', 'PNB.NS', 'PFS.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PVR.NS', 'PATELENG.NS', 'PIRAMALENT.NS', 'PRECISION.NS', 'PRISMJOINTS.NS', 'PROCTER.NS', 'PRISM.NS', 'PRIMO.NS', 'PROFINS.NS', 'RBLBANK.NS', 'RBL.NS', 'RECLTD.NS', 'RELINFRA.NS', 'ROHL.NS', 'RANBAXY.NS', 'RIL.NS', 'RPG.NS', 'RUPA.NS', 'RELIANCE.NS', 'RTNINDIA.NS', 'SEQUENT.NS', 'SFL.NS', 'SILVER.NS', 'SOMANYCERA.NS', 'SHAREINDIA.NS', 'SANDHAR.NS', 'SUNDARAMFAST.NS', 'SUNDARAMFIN.NS', 'SUSHIL.NS', 'SUNPHARMA.NS', 'SYNDICATE.NS', 'SYNGENE.NS', 'SYNPHARM.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATAMETALIK.NS', 'TATACHEM.NS', 'TATASTEEL.NS', 'TCNSBRANDS.NS', 'TECHM.NS', 'TEJASNET.NS', 'TEN.NS', 'TERASOFT.NS', 'THYROCARE.NS', 'TITAN.NS', 'TOUCHWOOD.NS', 'TROYTECH.NS', 'TRITON.NS', 'UCOBANK.NS', 'UPL.NS', 'UJJIVAN.NS', 'UNIPOS.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS', 'VBL.NS', 'VIRINCHI.NS', 'VIPIND.NS', 'VST.NS', 'VSTIND.NS', 'VBL.NS', 'VTL.NS', 'V2RETAIL.NS', 'VIGIL.NS', 'VSTLTD.NS', 'VIRINCHI.NS', 'VIRINCHI.NS', 'YESBANK.NS', 'YUMBRANDS.NS', 'ZEE.NS', 'ZOMATO.NS'
]

def get_stock_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            df.name = ticker
            data = pd.concat([data, df], axis=1)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
    return data

def align_data(data):
    # Drop columns with all NaN values
    data = data.dropna(axis=1, how='all')
    # Align all series to the same dates
    data = data.dropna()
    return data

def identify_pairs(data):
    tickers = data.columns
    pairs = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            stock1 = tickers[i]
            stock2 = tickers[j]
            prices1 = data[stock1]
            prices2 = data[stock2]
            if len(prices1) > 0 and len(prices2) > 0:
                prices1, prices2 = prices1.align(prices2, join='inner')
                if len(prices1) > 0 and len(prices2) > 0:
                    try:
                        score, p_value, _ = coint(prices1, prices2)
                        if p_value < 0.05:  # Example threshold for cointegration
                            pairs.append((stock1, stock2, p_value))
                    except Exception as e:
                        print(f"Cointegration failed for {stock1} and {stock2}: {e}")
    return sorted(pairs, key=lambda x: x[2])  # Sort by p-value

def analyze_sectors(pairs):
    # Placeholder sector data (replace with actual sector data)
    sectors = {
    'DISHTV': 'Media Entertainment & Publication',
    'HATHWAY': 'Media Entertainment & Publication',
    'NAZARA': 'Media Entertainment & Publication',
    'NETWORK18': 'Media Entertainment & Publication',
    'PVRINOX': 'Media Entertainment & Publication',
    'SAREGAMA': 'Media Entertainment & Publication',
    'SUNTV': 'Media Entertainment & Publication',
    'TV18BRDCST': 'Media Entertainment & Publication',
    'TIPSINDLTD': 'Media Entertainment & Publication',
    'ZEEL': 'Media Entertainment & Publication',
    
    # Capital Goods
    'APLAPOLLO': 'Capital Goods',
    'RATNAMANI': 'Capital Goods',
    'WELCORP': 'Capital Goods',
    
    # Metals & Mining
    'ADANIENT': 'Metals & Mining',
    'HINDALCO': 'Metals & Mining',
    'HINDCOPPER': 'Metals & Mining',
    'HINDZINC': 'Metals & Mining',
    'JSWSTEEL': 'Metals & Mining',
    'JSL': 'Metals & Mining',
    'JINDALSTEL': 'Metals & Mining',
    'NMDC': 'Metals & Mining',
    'NATIONALUM': 'Metals & Mining',
    'SAIL': 'Metals & Mining',
    'TATASTEEL': 'Metals & Mining',
    'VEDL': 'Metals & Mining',
    
    # Financial Services (with additions)
    'AXISBANK': 'Financial Services',
    'BAJFINANCE': 'Financial Services',
    'BAJAJFINSV': 'Financial Services',
    'CHOLAFIN': 'Financial Services',
    'HDFCAMC': 'Financial Services',
    'HDFCBANK': 'Financial Services',
    'HDFCLIFE': 'Financial Services',
    'ICICIBANK': 'Financial Services',
    'ICICIGI': 'Financial Services',
    'ICICIPRULI': 'Financial Services',
    'IDFC': 'Financial Services',
    'KOTAKBANK': 'Financial Services',
    'LICHSGFIN': 'Financial Services',
    'MUTHOOTFIN': 'Financial Services',
    'PFC': 'Financial Services',
    'RECLTD': 'Financial Services',
    'SBICARD': 'Financial Services',
    'SBILIFE': 'Financial Services',
    'SHRIRAMFIN': 'Financial Services',
    'SBIN': 'Financial Services',
    'AUBANK': 'Financial Services',
    'ABCAPITAL': 'Financial Services',
    'BSE': 'Financial Services',
    'BANDHANBNK': 'Financial Services',
    'CANFINHOME': 'Financial Services',
    'CDSL': 'Financial Services',
    'CHOLAHLDNG': 'Financial Services',
    'CUB': 'Financial Services',
    'CAMS': 'Financial Services',
    'FEDERALBNK': 'Financial Services',
    'IDFCFIRSTB': 'Financial Services',
    'IIFL': 'Financial Services',
    'IEX': 'Financial Services',
    'LTF': 'Financial Services',
    'M&MFIN': 'Financial Services',
    'MANAPPURAM': 'Financial Services',
    'MFSL': 'Financial Services',
    'MCX': 'Financial Services',
    'PAYTM': 'Financial Services',
    'POLICYBZR': 'Financial Services',
    'PEL': 'Financial Services',
    'POONAWALLA': 'Financial Services',
    'RBLBANK': 'Financial Services',
    'SUNDARMFIN': 'Financial Services',
    'UNIONBANK': 'Financial Services',
    'YESBANK': 'Financial Services',
    'BANKBARODA': 'Financial Services',
    'BANKINDIA': 'Financial Services',
    'MAHABANK': 'Financial Services',
    'CANBK': 'Financial Services',
    'CENTRALBK': 'Financial Services',
    'INDIANB': 'Financial Services',
    'IOB': 'Financial Services',
    'PSB': 'Financial Services',
    'PNB': 'Financial Services',
    'UCOBANK': 'Financial Services',
    
    # Fast Moving Consumer Goods
    'BALRAMCHIN': 'Fast Moving Consumer Goods',
    'BRITANNIA': 'Fast Moving Consumer Goods',
    'COLPAL': 'Fast Moving Consumer Goods',
    'DABUR': 'Fast Moving Consumer Goods',
    'GODREJCP': 'Fast Moving Consumer Goods',
    'HINDUNILVR': 'Fast Moving Consumer Goods',
    'ITC': 'Fast Moving Consumer Goods',
    'MARICO': 'Fast Moving Consumer Goods',
    'NESTLEIND': 'Fast Moving Consumer Goods',
    'PGHH': 'Fast Moving Consumer Goods',
    'RADICO': 'Fast Moving Consumer Goods',
    'TATACONSUM': 'Fast Moving Consumer Goods',
    'UBL': 'Fast Moving Consumer Goods',
    'UNITDSPR': 'Fast Moving Consumer Goods',
    'VBL': 'Fast Moving Consumer Goods',
    
    # Healthcare (with additions)
    'ABBOTINDIA': 'Healthcare',
    'ALKEM': 'Healthcare',
    'APOLLOHOSP': 'Healthcare',
    'AUROPHARMA': 'Healthcare',
    'BIOCON': 'Healthcare',
    'CIPLA': 'Healthcare',
    'DIVISLAB': 'Healthcare',
    'LALPATHLAB': 'Healthcare',
    'DRREDDY': 'Healthcare',
    'GLENMARK': 'Healthcare',
    'GRANULES': 'Healthcare',
    'IPCALAB': 'Healthcare',
    'LAURUSLABS': 'Healthcare',
    'LUPIN': 'Healthcare',
    'MAXHEALTH': 'Healthcare',
    'METROPOLIS': 'Healthcare',
    'SUNPHARMA': 'Healthcare',
    'SYNGENE': 'Healthcare',
    'TORNTPHARM': 'Healthcare',
    'ZYDUSLIFE': 'Healthcare',
    'GLAND': 'Healthcare',
    'JBCHEPHARM': 'Healthcare',
    'MANKIND': 'Healthcare',
    'NATCOPHARM': 'Healthcare',
    'SANOFI': 'Healthcare',
    'DUMMYSANOF': 'Healthcare',
    
    # Information Technology
    'COFORGE': 'Information Technology',
    'HCLTECH': 'Information Technology',
    'INFY': 'Information Technology',
    'LTTS': 'Information Technology',
    'LTIM': 'Information Technology',
    'MPHASIS': 'Information Technology',
    'PERSISTENT': 'Information Technology',
    'TCS': 'Information Technology',
    'TECHM': 'Information Technology',
    'WIPRO': 'Information Technology',

    # Oil Gas & Consumable Fuels (with additions)
    'ATGL': 'Oil Gas & Consumable Fuels',
    'AEGISLOG': 'Oil Gas & Consumable Fuels',
    'BPCL': 'Oil Gas & Consumable Fuels',
    'CASTROLIND': 'Oil Gas & Consumable Fuels',
    'GAIL': 'Oil Gas & Consumable Fuels',
    'GUJGASLTD': 'Oil Gas & Consumable Fuels',
    'GSPL': 'Oil Gas & Consumable Fuels',
    'HINDPETRO': 'Oil Gas & Consumable Fuels',
    'IOC': 'Oil Gas & Consumable Fuels',
    'IGL': 'Oil Gas & Consumable Fuels',
    'MGL': 'Oil Gas & Consumable Fuels',
    'ONGC': 'Oil Gas & Consumable Fuels',
    'OIL': 'Oil Gas & Consumable Fuels',
    'PETRONET': 'Oil Gas & Consumable Fuels',
    'RELIANCE': 'Oil Gas & Consumable Fuels',
    }

    sector_pairs = {}
    for stock1, stock2, p_value in pairs:
        sector1 = sectors.get(stock1)
        sector2 = sectors.get(stock2)
        if sector1 and sector2 and sector1 == sector2:
            if sector1 not in sector_pairs:
                sector_pairs[sector1] = []
            sector_pairs[sector1].append((stock1, stock2, p_value))
    
    # Filter out sectors with pairs where one stock is bullish and another is bearish
    stagnant_sectors = {}
    for sector, sector_pairs in sector_pairs.items():
        bullish_bearish_pairs = [pair for pair in sector_pairs]
        if bullish_bearish_pairs:
            stagnant_sectors[sector] = bullish_bearish_pairs

    return stagnant_sectors

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/fetch_pairs', methods=['GET'])
def fetch_pairs():
    start_date = '2023-01-01'  # Example start date
    end_date = '2024-01-01'    # Example end date
    
    data = get_stock_data(NIFTY_500_TICKERS, start_date, end_date)
    data = align_data(data)
    pairs = identify_pairs(data)
    stagnant_sectors = analyze_sectors(pairs)
    
    return jsonify(stagnant_sectors)

if __name__ == '__main__':
    app.run(debug=True)
