import yfinance as yf
import pandas as pd

# Sample list of Nifty 500 stock tickers
nifty500_tickers = ['360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS', 'AWL.NS', 'ABCAPITAL.NS', 
                    'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 
                    'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 
                    'ARE&M.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 
                    'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 
                    'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 
                    'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 
                    'DMART.NS', 'AXISBANK.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 
                    'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 
                    'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 
                    'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 
                    'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 
                    'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 
                    'BSOFT.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 
                    'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 
                    'CESC.NS', 'CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 
                    'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 
                    'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 
                    'CDSL.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 
                    'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHOLAHLDNG.NS', 
                    'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 
                    'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 
                    'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 
                    'CUMMINSIND.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 
                    'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 
                    'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 
                    'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 
                    'EIDPARRY.NS', 'EIHOTEL.NS', 'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 
                    'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 
                    'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FDC.NS', 
                    'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 
                    'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 
                    'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 
                    'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 
                    'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 
                    'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 
                    'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 
                    'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 
                    'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 
                    'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HSCL.NS', 'HINDALCO.NS', 
                    'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 
                    'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 
                    'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 
                    'IDFCFIRSTB.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 
                    'ITI.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 
                    'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 'IRFC.NS', 'INDIGOPNTS.NS', 
                    'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 
                    'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 
                    'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 
                    'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'JAMNAAUTO.NS', 
                    'JAYBARM.NS', 'JAYPEEASSO.NS', 'JBMGROUP.NS', 'JINDALSTEL.NS', 
                    'JSLHITACHI.NS', 'JSL.LTD', 'JINDALPOLY.NS', 'JYOTHYLAB.NS', 'KOTAKBANK.NS', 
                    'KSB.NS', 'KARURVYSYA.NS', 'KIRLOSKAR.O.NS', 'KIRLOSKARSYN.NS', 'KPRMILL.NS', 
                    'KIRLAPWR.NS', 'KUSHAL.NS', 'LTTS.NS', 'LUMAXTECH.NS', 'LUPIN.NS', 
                    'LYD.NS', 'MOTHERSON.NS', 'MUTHOOTFIN.NS', 'MANAPPURAM.NS', 'MAHINDCIE.NS', 
                    'MAHINDRA.NS', 'MPSLTD.NS', 'MRO-TEK.NS', 'MRF.NS', 'MSTC.NS', 'MINDTREE.NS', 
                    'MEGH.NS', 'MANORAMA.NS', 'METROPOLIS.NS', 'MINDTREE.NS', 'MAXHEALTH.NS', 
                    'NATIONALUM.NS', 'NESTLEIND.NS', 'NOCIL.NS', 'NCR.NS', 'NIITLTD.NS', 
                    'NITESH.NS', 'NTPC.NS', 'NATCO.NS', 'NHPC.NS', 'NINL.NS', 'NITINSPINNER.NS', 
                    'NVR.NS', 'NATURE.NS', 'NRB.NS', 'NETVESTORS.NS', 'NCS.NS', 'NEULAND.NS', 
                    'NATURAL.NS', 'OBEROIRLTY.NS', 'OIL.NS', 'ONGC.NS', 'ONEMOBILITY.NS', 
                    'ORG_NSN.NS', 'PBR.LTD', 'PEL.NS', 'PRISMJOHNDAL.NS', 'PRISMJO.NS', 
                    'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PVR.NS', 'PNB.NS', 
                    'PLASTIBLEND.NS', 'PPAPIND.NS', 'PPI.NS', 'PRAGMATIC.NS', 'PPA.NS', 
                    'PROZONE.NS', 'PUNJABALCO.NS', 'PUNJLLOYD.NS', 'PUNJAB.NS', 'PVR.LTD', 
                    'RAYMOND.NS', 'RECLTD.NS', 'RELIANCE.NS', 'SBI.NS', 'SILVER.BONDS', 
                    'SITSL.NS', 'SYNGENE.NS', 'SYNGENE.NS', 'TATAMOTORS.NS', 'TATAMETAL.NS', 
                    'TATACHEM.NS', 'TATAELXSI.NS', 'TATAGLOBAL.NS', 'TATAINVEST.NS', 
                    'TATAPOWER.NS', 'TATAPHARM.NS', 'TATAMETALIKS.NS', 'TATASPL.NS', 
                    'TCS.NS', 'TGBL.NS', 'TITAN.NS', 'TORMETAL.NS', 'TORNTPOWER.NS', 
                    'TRIDENT.NS', 'TRF.NS', 'TV18BRDCST.NS', 'TVSMOTOR.NS', 'UDAYJYOTI.NS', 
                    'UMI.NS', 'UPL.NS', 'UNITEDPHAR.NS', 'VIPIND.NS', 'VOLTAS.NS', 
                    'VSTIND.NS', 'WABAG.NS', 'WELSPUNIND.NS', 'WELCORP.NS', 'WILLH.BST', 
                    'YESBANK.NS', 'ZEEENT.NS', 'ZYDUSLIFE.NS']

def fetch_stock_info(tickers):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Use stock.info.get() to handle missing data
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')

        data.append({
            'Ticker': ticker,
            'Name': info.get('shortName', 'N/A'),
            'Sector': sector,
            'Industry': industry,
            'Market Cap': info.get('marketCap', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A')
        })
        
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('stock_data.csv', index=False)
    print("Data saved to 'stock_data.csv'")

# Fetch and save stock info
fetch_stock_info(nifty500_tickers)
