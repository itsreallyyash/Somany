import yfinance as yf
import pandas as pd
import numpy as np

# Define the stocks and parameters
stocks = ['360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS','AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'AEGISLOG.NS', 'AETHER.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'ARE&M.NS', 'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 'AXISBANK.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BORORENEW.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS','CGPOWER.NS', 'CIEINDIA.NS', 'CRISIL.NS', 'CSBBANK.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'LALPATHLAB.NS', 'DRREDDY.NS', 'DUMMYRAYMD.NS', 'DUMMYSANOF.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'EPL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FDC.NS', 'NYKAA.NS', 'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GAIL.NS', 'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLS.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 'GAEL.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'HEG.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONAUT.NS', 'HUDCO.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDIACEM.NS', 'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IRCTC.NS', 'IRFC.NS', 'INDIGOPNTS.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JAIBALAJI.NS', 'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 'KPITTECH.NS', 'KRBL.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KOTAKBANK.NS', 'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LICI.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'MTARTECH.NS', 'LODHA.NS', 'MGL.NS', 'MAHSEAMLES.NS', 'M&MFIN.NS', 'M&M.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MSUMI.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 'NLCINDIA.NS', 'NMDC.NS', 'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 'NAZARA.NS', 'NFL.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NILKAMAL.NS', 'NOCIL.NS', 'NOVOCO.NS', 'NUVOCO.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OLECTRA.NS', 'ONE97.NS', 'ORIENTELEC.NS', 'PAYTM.NS', 'PNB.NS', 'PFS.NS', 'PGHH.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PVR.NS', 'PATELENG.NS', 'PIRAMALENT.NS', 'PRECISION.NS', 'PRISMJOINTS.NS', 'PROCTER.NS', 'PRISM.NS', 'PRIMO.NS', 'PROFINS.NS', 'RBLBANK.NS', 'RBL.NS', 'RECLTD.NS', 'RELINFRA.NS', 'ROHL.NS', 'RANBAXY.NS', 'RIL.NS', 'RPG.NS', 'RUPA.NS', 'RELIANCE.NS', 'RTNINDIA.NS', 'SEQUENT.NS', 'SFL.NS', 'SILVER.NS', 'SOMANYCERA.NS', 'SHAREINDIA.NS', 'SANDHAR.NS', 'SUNDARAMFAST.NS', 'SUNDARAMFIN.NS', 'SUSHIL.NS', 'SUNPHARMA.NS', 'SYNDICATE.NS', 'SYNGENE.NS', 'SYNPHARM.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATAMETALIK.NS', 'TATACHEM.NS', 'TATASTEEL.NS', 'TCNSBRANDS.NS', 'TECHM.NS', 'TEJASNET.NS', 'TEN.NS', 'TERASOFT.NS', 'THYROCARE.NS', 'TITAN.NS', 'TOUCHWOOD.NS', 'TROYTECH.NS', 'TRITON.NS', 'UCOBANK.NS', 'UPL.NS', 'UJJIVAN.NS', 'UNIPOS.NS', 'UNIONBANK.NS', 'UTIAMC.NS', 'VAIBHAVGBL.NS', 'VBL.NS', 'VIRINCHI.NS', 'VIPIND.NS', 'VST.NS', 'VSTIND.NS', 'VBL.NS', 'VTL.NS', 'V2RETAIL.NS', 'VIGIL.NS', 'VSTLTD.NS', 'VIRINCHI.NS', 'VIRINCHI.NS', 'YESBANK.NS', 'YUMBRANDS.NS', 'ZEE.NS', 'ZOMATO.NS']
initial_investment = 10000  # Fixed initial investment per stock

def calculate_moving_averages(df, short_window=9, long_window=21):
    df['SMA9'] = df['Close'].rolling(window=short_window).mean()
    df['SMA21'] = df['Close'].rolling(window=long_window).mean()
    return df

def generate_signals(df):
    df['Signal'] = 0
    df['Signal'] = np.where((df['SMA9'].shift(1) <= df['SMA21'].shift(1)) & (df['SMA9'] > df['SMA21']), 1, df['Signal'])
    df['Signal'] = np.where((df['SMA9'].shift(1) >= df['SMA21'].shift(1)) & (df['SMA9'] < df['SMA21']), -1, df['Signal'])
    return df

def backtest(stock, initial_investment):
    try:
        # Download stock data for the last 1 year
        df = yf.download(stock, period='5y', interval='1d')
        if df.empty:
            print(f"No data for {stock}. Skipping...")
            return None
        
        # Calculate moving averages and generate buy/sell signals
        df = calculate_moving_averages(df)
        df = generate_signals(df)

        cash = initial_investment
        shares = 0
        position_opened = False  # Tracks if we have an open position

        for i in range(len(df)):
            if df['Signal'].iloc[i] == 1 and not position_opened:  # Buy condition
                shares = cash / df['Close'].iloc[i]  # Buy as many shares as possible with the initial investment
                cash = 0
                position_opened = True
                print(f"Bought {shares} shares of {stock} at {df['Close'].iloc[i]} on {df.index[i]}")
                
            elif df['Signal'].iloc[i] == -1 and position_opened:  # Sell condition
                cash = shares * df['Close'].iloc[i]  # Sell all shares
                shares = 0
                position_opened = False
                print(f"Sold all shares of {stock} at {df['Close'].iloc[i]} on {df.index[i]}")

        # Close any open position at the end of the year
        if position_opened:
            cash = shares * df['Close'].iloc[-1]  # Sell all remaining shares at the last available price
            print(f"Sold remaining shares of {stock} at {df['Close'].iloc[-1]} on {df.index[-1]}")
            shares = 0

        # Calculate final portfolio value
        final_value = cash
        profit_loss = final_value - initial_investment
        percentage_return = (profit_loss / initial_investment) * 100

        return {
            'Stock': stock,
            'Final Portfolio Value': final_value,
            'Profit/Loss': profit_loss,
            'Percentage Return (%)': percentage_return
        }
    except Exception as e:
        print(f"Error processing {stock}: {e}")
        return None

def main():
    total_initial_investment = 0
    total_final_value = 0
    results = []

    for stock in stocks:
        print(f"Processing {stock}...")
        result = backtest(stock, initial_investment)
        if result:
            results.append(result)
            total_initial_investment += initial_investment
            total_final_value += result['Final Portfolio Value']

    # Calculate overall portfolio return
    total_profit_loss = total_final_value - total_initial_investment
    overall_percentage_return = (total_profit_loss / total_initial_investment) * 100

    # Save individual stock results
    results_df = pd.DataFrame(results)
    results_df.to_csv('backtest_results.csv', index=False)

    # Display overall results
    print("\n--- Overall Portfolio Performance ---")
    print(f"Total Initial Investment: {total_initial_investment}")
    print(f"Total Final Value: {total_final_value}")
    print(f"Total Profit/Loss: {total_profit_loss}")
    print(f"Overall Percentage Return: {overall_percentage_return:.2f}%")

if __name__ == '__main__':
    main()

#gnn temporal (thursday)
#quarterly call puts and comparison (thursday)
#neutral market (thursday)
#pcr (nifty 50) + sma    too many puts (market increase) (scalping) 
#pcr + temporal
#vcp

#thursday 430 cci


