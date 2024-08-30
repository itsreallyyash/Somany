from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import time

app = Flask(__name__)

# Example pairs
pairs = [('AAPL', 'MSFT'), ('GOOGL', 'AMZN')]

def fetch_data(symbol, period="1y", interval="1d"):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_spread(df1, df2):
    df1 = df1[['Close']]
    df2 = df2[['Close']]
    df1.columns = ['Close1']
    df2.columns = ['Close2']
    combined_df = pd.concat([df1, df2], axis=1)
    combined_df['Spread'] = combined_df['Close1'] - combined_df['Close2']
    return combined_df

def calculate_correlation(df1, df2):
    df1 = df1['Close']
    df2 = df2['Close']
    correlation = df1.corr(df2)
    return correlation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_pair = request.form.get('pair')
        symbol1, symbol2 = selected_pair.split(',')

        # Fetch data for both symbols
        data1 = fetch_data(symbol1)
        data2 = fetch_data(symbol2)
        
        # Calculate spread and correlation
        spread_df = calculate_spread(data1, data2)
        correlation = calculate_correlation(data1, data2)
        
        return render_template('results2.html', 
                               tables=[spread_df.to_html(classes='data')], 
                               titles=spread_df.columns.values, 
                               correlation=correlation)
    
    return render_template('index2.html', pairs=pairs)
if __name__ == "__main__":
    app.run(port=5001)