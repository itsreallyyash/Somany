from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import time

app = Flask(__name__)
nifty500_symbols = ['360ONE.NS', '3MINDIA.NS']
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
