import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

def get_stock_data(ticker):
    stock_data = yf.download(ticker, period='6mo', interval='60m')
    return stock_data

def calculate_moving_averages(data):
    data['9D SMA'] = data['Close'].rolling(window=9).mean()
    data['21D SMA'] = data['Close'].rolling(window=21).mean()
    return data

def calculate_ema(data):
    data['9D EMA'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['21D EMA'] = data['Close'].ewm(span=21, adjust=False).mean()
    return data

def calculate_trend_line(data):    
    data['Trend Line'] = np.nan
    if len(data) < 45:
        return data

    last_45_days = data[-45:]
    x = np.arange(len(last_45_days))
    z = np.polyfit(x, last_45_days['Close'], 1)
    trend_line = np.polyval(z, x)
    
    data.loc[data.index[-45:], 'Trend Line'] = trend_line
    return data
def plot_interactive_candlestick(data, ticker, show_sma=True, show_ema=False):
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlesticks',
                                 increasing_line_color='green', 
                                 decreasing_line_color='red'))

    if show_sma:
        fig.add_trace(go.Scatter(x=data.index, y=data['9D SMA'], 
                                 line=dict(color='cyan', width=2), 
                                 name='9-Day SMA'))
        fig.add_trace(go.Scatter(x=data.index, y=data['21D SMA'], 
                                 line=dict(color='magenta', width=2), 
                                 name='21-Day SMA'))
    
    if show_ema:
        fig.add_trace(go.Scatter(x=data.index, y=data['9D EMA'], 
                                 line=dict(color='cyan', width=2, dash='dash'), 
                                 name='9-Day EMA'))
        fig.add_trace(go.Scatter(x=data.index, y=data['21D EMA'], 
                                 line=dict(color='magenta', width=2, dash='dash'), 
                                 name='21-Day EMA'))

    # Trend line
    fig.add_trace(go.Scatter(x=data.index[-45:], y=data['Trend Line'], 
                             line=dict(color='yellow', width=3, dash='dash'), 
                             name='45-Day Trend Line'))

    # Volume bar chart
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                         name='Volume', 
                         marker_color='rgba(158,202,225,.8)', 
                         yaxis='y2'))

    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False, 
                    position=0.95),
        height=800,
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    fig.show()

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        ma_type = request.form['ma_type']

        stock_data = get_stock_data(ticker)
        stock_data = calculate_moving_averages(stock_data)
        stock_data = calculate_ema(stock_data)
        stock_data = calculate_trend_line(stock_data)

        show_sma = ma_type == 'sma'
        show_ema = ma_type == 'ema'

        plot_interactive_candlestick(stock_data, ticker, show_sma, show_ema)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=5001)
