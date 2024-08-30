from kiteconnect import KiteConnect
import datetime as dt
import plotly.graph_objects as go
import pandas as pd
import numpy as np

api_key = "key"
access_token = "token"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

def get_stock_data(ticker, interval='day', period='6mo'):
    end_date = dt.datetime.today()
    start_date = end_date - pd.DateOffset(months=int(period.replace('mo', '')))

    interval_mapping = {
        '1d': 'day',
        '5m': '5minute',
        '15m': '15minute',
        '30m': '30minute',
        '1h': '60minute'
    }

    data = kite.historical_data(instrument_token=kite.ltp(ticker)[ticker]['instrument_token'],
                                from_date=start_date.strftime('%Y-%m-%d'),
                                to_date=end_date.strftime('%Y-%m-%d'),
                                interval=interval_mapping[interval])

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    return df

def calculate_moving_averages(data):
    data['9D MA'] = data['close'].rolling(window=9).mean()
    data['21D MA'] = data['close'].rolling(window=21).mean()
    return data

def calculate_trend_line(data):
    last_45_days = data[-45:]
    x = np.arange(len(last_45_days))
    z = np.polyfit(x, last_45_days['close'], 1)
    trend_line = np.polyval(z, x)
    data.loc[data.index[-45:], 'Trend Line'] = trend_line
    return data

def plot_interactive_candlestick(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['open'],
                                 high=data['high'],
                                 low=data['low'],
                                 close=data['close'],
                                 name='Candlesticks',
                                 increasing_line_color='green', 
                                 decreasing_line_color='red'))

    fig.add_trace(go.Scatter(x=data.index, y=data['9D MA'], 
                             line=dict(color='cyan', width=2), 
                             name='9-Day MA'))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['21D MA'], 
                             line=dict(color='magenta', width=2), 
                             name='21-Day MA'))

    fig.add_trace(go.Scatter(x=data.index[-45:], y=data['Trend Line'], 
                             line=dict(color='blue', width=2, dash='dash'), 
                             name='45-Day Trend Line'))

    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',  # Dark theme
        xaxis_rangeslider_visible=False
    )
    
    fig.show()

ticker = 'RELIANCE' 
stock_data = get_stock_data(ticker)
stock_data = calculate_moving_averages(stock_data)
stock_data = calculate_trend_line(stock_data)
plot_interactive_candlestick(stock_data, ticker)
# pip install kiteconnect plotly pandas numpy
