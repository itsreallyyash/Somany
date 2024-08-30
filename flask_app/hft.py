import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

# Define the stock symbol
stock_symbol = 'HCC.NS'

# Fetch the initial data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d', interval='1m')  # 1-day history with 1-minute intervals
    return data

# Initialize plot
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot([], [], lw=2)
time_format = mdates.DateFormatter('%H:%M')

def init():
    ax.set_xlim(pd.Timestamp.now() - pd.DateOffset(minutes=30), pd.Timestamp.now())
    ax.set_ylim(0, 100)  # Adjust based on expected stock price range
    ax.xaxis.set_major_formatter(time_format)
    return line,

def update(frame):
    global x_data, y_data  # Use global to modify the x_data and y_data lists
    
    # Fetch latest data
    data = fetch_stock_data(stock_symbol)
    
    # Update x and y data
    x_data = list(data.index)
    y_data = list(data['Close'])
    
    # Keep the last 60 minutes of data for the plot
    x_data = x_data[-60:]
    y_data = y_data[-60:]
    
    line.set_data(x_data, y_data)
    ax.relim()
    ax.autoscale_view()
    return line,

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=60000)  # Update every minute
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{stock_symbol} Real-Time Price')
plt.show()
