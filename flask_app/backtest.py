import yfinance as yf
import pandas as pd
import numpy as np

# Define the stocks and parameters
stocks = ['ACC.NS', 'BAJFINANCE.NS', 'NYKAA.NS', 'TATAMOTORS.NS']
initial_cash = 1000000  # Starting capital for each stock
investment_amount = 100000  # Amount to invest or withdraw at each trade

def calculate_moving_averages(df, short_window=9, long_window=21):
    df['SMA9'] = df['Close'].rolling(window=short_window).mean()
    df['SMA21'] = df['Close'].rolling(window=long_window).mean()
    return df

def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['SMA9'] > df['SMA21'], 'Signal'] = 1  # Buy signal
    df.loc[df['SMA9'] < df['SMA21'], 'Signal'] = -1  # Sell signal
    return df

def backtest(stock, initial_cash, investment_amount):
    try:
        df = yf.download(stock, period='5y', interval='1d')
        if df.empty:
            print(f"No data for {stock}. Skipping...")
            return None
        
        df = calculate_moving_averages(df)
        df = generate_signals(df)

        cash = initial_cash
        shares = 0
        invested_amount = 0

        for i in range(len(df)):
            if df['Signal'].iloc[i] == 1 and cash >= investment_amount:  # Buy
                shares_to_buy = investment_amount / df['Close'].iloc[i]
                shares += shares_to_buy
                cash -= investment_amount
                invested_amount += investment_amount
            elif df['Signal'].iloc[i] == -1 and shares * df['Close'].iloc[i] >= investment_amount:  # Partial Sell
                shares_to_sell = investment_amount / df['Close'].iloc[i]
                shares -= shares_to_sell
                cash += investment_amount

        if len(df) > 0:  # Ensure DataFrame is not empty
            final_value = shares * df['Close'].iloc[-1] + cash
        else:
            final_value = cash

        profit_loss = final_value - initial_cash
        percentage_return = (profit_loss / initial_cash) * 100

        return {
            'Stock': stock,
            'Total Invested': invested_amount,
            'Final Portfolio Value': final_value,
            'Profit/Loss': profit_loss,
            'Percentage Return (%)': percentage_return
        }
    except Exception as e:
        print(f"Error processing {stock}: {e}")
        return None

def main():
    results = []
    for stock in stocks:
        print(f"Processing {stock}...")
        result = backtest(stock, initial_cash, investment_amount)
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('backtest_results.csv', index=False)
    print("Backtest results saved to 'backtest_results.csv'.")

if __name__ == '__main__':
    main()