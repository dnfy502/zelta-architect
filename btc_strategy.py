import pandas as pd
from untrade.client import Client
from pprint import pprint
import talib as ta
import numpy as np

def hma(series, period):
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    wma_half = ta.WMA(series, half_length)
    wma_full = ta.WMA(series, period)
    hma_series = ta.WMA(2 * wma_half - wma_full, sqrt_length)
    return hma_series

def supertrend(df, period=7, multiplier=3):
    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = [True] * len(df)  # True indicates uptrend, False indicates downtrend
    for i in range(1, len(df)):
        if df['close'][i] > upper_band[i - 1]:
            supertrend[i] = True
        elif df['close'][i] < lower_band[i - 1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i - 1]
            upper_band[i] = min(upper_band[i], upper_band[i - 1]) if supertrend[i] else upper_band[i]
            lower_band[i] = max(lower_band[i], lower_band[i - 1]) if not supertrend[i] else lower_band[i]

    df['supertrend'] = supertrend
    df['supertrend_ub'] = upper_band
    df['supertrend_lb'] = lower_band

    return df

def process_data(data):
    data.drop(data[data['volume'] == 0].index, inplace=True)

    hma_fast_period = 10
    hma_slow_period = 30

    # Calculate HMA
    data['hma_fast'] = hma(data['close'], hma_fast_period)
    data['hma_slow'] = hma(data['close'], hma_slow_period)

    # Calculate Supertrend
    data = supertrend(data, period=7, multiplier=3)

    # Define entry/exit signals row-wise
    def entry_exit_signal(row):
        if row['hma_fast'] > row['hma_slow'] and row['supertrend']:  
            return 1  # Buy signal
        elif row['hma_fast'] < row['hma_slow'] and not row['supertrend']:  
            return -1  # Sell signal
        else:
            return 0  # No signal

    # Apply the signal generation to each row
    data['Signal'] = data.apply(entry_exit_signal, axis=1)

    return data

def strat(data):
    data.dropna(inplace=True)
    signal = []
    trade_type = []
    prev = None
    for value in data["Signal"]:
        if value == prev:
            signal.append(0)
        else:
            signal.append(value)
        prev = value
    
    # getting trade signals 
    count = 0
    for value in signal:
        if value == 0:
            trade_type.append('')
        elif count == 1:
            if value == -1:
                trade_type.append('close')
                count = 0
            else:
                trade_type.append('')
        elif count == -1:
            if value == 1:
                trade_type.append('close')
                count = 0
            else:
                trade_type.append("")
        elif count == 0:
            if value == 1:
                trade_type.append('long')
                count = 1
            elif value == -1:
                trade_type.append('short')
                count = -1
            else:
                trade_type.append('')
        else:
            trade_type.append('')

    data["signals"] = signal
    data["trade_type"] = trade_type
    
    # Keep only the necessary columns including datetime
    data = data[['open', 'high', 'low', 'close', 'volume', 'signals', 'trade_type']].copy()
    data.reset_index(inplace=True)  
    return data

def perform_backtest(csv_file_path):
    # Create an instance of the untrade client
    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        jupyter_id="dnfy.",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=1,
    )

    return result

def main():
    # Read data from CSV file
    data = pd.read_csv('BTC_2019_2023_2h.csv', parse_dates=['datetime'], index_col='datetime')

    # Process data
    processed_data = process_data(data)

    # Apply strategy
    strategy_signals = strat(processed_data)

    # Save processed data to CSV file
    strategy_signals.to_csv("results.csv",index=False)
    
    backtest_result = perform_backtest("results.csv")
    print(backtest_result)
    for value in backtest_result:
        print(value)

if __name__ == "__main__":
    main()
