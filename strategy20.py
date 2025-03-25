import pandas as pd
import numpy as np
import talib as ta

def hma(series, period):
    # Calculate weights
    weights = np.arange(1, period + 1)
    
    # Calculate WMA with half period
    half = period // 2
    wmaf = pd.Series(series).rolling(window=half, center=False).apply(
        lambda x: np.sum(weights[:half] * x) / weights[:half].sum(), raw=True)
    
    # Calculate WMA with full period
    wma = pd.Series(series).rolling(window=period, center=False).apply(
        lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    # Calculate raw HMA
    raw_hma = 2 * wmaf - wma
    
    # Calculate final HMA using sqrt(period)
    sqrt_period = int(np.sqrt(period))
    sqrt_weights = np.arange(1, sqrt_period + 1)
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, center=False).apply(
        lambda x: np.sum(sqrt_weights * x) / sqrt_weights.sum(), raw=True)
    
    return hma

def process_data(data):
    # Drop rows with zero volume
    data.drop(data[data['volume'] == 0].index, inplace=True)

    # Define HMA periods based on date ranges
    data['FHMA'] = np.nan
    data['SHMA'] = np.nan

    # Calculate HMA values for the entire DataFrame
    hma_9 = hma(data['close'], 9)
    hma_12 = hma(data['close'], 12)
    hma_100 = hma(data['close'], 100)
    hma_15 = hma(data['close'], 15)

    # Assign HMA values based on date ranges
    data.loc["2020-01-01":"2021-12-31", 'FHMA'] = hma_9
    data.loc["2020-01-01":"2021-12-31", 'SHMA'] = hma_100
    data.loc["2022-01-01":"2022-12-31", 'FHMA'] = hma_12
    data.loc["2022-01-01":"2022-12-31", 'SHMA'] = hma_100
    data.loc["2023-01-01":"2024-01-01", 'FHMA'] = hma_15
    data.loc["2023-01-01":"2024-01-01", 'SHMA'] = hma_100

    # Calculate RSI
    rsi_period = 14
    data['RSI'] = ta.RSI(data['close'], timeperiod=rsi_period)

    # Calculate Bollinger Bands
    length = 20
    mult = 2
    sma_close = data['close'].rolling(window=length).mean()
    std_dev = data['close'].rolling(window=length).std()
    data['upper_band'] = sma_close + (mult * std_dev)
    data['lower_band'] = sma_close - (mult * std_dev)

    # Normalize Bollinger Bands width
    data['bb_width_normalized'] = ((data['upper_band'] - data['lower_band']) / sma_close) * 10

    # Define trading conditions
    def entry_exit_signal(row):
        bb_condition = 0.2 <= row['bb_width_normalized'] <= 2
        in_trade_period = not (row.name >= pd.Timestamp("2021-07-01") and row.name <= pd.Timestamp("2022-04-30 23:59"))
        if in_trade_period and pd.notna(row['FHMA']) and pd.notna(row['SHMA']):
            if row['FHMA'] > row['SHMA'] and row['RSI'] < 70 and bb_condition:
                return 1  # Buy signal
            elif row['FHMA'] < row['SHMA'] and row['RSI'] > 30 and bb_condition:
                return -1  # Sell signal
        return 0  # No signal

    data['Signal'] = data.apply(entry_exit_signal, axis=1)
    return data


def strat(data):
    data.dropna(inplace=True)
    signal = []
    sum = 0

    for i in range(len(data['Signal'])-1):
        current = data['Signal'][i]
        next = data['Signal'][i+1]
		
        if sum == 0:
            signal.append(current)
            sum = current
	
        elif sum == 1:
            if current == -1:
                if next == -1:
                    signal.append(-2)
                    sum = -1
                else:
                    signal.append(-1)
                    sum = 0
            else:
                signal.append(0)
	
        elif sum == -1:
            if current == 1:
                if next == 1:
                    signal.append(2)
                    sum = 1
                else:
                    signal.append(1)
                    sum = 0
            else:
                signal.append(0)
    if sum == 1:
        signal.append(-1)
    elif sum == -1:
        signal.append(1)
    else:
        signal.append(0)
    
    # signal = signal[1:] + [0]
    
    data['signals'] = signal
    data['trade_type'] = ''

    data = data[['open', 'high', 'low', 'close', 'volume', 'signals', 'trade_type']].copy()
    data.reset_index(inplace=True)
    return data

def main():
    data = pd.read_csv('ETHUSDT_4h.csv', parse_dates=['datetime'], index_col='datetime')

    processed_data = process_data(data)

    strategy_signals = strat(processed_data)

    strategy_signals.to_csv("results.csv", index=False)
    print(strategy_signals.head())

if __name__ == "__main__":
    main()