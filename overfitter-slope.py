import pandas as pd
import numpy as np
from backtesting import run_backtest

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

def crossover(data, col1, col2, i): 
    if data[col1].iloc[i] > data[col2].iloc[i] and data[col1].iloc[i-1] < data[col2].iloc[i-1]:
        return True
    else:
        return False
    
def process_data(data, f, m, s):
    # Drop rows with zero volume
    data.drop(data[data['volume'] == 0].index, inplace=True)

    # Define HMA periods based on date ranges
    data['FHMA'] = np.nan
    data['MHMA'] = np.nan
    data['SHMA'] = np.nan

    # Calculate HMA values for the entire DataFrame
    fhma = hma(data['close'], f)
    mhma = hma(data['close'], m)
    shma = hma(data['close'], s)

    # Assign HMA values based on date ranges
    data['FHMA'] = fhma
    data['SHMA'] = shma
    data['MHMA'] = mhma
    
def strat(data):
    #data.dropna(inplace=True)
    signal = [0] * len(data)
    sum = 0

    for i in range(200, len(data)):
        if sum == 0:
            if crossover(data, 'FHMA', 'MHMA', i):
                if data['MHMA'].iloc[i] > data['SHMA'].iloc[i]:
                    s1 = data['MHMA'].iloc[i] - data['MHMA'].iloc[i-2]
                    if s1 > 0:
                        signal[i] = 1
                        sum = 1
                    # signal[i] = 1
                    # sum = 1
            
            elif crossover(data, 'MHMA', 'FHMA', i):
                if data['MHMA'].iloc[i] < data['SHMA'].iloc[i]:
                    s2 = data['MHMA'].iloc[i] - data['MHMA'].iloc[i-2]
                    if s2 < 0:
                        signal[i] = -1
                        sum = -1
                    # signal[i] = -1
                    # sum = -1

        elif sum == 1:
            if crossover(data, 'MHMA', 'FHMA', i):
                if data['MHMA'].iloc[i] < data['SHMA'].iloc[i]:
                    signal[i] = -2
                    sum = -1
            
            elif data['FHMA'].iloc[i] < data['MHMA'].iloc[i]:
                signal[i] = -1
                sum = 0

        elif sum == -1:
            if crossover(data, 'FHMA', 'MHMA', i):
                if data['MHMA'].iloc[i] > data['SHMA'].iloc[i]:
                    signal[i] = 2
                    sum = 1
            
            elif data['FHMA'].iloc[i] > data['MHMA'].iloc[i]:
                signal[i] = 1
                sum = 0

    data['signals'] = signal

    data = data[['open', 'high', 'low', 'close', 'volume', 'signals']].copy()
    data.reset_index(inplace=True)
    return data

def loop(f, m, s):
    data = pd.read_csv('data/ETHUSDT_1h_data.csv', parse_dates=['datetime'], index_col='datetime')
    
    process_data(data,f,m,s)
    
    strategy_signals = strat(data)

    strategy_signals.to_csv("results.csv", index=False)

    run_backtest('results.csv')
    
if __name__ == "__main__":
    print(loop(30,35,180))