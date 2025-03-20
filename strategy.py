import pandas as pd
import numpy as np


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
    
def process_data(data):
    # Drop rows with zero volume
    data.drop(data[data['volume'] == 0].index, inplace=True)

    # Define HMA periods based on date ranges
    data['FHMA'] = np.nan
    data['MHMA'] = np.nan
    data['SHMA'] = np.nan

    # Calculate HMA values for the entire DataFrame
    hma_9 = hma(data['close'], 9)
    hma_12 = hma(data['close'], 12)
    hma_50 = hma(data['close'], 50)

    # Assign HMA values based on date ranges
    data['FHMA'] = hma_9
    data['SHMA'] = hma_50
    data['MHMA'] = hma_12
    
def strat(data):
    #data.dropna(inplace=True)
    signal = [0] * len(data)
    sum = 0

    for i in range(100, len(data)):
        if sum == 0:
            if crossover(data, 'FHMA', 'MHMA', i):
                if data['MHMA'].iloc[i] > data['SHMA'].iloc[i]:
                    signal[i] = 1
                    sum = 1
            
            elif crossover(data, 'MHMA', 'FHMA', i):
                if data['MHMA'].iloc[i] < data['SHMA'].iloc[i]:
                    signal[i] = -1
                    sum = -1

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

def main():
    data = pd.read_csv('data/ETHUSDT_1h_data.csv', parse_dates=['datetime'], index_col='datetime')
    
    process_data(data)
    
    strategy_signals = strat(data)
    print(strategy_signals)
    strategy_signals.to_csv("results.csv", index=False)
    
    for i in range(len(strategy_signals)):
        if sum(strategy_signals[:i]['signals']) > 1:
            print(i, 'wtf')

if __name__ == "__main__":
    main()