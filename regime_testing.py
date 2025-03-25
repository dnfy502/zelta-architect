import pandas as pd
import numpy as np
from backtesting import run_backtest
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def atr(data, period=14):
    """
    Calculate Average True Range (ATR)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Must contain 'high', 'low', and 'close' columns
    period : int
        The period over which to calculate ATR, default is 14
        
    Returns:
    --------
    pandas.Series
        The ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def sma(series, period):
    return pd.Series(series).rolling(window=period).mean()

def kama(series, er_period=10, fast_period=2, slow_period=21):
    # Convert to pandas Series if not already
    price = pd.Series(series)
    
    # Calculate the change in price
    change = price.diff(er_period).abs()
    
    # Calculate the volatility (sum of absolute price changes over er_period)
    volatility = price.diff().abs().rolling(window=er_period).sum()
    
    # Avoid division by zero
    volatility = volatility.replace(0, 0.00001)
    
    # Calculate Efficiency Ratio (ER)
    er = change / volatility
    
    # Calculate smoothing constant (SC)
    # SC = [ER x (fastest SC - slowest SC) + slowest SC]^2
    fast_alpha = 2 / (fast_period + 1)
    slow_alpha = 2 / (slow_period + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    
    # Calculate KAMA
    kama_result = pd.Series(index=price.index, dtype='float64')
    
    # Initialize with simple average of first er_period values
    first_value_index = er_period
    while pd.isna(price.iloc[first_value_index]) and first_value_index < len(price) - 1:
        first_value_index += 1
    
    if first_value_index < len(price):
        kama_result.iloc[first_value_index] = price.iloc[:er_period+1].mean()
        
        # Calculate subsequent KAMA values
        for i in range(first_value_index + 1, len(price)):
            if not pd.isna(price.iloc[i]) and not pd.isna(sc.iloc[i]) and not pd.isna(kama_result.iloc[i-1]):
                kama_result.iloc[i] = kama_result.iloc[i-1] + sc.iloc[i] * (price.iloc[i] - kama_result.iloc[i-1])
    
    return kama_result

def process_data(data, f=None, m=None, s=None):
    # Drop rows with zero volume
    data.drop(data[data['volume'] == 0].index, inplace=True)
    
    data['ATR'] = atr(data)
    data['HMA'] = hma(data['close'], 14)   
    data['SMA'] = sma(data['close'], 14)
    # Add KAMA with default parameters
    data['KAMA'] = kama(data['close'])
    # You can also add KAMA with custom parameters
    # data['KAMA_FAST'] = kama(data['close'], er_period=10, fast_period=2, slow_period=20)
    
    data['NATR'] = data['ATR'] / data['SMA']

    data['KAMA'] = kama(data['close'])
    
    

def strat(data):
    #data.dropna(inplace=True)
    signal = [0] * len(data)
    sum = 0

    for i in range(200, len(data)):
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

def plot_close_and_atr(data):
    """
    Plot the close price and ATR using Plotly
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Must contain 'close' and 'ATR' columns
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add close price line
    fig.add_trace(
        go.Scatter(x=data.index, y=data['close'], name="Close Price"),
        secondary_y=False,
    )
    
    # Add ATR line
    fig.add_trace(
        go.Scatter(x=data.index, y=data['KAMA'], name="KAMA", line=dict(color='red')),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text="Price and ATR Over Time",
        xaxis_title="Date",
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="ATR", secondary_y=False)
    
    fig.show()

def loop(f, m, s):
    data = pd.read_csv('data/ETHUSDT_1h_data.csv', parse_dates=['datetime'], index_col='datetime')
    
    process_data(data,f,m,s)
    
    strategy_signals = strat(data)

    strategy_signals.to_csv("results.csv", index=False)

    run_backtest('results.csv')
    
if __name__ == "__main__":
    #print(loop(30,35,180))
    data = pd.read_csv('data/ETHUSDT_1w_data.csv', parse_dates=['datetime'], index_col='datetime')
    
    process_data(data)
    print(data)
    
    # Plot close price and ATR
    plot_close_and_atr(data)