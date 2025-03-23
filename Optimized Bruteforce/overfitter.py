import pandas as pd
import numpy as np
from backtesting import overfit_result
import multiprocessing as mp
import os
import time
from functools import partial
import datetime  # Add this for better time formatting

# Start timing at the very beginning of the script
start_time = time.time()

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

def loop(params, lock):
    try:
        f, m, s = params
        # Create a unique filename for this process
        pid = os.getpid()
        results_file = f"results_{pid}.csv"
        
        # Load data
        data = pd.read_csv(r'/home/dnfy/Desktop/AlgoTrading/Optimized Bruteforce/ETHUSDT_2h_data.csv', parse_dates=['datetime'], index_col='datetime')
        
        process_data(data, f, m, s)
        
        strategy_signals = strat(data)

        strategy_signals.to_csv(results_file, index=False)
        
        result = overfit_result(results_file)
        
        # Clean up temp file
        try:
            os.remove(results_file)
        except:
            pass
        
        # Create a dictionary with parameters and result
        results_dict = {
            'fast': f,
            'medium': m,
            'slow': s,
            'Final_Balance': result[0],
            'Win Rate': result[1],
            'Max Drawdown': result[2],
            'Sharpe Ratio': result[3],
            'Average TTR': result[4],
            'Max TTR': result[5],
            'No of Trades': result[6],
            'No of longs': result[7],
            'No of shorts': result[8],
            'Max Win': result[9],
            'Max Loss': result[10]
        }
        
        # Convert to DataFrame
        results_df = pd.DataFrame([results_dict])
        
        # Use lock to safely write to the shared CSV file
        with lock:
            # Append to CSV file if it exists, create new if it doesn't
            results_df.to_csv('parameter_results.csv', 
                             mode='a', 
                             header=not os.path.exists('parameter_results.csv'),
                             index=False)
        
        # print(f"Completed f={f}, m={m}, s={s}")
        return result
    
    except Exception as e:
        print(f"Error processing f={f}, m={m}, s={s}: {e}")
        return None
    
if __name__ == "__main__":
    print(f"Starting parameter sweep at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    # Create parameters list
    params_list = [(f, m, s) for s in range(30, 200, 5) for m in range(15, s, 2) for f in range(10, m, 2)]
    total_combinations = len(params_list)
    print(f"Total parameter combinations to test: {total_combinations}")
    
    # Clear any existing results file
    if os.path.exists('parameter_results.csv'):
        os.remove('parameter_results.csv')
    
    # Create a multiprocessing lock for safe file writing
    lock = mp.Manager().Lock()
    
    # Limit processes to avoid memory issues
    num_processes = min(32, mp.cpu_count() - 2)  # Adjust as needed
    print(f"Starting with {num_processes} processes")
    
    # Create a process pool with the partial function that includes the lock
    pool = mp.Pool(processes=num_processes)
    loop_with_lock = partial(loop, lock=lock)
    
    # Process parameters in smaller batches to reduce memory pressure
    batch_size = 100  # Adjust as needed
    all_results = []
    
    # Add batch timing
    batch_start_time = time.time()
    
    for i in range(0, len(params_list), batch_size):
        batch = params_list[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(params_list)-1)//batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} combinations)")
        
        # Process the batch
        results = pool.map(loop_with_lock, batch)
        all_results.extend(results)
        
        # Force garbage collection between batches
        import gc
        gc.collect()
        
        # Short pause to allow system to recover resources
        time.sleep(0.05)
        
        # Show progress and estimated time remaining
        combinations_done = min((i + batch_size), total_combinations)
        percent_done = combinations_done / total_combinations * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / percent_done * 100 if percent_done > 0 else 0
        time_remaining = estimated_total_time - elapsed_time
        
        # Format times as hours:minutes:seconds
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        remaining_str = str(datetime.timedelta(seconds=int(time_remaining)))
        
        print(f"Progress: {percent_done:.1f}% ({combinations_done}/{total_combinations})")
        print(f"Time elapsed: {elapsed_str}, estimated time remaining: {remaining_str}")
        print(f"Batch processing time: {time.time() - batch_start_time:.2f} seconds")
        
        # Reset batch timer
        batch_start_time = time.time()
    
    # Close and join the pool
    pool.close()
    pool.join()
    
    # Calculate and display the total run time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"All processing complete!")
    print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Average time per parameter combination: {total_time/total_combinations:.2f} seconds")
    print("="*50)

