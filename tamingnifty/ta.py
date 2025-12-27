import pandas as pd
pd.set_option('mode.chained_assignment', None)
from tamingnifty import connect_definedge as edge
from datetime import datetime
import numpy as np
import copy

def atr(DF: pd.DataFrame, period: int):
    df = DF.copy()
    df['high-low'] = abs( df['high'] - df['low'] )
    df['high-previousclose'] = abs( df['high'] - df['close'].shift(1) )
    df['low-previousclose'] = abs( df['low'] - df['close'].shift(1) )
    df['TrueRange'] = df[ ['high-low', 'high-previousclose', 'low-previousclose'] ].max(axis=1, skipna=False)
    #df['ATR'] = df['TrueRange'].ewm(span=period, adjust=False, min_periods=period).mean()

    df['ATR'] = df['TrueRange'].ewm(com=period, min_periods = period).mean() #Very close to TradingView
    #df['ATR'] = df['TrueRange'].rolling(window=period).mean() #Very close to Definedge
    df['ATR'] = df['ATR'].round(2)
    return df['ATR']


def supertrend(df: pd.DataFrame, period: int, multiplier: int):
    df['ATR'] = atr(df, period)
    df['hl2'] = (df['high'] + df['low']) / 2
    df['basic_upperband'] = df['hl2'] + (df['ATR'] * multiplier)
    df['basic_lowerband'] = df['hl2'] - (df['ATR'] * multiplier)
    df['final_upperband'] = 0
    df['final_lowerband'] = 0
    df['ST'] = 0
    df['in_uptrend'] = True
    df['signal'] = "Bullish"
    for i in range(period, len(df)):
        # Update final bands
        df.at[i, 'final_upperband'] = df.at[i, 'basic_upperband'] if (df.at[i, 'basic_upperband'] < df.at[i-1, 'final_upperband']) or (df.at[i-1, 'close'] > df.at[i-1, 'final_upperband']) else df.at[i-1, 'final_upperband']
        df.at[i, 'final_lowerband'] = df.at[i, 'basic_lowerband'] if (df.at[i, 'basic_lowerband'] > df.at[i-1, 'final_lowerband']) or (df.at[i-1, 'close'] < df.at[i-1, 'final_lowerband']) else df.at[i-1, 'final_lowerband']    
    for current in range(1, len(df.index)):
        previous = current-1
        if df['close'][current] > df['final_upperband'][previous]:
            df['in_uptrend'][current] = True
            df['signal'][current] = "Bullish"
            df['ST'][current] = df['final_lowerband'][current]
        elif df['close'][current] < df['final_lowerband'][previous]:
            df['in_uptrend'][current] = False
            df['signal'][current] = "Bearish"
            df['ST'][current] = df['final_upperband'][current]
        else:
            df['in_uptrend'][current] = df['in_uptrend'][previous]
            if df['in_uptrend'][current]:
                df['ST'][current] = df['final_lowerband'][current]
                df['signal'][current] = "Bullish"
            else:
                df['ST'][current] = df['final_upperband'][current]
                df['signal'][current] = "Bearish"
    df['ST'] = df['ST'].round(2)
    df.drop(['basic_upperband', 'basic_lowerband', 'hl2', 'final_upperband', 'final_lowerband', 'in_uptrend', 'ATR'], axis='columns', inplace=True)
    return df


def ema_channel(df: pd.DataFrame, period = 21):
    df['ema_low'] = df['low'].ewm(com=10, min_periods = period).mean() #calculating EMA
    df['ema_high'] = df['high'].ewm(com=10, min_periods = period).mean() #calculating EMA

    df['ema_low'] = df['ema_low'].round(2)
    df['ema_high'] = df['ema_high'].round(2)
    df.dropna(subset=['ema_low', 'ema_high'], inplace=True)
    df['trend'] = np.where(df['close'] <= df['ema_low'], 'Bearish', 'Bullish')    
    return df

def tma(df: pd.DataFrame):
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean() #calculating EMA
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean() #calculating EMA
    df['ema_40'] = df['close'].ewm(span=40, adjust=False).mean() #calculating EMA

    df['ema_20'] = df['ema_20'].round(2)
    df['ema_30'] = df['ema_30'].round(2)
    df['ema_40'] = df['ema_40'].round(2)
    return df


def tma_custom(df: pd.DataFrame, p1, p2, p3):
    df['ema_p1'] = df['close'].ewm(span=p1, adjust=False).mean() #calculating EMA
    df['ema_p2'] = df['close'].ewm(span=p2, adjust=False).mean() #calculating EMA
    df['ema_p3'] = df['close'].ewm(span=p3, adjust=False).mean() #calculating EMA

    df['ema_p1'] = df['ema_p1'].round(2)
    df['ema_p2'] = df['ema_p2'].round(2)
    df['ema_p3'] = df['ema_p3'].round(2)
    return df


def rs(df: pd.DataFrame, nifty: pd.DataFrame):
    stock = df.copy()
    if stock['close'].min() < 100:
        multiplier = 10000
    else:
        multiplier = 1000
    stock['close'] = (stock['close'] * multiplier)/nifty['close']
    stock['close'] = stock['close'].round(2)
    stock.drop(['open','high','low','volume'], axis='columns', inplace=True)
    return stock

def sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate the Simple Moving Average (SMA) on the close price.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'close' column.
    period (int): The period for calculating the SMA.

    Returns:
    pd.DataFrame: The original DataFrame with an additional 'sma' column.
    """
    df['sma'] = df['close'].rolling(window=period).mean()
    df['sma'] = df['sma'].round(2)  # Round to 2 decimal places for better readability
    return df

def straddle_chart(call_df: pd.DataFrame, put_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure both dataframes have the same datetime index
    call_df['datetime'] = pd.to_datetime(call_df['datetime'])
    put_df['datetime'] = pd.to_datetime(put_df['datetime'])
    
    # Merge the dataframes on datetime
    combined_df = pd.merge(call_df, put_df, on='datetime', suffixes=('_call', '_put'))
    
    # Sum the OHLC data for call and put options
    combined_df['open'] = combined_df['open_call'] + combined_df['open_put']
    combined_df['high'] = combined_df['high_call'] + combined_df['high_put']
    combined_df['low'] = combined_df['low_call'] + combined_df['low_put']
    combined_df['close'] = combined_df['close_call'] + combined_df['close_put']
    
    # Select relevant columns
    combined_df = combined_df[['datetime', 'open', 'high', 'low', 'close']]
    
    return combined_df

def up_step(brick_size = .1, value=10.0):
    brick_size = float(brick_size)/100
    return round((value * brick_size),2)

def down_step(brick_size = .1, value=10.0):
    return round((value - round((value * 100) / (100 + brick_size),2)),2)


def renko(conn, exchange: str, trading_symbol: str, start: datetime, end: datetime, interval = 'min', brick_size = .1, last_high = 0, last_low=0, initial_color = "", initial_datetime = None) -> pd.DataFrame:
    df = edge.fetch_historical_data(conn, exchange, trading_symbol, start, end, interval)
    df['datetime'] = pd.to_datetime(df['datetime'])
    if last_high == 0 and last_low == 0:
        first_brick = {
            'datetime': df['datetime'].iloc[0],
            'low': round(df['open'].iloc[0],2),
            'high': round(df['open'].iloc[0],2),
            'color': 'green'
        }
    else:
        first_brick = {
            'datetime': initial_datetime if initial_datetime else df['datetime'].iloc[0],
            'low': last_low,
            'high': last_high,
            'color': initial_color if initial_color else 'green'
        }
        
    renko = [first_brick]
    dic =  df.to_dict(orient='records')
    for row in dic:
        if row['close'] >= renko[-1]['high'] + up_step(brick_size, renko[-1]['high']):
            while row['close'] >= renko[-1]['high'] + up_step(brick_size, renko[-1]['high']):
                new_brick=[{
                    'datetime': row['datetime'],
                    'low': round(renko[-1]['high'], 2),
                    'high': round((renko[-1]['high'] + up_step(brick_size, renko[-1]['high'])), 2),
                    'color': 'green'  
                }]
                renko = renko + new_brick
        if row['close'] <= renko[-1]['low'] - down_step(brick_size,renko[-1]['low']):
            while row['close'] <= renko[-1]['low'] - down_step(brick_size,renko[-1]['low']):
                new_brick=[{
                    'datetime': row['datetime'],
                    'low': round((renko[-1]['low'] - down_step(brick_size,renko[-1]['low'])), 2),
                    'high': round(renko[-1]['low'], 2),
                    'color': 'red'  
                }]
                renko = renko + new_brick
    df = pd.DataFrame(renko)
    df['close'] = np.where(df['color'] == 'red', df['low'], df['high'])
    return df 


def convert_to_renko(brick_size = .1, df: pd.DataFrame = pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['datetime'])
    try:
        first_brick = {
            'datetime': df['datetime'].iloc[0],
            'low': round(df['open'].iloc[0],2),
            'high': round(df['open'].iloc[0],2),
            'color': 'green'
        }
    except KeyError:
        first_brick = {
            'datetime': df['datetime'].iloc[0],
            'low': round(df['close'].iloc[0],2),
            'high': round(df['close'].iloc[0],2),
            'color': 'green'
        }
    renko = [first_brick]
    dic =  df.to_dict(orient='records')
    for row in dic:
        if row['close'] >= renko[-1]['high'] + up_step(brick_size, renko[-1]['high']):
            while row['close'] >= renko[-1]['high'] + up_step(brick_size, renko[-1]['high']):
                new_brick=[{
                    'datetime': row['datetime'],
                    'low': round(renko[-1]['high'], 2),
                    'high': round((renko[-1]['high'] + up_step(brick_size, renko[-1]['high'])), 2),
                    'color': 'green'  
                }]
                renko = renko + new_brick
        if row['close'] <= renko[-1]['low'] - down_step(brick_size,renko[-1]['low']):
            while row['close'] <= renko[-1]['low'] - down_step(brick_size,renko[-1]['low']):
                new_brick=[{
                    'datetime': row['datetime'],
                    'low': round((renko[-1]['low'] - down_step(brick_size,renko[-1]['low'])), 2),
                    'high': round(renko[-1]['low'], 2),
                    'color': 'red'  
                }]
                renko = renko + new_brick
    df = pd.DataFrame(renko)
    df['close'] = np.where(df['color'] == 'red', df['low'], df['high'])
    return df

def convert_to_heiken_ashi(df: pd.DataFrame, interval: str = '1min') -> pd.DataFrame:
    """
    Convert OHLC data to Heiken Ashi candles with optional resampling.
    Only returns fully formed candles (excludes running/incomplete candle).

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['datetime', 'open', 'high', 'low', 'close']
    interval (str): Pandas resample rule (e.g., '5min', '15min', '1H', '1D')

    Returns:
    pd.DataFrame: DataFrame with Heiken Ashi columns ['open', 'high', 'low', 'close', 'color', 'body_height']
    """
    import pandas as pd

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Calculate how many 1-min bars are needed for the interval
    freq = pd.Timedelta(interval)
    base_freq = pd.Timedelta('1min')
    bars_needed = int(freq / base_freq)

    # Resample to desired interval, but also count the number of bars in each interval
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'count': 'count'
    }
    df['count'] = 1
    df_resampled = df.resample(interval).agg(ohlc_dict).dropna().reset_index()

    # Only keep rows where count == bars_needed (fully formed candles)
    df_resampled = df_resampled[df_resampled['count'] == bars_needed].reset_index(drop=True)

    # Calculate Heiken Ashi
    ha_close = (df_resampled['open'] + df_resampled['high'] + df_resampled['low'] + df_resampled['close']) / 4
    ha_open = [(df_resampled['open'].iloc[0] + df_resampled['close'].iloc[0]) / 2]
    for i in range(1, len(df_resampled)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)
    ha_open = pd.Series(ha_open)

    ha_high = pd.concat([ha_open, ha_close, df_resampled['high']], axis=1).max(axis=1)
    ha_low = pd.concat([ha_open, ha_close, df_resampled['low']], axis=1).min(axis=1)

    # Overwrite columns as requested and round to 2 decimals
    df_resampled['open'] = ha_open.round(2)
    df_resampled['high'] = ha_high.round(2)
    df_resampled['low'] = ha_low.round(2)
    df_resampled['close'] = ha_close.round(2)

    # Add color and body_height columns, rounded
    df_resampled['color'] = np.where(df_resampled['close'] > df_resampled['open'], 'green', 'red')
    df_resampled['body_height'] = (df_resampled['close'] - df_resampled['open']).abs().round(2)

    # Drop the 'count' column before returning
    return df_resampled.drop(columns=['count'])


def heiken_ashi(conn, exchange: str, trading_symbol: str, start: datetime, end: datetime, interval = '3min') -> pd.DataFrame:
    """
    Convert OHLC data to Heiken Ashi candles with optional resampling. 
    Returns the running candle

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['datetime', 'open', 'high', 'low', 'close']
    interval (str): Pandas resample rule (e.g., '5min', '15min', '1H', '1D')

    Returns:
    pd.DataFrame: DataFrame with Heiken Ashi columns ['open', 'high', 'low', 'close', 'color', 'body_height']
    """
    df = edge.fetch_historical_data(conn, exchange, trading_symbol, start, end, 'min')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Resample to desired interval
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    df_resampled = df.resample(interval).agg(ohlc_dict).dropna().reset_index()

    # Calculate Heiken Ashi
    ha_close = (df_resampled['open'] + df_resampled['high'] + df_resampled['low'] + df_resampled['close']) / 4
    ha_open = [(df_resampled['open'].iloc[0] + df_resampled['close'].iloc[0]) / 2]
    for i in range(1, len(df_resampled)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)
    ha_open = pd.Series(ha_open)

    ha_high = pd.concat([ha_open, ha_close, df_resampled['high']], axis=1).max(axis=1)
    ha_low = pd.concat([ha_open, ha_close, df_resampled['low']], axis=1).min(axis=1)

    # Overwrite columns as requested
    df_resampled['open'] = ha_open.round(2)
    df_resampled['high'] = ha_high.round(2)
    df_resampled['low'] = ha_low.round(2)
    df_resampled['close'] = ha_close.round(2)

    # Add color and body_height columns
    df_resampled['color'] = np.where(df_resampled['close'] > df_resampled['open'], 'green', 'red')
    df_resampled['body_height'] = (df_resampled['close'] - df_resampled['open']).abs().round(2)

    return df_resampled

def rsi(data, period=14, sma=20):
    """
    Calculate the Relative Strength Index (RSI) for OHLC data using Wilder's method.
    
    Parameters:
    data (DataFrame): DataFrame containing OHLC data with columns ['Open', 'High', 'Low', 'Close'].
    period (int): Period for calculating RSI. Default is 14.
    
    Returns:
    DataFrame: DataFrame containing RSI values.
    """
    data = data.copy()
    data['delta'] = data['close'].diff()
    
    gain = data['delta'].where(data['delta'] > 0, 0)
    loss = -data['delta'].where(data['delta'] < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    for i in range(period, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    data['rsi'] = data['rsi'].round(2)
    # Apply SMA to RSI
    data['rsi_sma'] = data['rsi'].rolling(window=sma, min_periods=1).mean()
    data['rsi_sma'] = data['rsi_sma'].round(2)

    data.drop(['delta'], axis='columns', inplace=True)
    
    return data


def rsi_avg(data, period=14, ema=9):
    """
    Calculate the Relative Strength Index (RSI) for OHLC data using Wilder's method.
    
    Parameters:
    data (DataFrame): DataFrame containing OHLC data with columns ['Open', 'High', 'Low', 'Close'].
    period (int): Period for calculating RSI. Default is 14.
    
    Returns:
    DataFrame: DataFrame containing RSI values.
    """
    data = data.copy()
    data['avg_close'] = (data['high'] + data['low']) / 2
    data['delta'] = data['avg_close'].diff()
    
    gain = data['delta'].where(data['delta'] > 0, 0)
    loss = -data['delta'].where(data['delta'] < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    for i in range(period, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    data['rsi'] = data['rsi'].round(2)
    # Apply EMA to RSI
    data['rsi_ema'] = data['rsi'].ewm(span=ema, adjust=False).mean().round(2)

    data.drop(['delta','avg_close'], axis='columns', inplace=True)
    
    return data

def bullish_reversal(brick_size = .1, value=10.0, reversal = 3):
    while reversal > 0:
        value = value + up_step(brick_size, value)
        reversal = reversal - 1
    return round(value,2)

def bearish_reversal(brick_size = .1, value=10.0, reversal = 3):
    while reversal > 0:
        value = value - down_step(brick_size, value)
        reversal = reversal - 1
    return round(value,2)


def detect_double_top_buy(df: pd.DataFrame) -> pd.DataFrame:
    double_top_buys = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['color'] == 'green' and df.iloc[i]['color'] == 'green':
            if df.iloc[i]['high'] > df.iloc[i-2]['high']:
                double_top_buys.append(i)
    return double_top_buys

def detect_double_bottom_sell(df: pd.DataFrame) -> pd.DataFrame:
    double_bottom_sells = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['color'] == 'red' and df.iloc[i]['color'] == 'red':
            if df.iloc[i]['low'] < df.iloc[i-2]['low']:
                double_bottom_sells.append(i)
    return double_bottom_sells

def detect_bull_trap(df: pd.DataFrame) -> pd.DataFrame:
    bull_traps = []
    for i in range(3, len(df)):
        if (df.iloc[i-3]['color'] == 'green' and df.iloc[i-1]['color'] == 'green' and
            df.iloc[i-2]['color'] == 'red' and df.iloc[i]['color'] == 'red'):
            if (df.iloc[i-1]['high'] > df.iloc[i-3]['high'] and df.iloc[i]['low'] < df.iloc[i-2]['low']):
                bull_traps.append(i)
    return bull_traps

def detect_bear_trap(df: pd.DataFrame) -> pd.DataFrame:
    bear_traps = []
    for i in range(3, len(df)):
        if (df.iloc[i-3]['color'] == 'red' and df.iloc[i-1]['color'] == 'red' and
            df.iloc[i-2]['color'] == 'green' and df.iloc[i]['color'] == 'green'):
            if (df.iloc[i-1]['low'] < df.iloc[i-3]['low'] and df.iloc[i]['high'] > df.iloc[i-2]['high']):
                bear_traps.append(i)
    return bear_traps


def detect_high_pole(df: pd.DataFrame, brick_size = 0.05) -> pd.DataFrame:
    high_poles = []
    for i in range(3, len(df)):
        # Check for double top buy pattern
        if df.iloc[i-3]['color'] == 'green' and df.iloc[i-1]['color'] == 'green' and df.iloc[i-2]['color'] == 'red':
            if df.iloc[i-1]['high'] > df.iloc[i-3]['high']:
                # Check for more than 5 boxes after double top buy pattern
                if round(((df.iloc[i-1]['high'] - df.iloc[i-3]['high'])/df.iloc[i-3]['high']) * 100,2) >= brick_size * 5:
                    # Check for more than 50% retracement of the entire previous column of X
                    if df.iloc[i]['count'] >= df.iloc[i-1]['count'] / 2:
                        high_poles.append(i-1)
    return high_poles


def detect_low_pole(df: pd.DataFrame, brick_size = 0.05) -> pd.DataFrame:
    low_poles = []
    for i in range(3, len(df)):
        # Check for double bottom sell pattern
        if df.iloc[i-3]['color'] == 'red' and df.iloc[i-1]['color'] == 'red' and df.iloc[i-2]['color'] == 'green':
            if df.iloc[i-1]['low'] < df.iloc[i-3]['low']:
                # Check for more than 5 boxes after double bottom sell pattern
                if round(((df.iloc[i-3]['low'] - df.iloc[i-1]['low'])/df.iloc[i-3]['low']) * 100,2) >= brick_size * 5:
                    # Check for more than 50% retracement of the entire previous column of O
                    if df.iloc[i]['count'] >= df.iloc[i-1]['count'] / 2:
                        low_poles.append(i-1)
    return low_poles


def detect_bearish_turtle_breakout(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    bearish_turtle_breakouts = []

    # Identify bearish (red) columns
    red_indices = df.index[df['color'] == 'red'].tolist()

    # Iterate through red columns to find breakouts
    previous_breakout_index = -1

    for i in range(n, len(red_indices)):
        current_index = red_indices[i]

        # Skip if the previous breakout was the last red column
        if previous_breakout_index == red_indices[i - 1]:
            continue

        # Get the last 'n-1' red column lows
        last_n_lows = df.loc[red_indices[i - (n - 1): i], 'low']

        # Check if the current red column's low is below the lowest of the last 'n-1' lows
        if df.at[current_index, 'low'] < last_n_lows.min():
            bearish_turtle_breakouts.append(current_index)
            previous_breakout_index = current_index

    return bearish_turtle_breakouts


def detect_bullish_turtle_breakout(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    bullish_turtle_breakouts = []

    # Identify bullish (green) columns
    green_indices = df.index[df['color'] == 'green'].tolist()

    # Iterate through green columns to find breakouts
    previous_breakout_index = -1

    for i in range(n, len(green_indices)):
        current_index = green_indices[i]

        # Skip if the previous breakout was the last green column
        if previous_breakout_index == green_indices[i - 1]:
            continue

        # Get the last 'n-1' green column highs
        last_n_highs = df.loc[green_indices[i - (n - 1): i], 'high']

        # Check if the current green column's high is above the highest of the last 'n-1' highs
        if df.at[current_index, 'high'] > last_n_highs.max():
            bullish_turtle_breakouts.append(current_index)
            previous_breakout_index = current_index

    return bullish_turtle_breakouts



def convert_to_pnf(brick_size = .1, df: pd.DataFrame = pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['datetime'])
    multiplier = 1
    if df['close'].min() < 10:
        multiplier = 10
        df['close'] = df['close'] * multiplier
        df['open'] = df['open'] * multiplier

    initial_point = round(df['open'].iloc[0],2)
    pnf = []
    dic =  df.to_dict(orient='records')
    temp_coloumn = {
                'datetime': df['datetime'].iloc[0],
                'color' : "",
                'low' : initial_point,
                'high' : initial_point,
                'count' : 0
    }
    for row in dic:
        if not pnf:
            if temp_coloumn['high'] == temp_coloumn['low'] and row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                    temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                    temp_coloumn['color'] = "green"
                    temp_coloumn['datetime'] = row['datetime']
                    temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['high'] == temp_coloumn['low'] and row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                    temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                    temp_coloumn['color'] = "red"
                    temp_coloumn['datetime'] = row['datetime']
                    temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['color'] == "green":
                if row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] <= bearish_reversal(brick_size, temp_coloumn['high'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['high'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    temp_coloumn['color'] = 'red'
                    temp_coloumn['low'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    temp_coloumn['count'] = 2
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['color'] == "red":
                if row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['color'] = "red"
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] >= bullish_reversal(brick_size, temp_coloumn['low'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['count'] = 2
                    temp_coloumn['low'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    temp_coloumn['color'] = 'green'
                    temp_coloumn['high'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
        else:
            if temp_coloumn['color'] == "red":
                if row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['color'] = "red"
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] >= bullish_reversal(brick_size, temp_coloumn['low'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['count'] = 2
                    temp_coloumn['low'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    temp_coloumn['color'] = 'green'
                    temp_coloumn['high'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['color'] == "green":
                if row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] <= bearish_reversal(brick_size, temp_coloumn['high'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['count'] = 2
                    temp_coloumn['high'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    temp_coloumn['color'] = 'red'
                    temp_coloumn['low'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
            
    pnf.append(copy.deepcopy(temp_coloumn))
    df = pd.DataFrame(pnf)
    if multiplier > 1:
        df['high'] = round((df['high'] / multiplier),2)
        df['low'] = round((df['low'] / multiplier),2)

    df['close'] = np.where(df['color'] == 'red', df['low'], df['high'])

    # Detect patterns
    df['double_top_buy'] = False
    df['double_bottom_sell'] = False
    df['bull_trap'] = False
    df['bear_trap'] = False
    df['high_pole'] = False
    df['low_pole'] = False
    df['bullish_turtle'] = False
    df['bearish_turtle'] = False

    dtb_indices = detect_double_top_buy(df)
    for idx in dtb_indices:
        df.at[idx, 'double_top_buy'] = True

    dbs_indices = detect_double_bottom_sell(df)
    for idx in dbs_indices:
        df.at[idx, 'double_bottom_sell'] = True

    bull_trap_indices = detect_bull_trap(df)
    for idx in bull_trap_indices:
        df.at[idx, 'bull_trap'] = True

    bear_trap_indices = detect_bear_trap(df)
    for idx in bear_trap_indices:
        df.at[idx, 'bear_trap'] = True

    high_pole_indices = detect_high_pole(df, brick_size)
    for idx in high_pole_indices:
        df.at[idx, 'high_pole'] = True

    low_pole_indices = detect_low_pole(df, brick_size)
    for idx in low_pole_indices:
        df.at[idx, 'low_pole'] = True

    bullish_turtle_indices = detect_bullish_turtle_breakout(df, 5)
    for idx in bullish_turtle_indices:
        df.at[idx, 'bullish_turtle'] = True

    bearish_turtle_indices = detect_bearish_turtle_breakout(df, 5)
    for idx in bearish_turtle_indices:
        df.at[idx, 'bearish_turtle'] = True

    return df


def pnf(conn, exchange: str, trading_symbol: str, start: datetime, end: datetime, interval = 'min', brick_size = .1, last_high = 0, last_low=0, initial_color = "") -> pd.DataFrame:
    df = edge.fetch_historical_data(conn, exchange, trading_symbol, start, end, interval)
    df['datetime'] = pd.to_datetime(df['datetime'])
    multiplier = 1
    if df['close'].min() < 10:
        multiplier = 10
        df['close'] = df['close'] * multiplier
        df['open'] = df['open'] * multiplier
    
    if last_high == 0 and last_low == 0:
        low = round(df['open'].iloc[0],2)
        high = round(df['open'].iloc[0],2)
    else:
        low = last_low
        high = last_high
    #initial_point = round(df['open'].iloc[0],2)
    pnf = []
    dic =  df.to_dict(orient='records')
    temp_coloumn = {
                'datetime': df['datetime'].iloc[0],
                'color' : initial_color,
                'low' : low,
                'high' : high,
                'count' : 0
    }
    for row in dic:
        if not pnf:
            if temp_coloumn['high'] == temp_coloumn['low'] and row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                    temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                    temp_coloumn['color'] = "green"
                    temp_coloumn['datetime'] = row['datetime']
                    temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['high'] == temp_coloumn['low'] and row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                    temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                    temp_coloumn['color'] = "red"
                    temp_coloumn['datetime'] = row['datetime']
                    temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['color'] == "green":
                if row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] <= bearish_reversal(brick_size, temp_coloumn['high'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['count'] = 2
                    temp_coloumn['high'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    temp_coloumn['color'] = 'red'
                    temp_coloumn['low'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['color'] == "red":
                if row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['color'] = "red"
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] >= bullish_reversal(brick_size, temp_coloumn['low'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['count'] = 2
                    temp_coloumn['low'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    temp_coloumn['color'] = 'green'
                    temp_coloumn['high'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
        else:
            if temp_coloumn['color'] == "red":
                if row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['color'] = "red"
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] >= bullish_reversal(brick_size, temp_coloumn['low'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['count'] = 2
                    temp_coloumn['low'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    temp_coloumn['color'] = 'green'
                    temp_coloumn['high'] = round((temp_coloumn['low'] + up_step(brick_size, temp_coloumn['low'])),2)
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
            elif temp_coloumn['color'] == "green":
                if row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                    while row['close'] >= temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high']):
                        temp_coloumn['high'] = round((temp_coloumn['high'] + up_step(brick_size, temp_coloumn['high'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
                elif row['close'] <= bearish_reversal(brick_size, temp_coloumn['high'], 3):
                    pnf.append(copy.deepcopy(temp_coloumn))
                    temp_coloumn['count'] = 2
                    temp_coloumn['high'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    temp_coloumn['color'] = 'red'
                    temp_coloumn['low'] = round((temp_coloumn['high'] - down_step(brick_size, temp_coloumn['high'])),2)
                    while row['close'] <= temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low']):
                        temp_coloumn['low'] = round((temp_coloumn['low'] - down_step(brick_size, temp_coloumn['low'])),2)
                        temp_coloumn['datetime'] = row['datetime']
                        temp_coloumn['count'] = temp_coloumn['count'] + 1
            
    pnf.append(copy.deepcopy(temp_coloumn))
    df = pd.DataFrame(pnf)
    if multiplier > 1:
        df['high'] = round((df['high'] / multiplier),2)
        df['low'] = round((df['low'] / multiplier),2)
    df['close'] = np.where(df['color'] == 'red', df['low'], df['high'])
    # Detect patterns
    df['double_top_buy'] = False
    df['double_bottom_sell'] = False
    df['bull_trap'] = False
    df['bear_trap'] = False
    df['high_pole'] = False
    df['low_pole'] = False
    df['bullish_turtle'] = False
    df['bearish_turtle'] = False

    dtb_indices = detect_double_top_buy(df)
    for idx in dtb_indices:
        df.at[idx, 'double_top_buy'] = True

    dbs_indices = detect_double_bottom_sell(df)
    for idx in dbs_indices:
        df.at[idx, 'double_bottom_sell'] = True

    bull_trap_indices = detect_bull_trap(df)
    for idx in bull_trap_indices:
        df.at[idx, 'bull_trap'] = True

    bear_trap_indices = detect_bear_trap(df)
    for idx in bear_trap_indices:
        df.at[idx, 'bear_trap'] = True

    high_pole_indices = detect_high_pole(df, brick_size)
    for idx in high_pole_indices:
        df.at[idx, 'high_pole'] = True

    low_pole_indices = detect_low_pole(df, brick_size)
    for idx in low_pole_indices:
        df.at[idx, 'low_pole'] = True

    bullish_turtle_indices = detect_bullish_turtle_breakout(df, 5)
    for idx in bullish_turtle_indices:
        df.at[idx, 'bullish_turtle'] = True

    bearish_turtle_indices = detect_bearish_turtle_breakout(df, 5)
    for idx in bearish_turtle_indices:
        df.at[idx, 'bearish_turtle'] = True
        
    return df


def xo_zone(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Calculate the XO Zone indicator for the last n columns of a PnF chart.

    Parameters:
    df (pd.DataFrame): The PnF DataFrame.
    n (int): The number of columns to consider.

    Returns:
    pd.DataFrame: The original DataFrame with an additional 'xo_zone' column.
    """
    # Initialize the xo_zone column with None
    df['xo_zone'] = None

    for i in range(n-1, len(df)):
        last_n_columns = df.iloc[i-n+1:i+1]
        
        count = 0
        
        for _, row in last_n_columns.iterrows():
            if row['color'] == 'green':
                count += row['count']
            elif row['color'] == 'red':
                count -= row['count']
        
        df.at[i, 'xo_zone'] = count
    return df

        


        

