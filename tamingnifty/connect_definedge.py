from integrate import ConnectToIntegrate, IntegrateData, IntegrateOrders
import pandas as pd
import pyotp
from datetime import datetime, timedelta
from dateutil import parser
import requests
import zipfile
import io
from retry import retry
import os
from dotenv import (  # pip install python-dotenv
    find_dotenv,
    load_dotenv,
    set_key,
)

@retry(tries=5, delay=5, backoff=2)
def login_to_integrate(fresh = False) -> ConnectToIntegrate:
    """
    Login to Integrate and return the connection object.
    """
    dotenv_file: str = find_dotenv()
    load_dotenv(dotenv_file)
    try:
        # Get the API token and secret from environment variables.
        api_token: str = os.environ["INTEGRATE_API_TOKEN"]
        api_secret: str = os.environ["INTEGRATE_API_SECRET"]
        totp: str = os.environ["TOTP"]
    except KeyError:
        raise KeyError(
            "Please set INTEGRATE_API_TOKEN and INTEGRATE_API_SECRET in .env file."
        )

    # Initialise the connection.
    conn = ConnectToIntegrate()
    try:
        if fresh == True:
            raise KeyError("Getting Fresh Token..")
        uid: str = os.environ["INTEGRATE_UID"]
        actid: str = os.environ["INTEGRATE_ACTID"]
        api_session_key: str = os.environ["INTEGRATE_API_SESSION_KEY"]
        ws_session_key: str = os.environ["INTEGRATE_WS_SESSION_KEY"]
        conn.set_session_keys(uid, actid, api_session_key, ws_session_key)
        # Please note that the session keys are valid for 24 hours. After that you
        # will have to login again. The logic to handle this is left to the user.
    except KeyError:
        totp = pyotp.TOTP(totp).now()
        conn.login(
            api_token=api_token,
            api_secret=api_secret,
            totp=totp
        )
        (uid, actid, api_session_key, ws_session_key) = conn.get_session_keys()
        os.environ["INTEGRATE_UID"] = uid
        os.environ["INTEGRATE_ACTID"] = actid
        os.environ["INTEGRATE_API_SESSION_KEY"] = api_session_key
        os.environ["INTEGRATE_WS_SESSION_KEY"] = ws_session_key
        print("Login successful.")
    return conn

@retry(tries=5, delay=5, backoff=2)
def fetch_historical_data(conn: ConnectToIntegrate, exchange: str, trading_symbol: str, start: datetime, end: datetime, interval = 'min') -> pd.DataFrame:
    """
    Fetch historical data and return as a pandas DataFrame.
    """
    if interval == 'day':
        tf = conn.TIMEFRAME_TYPE_DAY
    elif interval == 'min':
        tf = conn.TIMEFRAME_TYPE_MIN

    ic = IntegrateData(conn)
    history = ic.historical_data(
        exchange=exchange,
        trading_symbol=trading_symbol,
        timeframe=tf,  # Use the specific timeframe value
        start=start,
        end=end,
    )
    df = pd.DataFrame(list(history))  # Ensure conversion to list if generator
    return df

@retry(tries=5, delay=5, backoff=2)
def fetch_ltp(conn: ConnectToIntegrate, exchange: str, trading_symbol: str):
    try:
        #ic = IntegrateData(conn)
        # quote = ic.quotes(exchange=exchange, trading_symbol=trading_symbol)
        # return float(quote['ltp'])
        quote = fetch_historical_data(conn, exchange, trading_symbol, datetime.today() - timedelta(days=5), datetime.today(), 'min')
        return round(quote['close'].iloc[-1],2)
    except Exception as e:
        print(f"Exception encountered: {e}. Retrying...")
    #     conn = login_to_integrate(fresh = True)
    #     ic = IntegrateData(conn)
    #     quote = ic.quotes(exchange=exchange, trading_symbol=trading_symbol)
    #     return float(quote['ltp'])



@retry(tries=5, delay=5, backoff=2)
def get_option_price(exchange: str , trading_symbol: str, start: datetime, end: datetime, interval = 'min'):
    conn = login_to_integrate()
    if interval == 'day':
        tf = conn.TIMEFRAME_TYPE_DAY
    elif interval == 'min':
        tf = conn.TIMEFRAME_TYPE_MIN

    ic = IntegrateData(conn)
    history = ic.historical_data(
        exchange=exchange,
        trading_symbol=trading_symbol,
        timeframe=tf,  # Use the specific timeframe value
        start=start,
        end=end,
    )
    df = pd.DataFrame(list(history))  # Ensure conversion to list if generator
    return df['close'].iloc[-1]  

@retry(tries=5, delay=5, backoff=2)
def get_index_future(url='https://app.definedgesecurities.com/public/allmaster.zip', instrument_name = "NIFTY"):
    current_date = datetime.now()
    column_names = ['SEGMENT', 'TOKEN', 'SYMBOL', 'TRADINGSYM', 'INSTRUMENT TYPE', 'EXPIRY', 'TICKSIZE', 'LOTSIZE', 'OPTIONTYPE', 'STRIKE', 'PRICEPREC', 'MULTIPLIER', 'ISIN', 'PRICEMULT', 'UnKnown']
    # Send a GET request to download the zip file
    response = requests.get(url)
    response.raise_for_status()  # This will raise an exception for HTTP errors
    # Open the zip file from the bytes-like object
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        # Extract the name of the first CSV file in the zip archive
        csv_name = thezip.namelist()[0]
        # Extract and read the CSV file into a pandas DataFrame
        with thezip.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file, header=None, names=column_names, low_memory=False, on_bad_lines='skip')
    df = df[(df['SEGMENT'] == 'NFO') & (df['INSTRUMENT TYPE'] == 'FUTIDX')]
    df = df[(df['SYMBOL'] == instrument_name)]
    df['EXPIRY'] = pd.to_datetime(df['EXPIRY'], format='%d%m%Y', errors='coerce')
    df = df.sort_values(by='EXPIRY', ascending=True)
    df= df[df['EXPIRY'] > current_date]
    # Return the loaded DataFrame
    return df.iloc[0]['TRADINGSYM']

@retry(tries=5, delay=5, backoff=2)
def load_csv_from_zip(url='https://app.definedgesecurities.com/public/allmaster.zip', instrument_name = "NIFTY"):
    column_names = ['SEGMENT', 'TOKEN', 'SYMBOL', 'TRADINGSYM', 'INSTRUMENT TYPE', 'EXPIRY', 'TICKSIZE', 'LOTSIZE', 'OPTIONTYPE', 'STRIKE', 'PRICEPREC', 'MULTIPLIER', 'ISIN', 'PRICEMULT', 'UnKnown']
    # Send a GET request to download the zip file
    response = requests.get(url)
    response.raise_for_status()  # This will raise an exception for HTTP errors
    # Open the zip file from the bytes-like object
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        # Extract the name of the first CSV file in the zip archive
        csv_name = thezip.namelist()[0]
        # Extract and read the CSV file into a pandas DataFrame
        with thezip.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file, header=None, names=column_names, low_memory=False, on_bad_lines='skip')
    df = df[(df['SEGMENT'] == 'NFO') & (df['INSTRUMENT TYPE'] == 'OPTIDX')]
    df = df[(df['SYMBOL'].str.startswith(instrument_name))]
    df = df[df['SYMBOL'] == instrument_name]
    df['EXPIRY'] = df['EXPIRY'].astype(str).apply(lambda x: x.zfill(8))
    df['EXPIRY'] = pd.to_datetime(df['EXPIRY'], format='%d%m%Y', errors='coerce')
    df = df.sort_values(by='EXPIRY', ascending=True)
    # Return the loaded DataFrame
    return df


@retry(tries=5, delay=5, backoff=2)
def get_index_option_symbol(strike=19950, option_type = "PE" ):
    df = load_csv_from_zip(instrument_name = "NIFTY")
    df = df[df['TRADINGSYM'].str.contains(str(strike))]
    df = df[df['OPTIONTYPE'].str.match(option_type)]
    # Get the current date
    current_date = datetime.now()
    # Calculate the start and end dates of the current week
    df= df[(df['EXPIRY'] > (current_date + timedelta(days=0)))]
    df = df.head(1)
    print("Getting options Symbol...")
    print(f"Symbol: {df['TRADINGSYM'].values[0]} , Expiry: {df['EXPIRY'].values[0]}")
    return df['TRADINGSYM'].values[0], parser.parse(str(df['EXPIRY'].values[0])).date()