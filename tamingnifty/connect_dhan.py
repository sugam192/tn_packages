"""
Dhan (DhanHQ v2) connection layer for tamingnifty.

This mirrors connect_definedge.py function for function, so a strategy bot can move
brokers by changing a single import line:

    from tamingnifty import connect_definedge as edge   # old
    from tamingnifty import connect_dhan as edge        # new

connect_definedge.py is deliberately left untouched in case we ever move back.

There is no Dhan SDK here on purpose - everything is plain `requests` calls against
the documented REST endpoints, so what you read in this file matches what you read
in the Dhan docs when something breaks at 9:16 in the morning.

Facts below were verified against the live API and the real scrip master on
2026-09-25. Where a number looks arbitrary, the comment explains where it came from.
"""

import pandas as pd
import pyotp
import requests
import time
from datetime import datetime, timedelta
from retry import retry
import os
from dotenv import (  # pip install python-dotenv
    find_dotenv,
    load_dotenv,
)

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

API_BASE = "https://api.dhan.co/v2"
AUTH_URL = "https://auth.dhan.co/app/generateAccessToken"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

# Dhan keys every instrument by a numeric security id, not by a symbol string like
# Definedge did. For the three indices we care about the ids are fixed, so we just
# hardcode them instead of searching the 198,000-row scrip master every time.
INDEX_IDS = {
    "Nifty 50": 13,
    "Nifty Bank": 25,
    "India VIX": 21,
}

# Dhan rejects any intraday request wider than 90 days with error DH-905. We ask for
# 80 at a time to leave a margin, and fetch_historical_data() loops to cover the rest.
MAX_DAYS_PER_REQUEST = 80

# Dhan returns candle timestamps as true UTC epoch seconds. Converting them through
# UTC and then to Asia/Kolkata lands the first candle of the session exactly on
# 09:15:00, which is how this was verified. We then drop the timezone, because the
# bots compare these values against a plain datetime.today() and pandas raises a
# TypeError if you compare timezone-aware against timezone-naive.
IST = "Asia/Kolkata"

# NIFTY options: 65 per lot, and the exchange rejects any single order above 1756.
# Both read straight off the scrip master.
NIFTY_LOT_SIZE = 65
NIFTY_FREEZE_QTY = 1756

# Filled in by load_instruments() the first time it is needed, then reused for the
# life of the process so we do not download a 30 MB file inside the trading loop.
_instruments = None


# --------------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------------

@retry(tries=5, delay=5, backoff=2)
def login_to_dhan(fresh=False):
    """
    Log in to Dhan and return a connection dict:

        {"client_id": "1100341129", "access_token": "eyJ0eXAi..."}

    Every other function in this file takes that dict as its first argument, the same
    way the Definedge functions take a ConnectToIntegrate object.

    An access token is valid for 24 hours. If DHAN_ACCESS_TOKEN is already set in the
    environment we reuse it; otherwise we mint a fresh one using the TOTP secret.
    Pass fresh=True to force a new token.
    """
    dotenv_file = find_dotenv()
    load_dotenv(dotenv_file)

    try:
        client_id = os.environ["DHAN_CLIENT_ID"]
        pin = os.environ["DHAN_PIN"]
        totp_secret = os.environ["DHAN_TOTP"]
    except KeyError:
        raise KeyError(
            "Please set DHAN_CLIENT_ID, DHAN_PIN and DHAN_TOTP in the .env file."
        )

    # Reuse the token we already have, unless the caller asked for a fresh one.
    if fresh == False and os.environ.get("DHAN_ACCESS_TOKEN"):
        return {
            "client_id": client_id,
            "access_token": os.environ["DHAN_ACCESS_TOKEN"],
        }

    # Mint a new 24 hour token. TOTP must be enabled on the Dhan account for this
    # endpoint to work - without it there is no way to log in without a browser.
    totp_now = pyotp.TOTP(totp_secret).now()
    response = requests.post(
        AUTH_URL,
        params={"dhanClientId": client_id, "pin": pin, "totp": totp_now},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if "accessToken" not in data:
        raise Exception(f"Dhan login failed: {data}")

    os.environ["DHAN_ACCESS_TOKEN"] = data["accessToken"]
    print(f"Login successful. Token valid till {data.get('expiryTime')}")

    return {"client_id": client_id, "access_token": data["accessToken"]}


def build_headers(conn, include_client_id=False):
    """
    Build the request headers. Most endpoints only need the access token, but the
    market feed (LTP) endpoint also wants the client id, hence the flag.
    """
    headers = {
        "access-token": conn["access_token"],
        "Content-Type": "application/json",
    }
    if include_client_id == True:
        headers["client-id"] = conn["client_id"]
    return headers


# --------------------------------------------------------------------------------
# Instrument lookup
# --------------------------------------------------------------------------------

def get_security_id(trading_symbol):
    """
    Turn an index name like "Nifty 50" into its Dhan security id.
    """
    if trading_symbol not in INDEX_IDS:
        raise KeyError(
            f"Unknown index '{trading_symbol}'. Known indices: {list(INDEX_IDS.keys())}"
        )
    return INDEX_IDS[trading_symbol]


@retry(tries=5, delay=5, backoff=2)
def load_instruments():
    """
    Download Dhan's scrip master once and keep it in memory.

    This replaces Definedge's allmaster.zip. It is a plain CSV of about 198,000 rows,
    so we download it at most once per run of the bot.
    """
    global _instruments
    if _instruments is None:
        print("Downloading Dhan scrip master...")
        _instruments = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)
        print(f"Scrip master loaded: {len(_instruments)} rows")
    return _instruments


@retry(tries=5, delay=5, backoff=2)
def get_index_option_symbol(strike, option_type, instrument_name="NIFTY", min_dte=3):
    """
    Find the option contract for a given strike and type, using the nearest expiry
    that is more than `min_dte` days away.

    Returns (trading_symbol, security_id, expiry_date, lot_size).

    The min_dte=3 default matches the old Definedge behaviour, which skipped any
    contract expiring within 3 days.
    """
    df = load_instruments()

    df = df[
        (df["UNDERLYING_SYMBOL"] == instrument_name)
        & (df["INSTRUMENT"] == "OPTIDX")
        & (df["OPTION_TYPE"] == option_type)
        & (df["STRIKE_PRICE"] == float(strike))
    ]

    if len(df) == 0:
        raise Exception(f"No {instrument_name} {strike} {option_type} contract found.")

    # SM_EXPIRY_DATE comes through as a YYYY-MM-DD string.
    df = df.copy()
    df["EXPIRY"] = pd.to_datetime(df["SM_EXPIRY_DATE"], errors="coerce")
    cutoff = datetime.now() + timedelta(days=min_dte)
    df = df[df["EXPIRY"] > cutoff]
    df = df.sort_values(by="EXPIRY", ascending=True)

    if len(df) == 0:
        raise Exception(
            f"No {instrument_name} {strike} {option_type} contract expiring after {cutoff.date()}."
        )

    row = df.iloc[0]
    trading_symbol = row["DISPLAY_NAME"]
    security_id = int(row["SECURITY_ID"])
    expiry = row["EXPIRY"].date()
    lot_size = int(row["LOT_SIZE"])

    print("Getting options Symbol...")
    print(f"Symbol: {trading_symbol} , Security Id: {security_id} , Expiry: {expiry}")
    return trading_symbol, security_id, expiry, lot_size


# --------------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------------

def candles_to_dataframe(data):
    """
    Dhan returns candles as parallel lists rather than a list of rows:

        {"open": [...], "high": [...], "low": [...],
         "close": [...], "volume": [...], "timestamp": [...]}

    Convert that into the same DataFrame shape the Definedge layer produced, so
    ta.py does not need to change: columns datetime, open, high, low, close, volume.
    """
    if not data or not data.get("timestamp"):
        return pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        {
            "datetime": data["timestamp"],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
        }
    )

    # Epoch seconds (UTC) -> IST -> drop the timezone. See the IST note at the top.
    df["datetime"] = (
        pd.to_datetime(df["datetime"], unit="s", utc=True)
        .dt.tz_convert(IST)
        .dt.tz_localize(None)
    )
    return df


@retry(tries=5, delay=5, backoff=2)
def fetch_candles(conn, security_id, exchange_segment, instrument, start, end, interval="min"):
    """
    Fetch candles for one security over a date range that is already known to be
    within Dhan's 90 day limit. Most callers want fetch_historical_data() instead,
    which handles the chunking.
    """
    body = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "oi": False,
        "fromDate": start.strftime("%Y-%m-%d"),
        "toDate": end.strftime("%Y-%m-%d"),
    }

    if interval == "day":
        url = f"{API_BASE}/charts/historical"
    else:
        url = f"{API_BASE}/charts/intraday"
        body["interval"] = "1"

    response = requests.post(url, headers=build_headers(conn), json=body, timeout=60)
    response.raise_for_status()
    return candles_to_dataframe(response.json())


@retry(tries=5, delay=5, backoff=2)
def fetch_historical_data(conn, exchange, trading_symbol, start, end, interval="min"):
    """
    Fetch historical candles for an index and return a DataFrame with columns
    datetime, open, high, low, close, volume.

    The signature deliberately matches connect_definedge.fetch_historical_data so
    that ta.py works unchanged. `exchange` is accepted and ignored - Dhan works out
    the venue from the segment, but keeping the argument means no call site changes.

    Intraday requests are split into MAX_DAYS_PER_REQUEST windows and stitched back
    together, because Dhan refuses anything wider than 90 days. This matters after
    the bot has been down for a while: the signal doc can hold a start_date months
    back, and without chunking that request would simply fail.
    """
    security_id = get_security_id(trading_symbol)

    # Daily candles have no 90 day limit, so ask for the whole range in one go.
    if interval == "day":
        return fetch_candles(conn, security_id, "IDX_I", "INDEX", start, end, "day")

    frames = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST)
        if chunk_end > end:
            chunk_end = end

        part = fetch_candles(
            conn, security_id, "IDX_I", "INDEX", chunk_start, chunk_end, "min"
        )
        if len(part) > 0:
            frames.append(part)

        # Move to the day after this chunk so the windows do not overlap.
        chunk_start = chunk_end + timedelta(days=1)

    if len(frames) == 0:
        return pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )

    df = pd.concat(frames, ignore_index=True)
    # Belt and braces: if two chunks ever do overlap, keep one copy of each candle.
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    df = df.reset_index(drop=True)
    return df


@retry(tries=5, delay=5, backoff=2)
def fetch_ltp(conn, exchange, trading_symbol):
    """
    Last traded price for an index, for example "India VIX".

    `exchange` is accepted and ignored, to match the Definedge signature.
    """
    security_id = get_security_id(trading_symbol)
    response = requests.post(
        f"{API_BASE}/marketfeed/ltp",
        headers=build_headers(conn, include_client_id=True),
        json={"IDX_I": [security_id]},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    price = data["data"]["IDX_I"][str(security_id)]["last_price"]
    return round(float(price), 2)


@retry(tries=5, delay=5, backoff=2)
def get_option_ltp(conn, security_id):
    """
    Last traded price for a single option contract. Used by the running PnL loop.
    """
    response = requests.post(
        f"{API_BASE}/marketfeed/ltp",
        headers=build_headers(conn, include_client_id=True),
        json={"NSE_FNO": [int(security_id)]},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    price = data["data"]["NSE_FNO"][str(security_id)]["last_price"]
    return round(float(price), 2)


@retry(tries=5, delay=5, backoff=2)
def get_option_price(conn, security_id, start, end, interval="min"):
    """
    Closing price of the most recent candle for an option contract.

    This is what the simulated (live_trading=false) order path uses as a fill price,
    the same way the Definedge version did.
    """
    df = fetch_candles(conn, security_id, "NSE_FNO", "OPTIDX", start, end, interval)
    if len(df) == 0:
        raise Exception(f"No price data returned for security id {security_id}.")
    return round(float(df["close"].iloc[-1]), 2)


# --------------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------------

@retry(tries=3, delay=2, backoff=2)
def place_order(conn, security_id, transaction_type, quantity):
    """
    Place a market order and return Dhan's response, which looks like:

        {"orderId": "112111182198", "orderStatus": "PENDING"}

    Note that this only tells you the order was accepted. Call wait_for_fill() to
    find out what price it actually filled at.

    productType is MARGIN on purpose. INTRADAY would be auto-squared-off by the
    broker at around 3:20pm every day, which would silently close a weekly spread
    that is meant to be held to expiry.
    """
    if quantity > NIFTY_FREEZE_QTY:
        raise Exception(
            f"Quantity {quantity} is above the exchange freeze limit of "
            f"{NIFTY_FREEZE_QTY}. Split the order into smaller slices."
        )

    body = {
        "dhanClientId": conn["client_id"],
        "transactionType": transaction_type,   # "BUY" or "SELL"
        "exchangeSegment": "NSE_FNO",
        "productType": "MARGIN",
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": str(security_id),
        "quantity": int(quantity),
        "price": 0,
    }

    response = requests.post(
        f"{API_BASE}/orders", headers=build_headers(conn), json=body, timeout=30
    )
    response.raise_for_status()
    return response.json()


@retry(tries=5, delay=2, backoff=2)
def get_order(conn, order_id):
    """
    Fetch the current state of one order. The useful fields are orderStatus and
    averageTradedPrice.
    """
    response = requests.get(
        f"{API_BASE}/orders/{order_id}", headers=build_headers(conn), timeout=30
    )
    response.raise_for_status()
    data = response.json()
    # Dhan returns a single-element list for this endpoint.
    if isinstance(data, list):
        return data[0]
    return data


def wait_for_fill(conn, order_id, tries=15, delay=2):
    """
    Poll an order until it reaches a final state, and return that final order dict.

    A market order normally fills in well under a second, but we poll rather than
    assume. Terminal states on Dhan are TRADED, REJECTED and CANCELLED - note that
    the success value is TRADED, not COMPLETE as it was on Definedge.

    If the order is still pending after all the tries, we return whatever we last
    saw rather than raising, so the caller can decide what to do. The caller must
    always check orderStatus before using averageTradedPrice.
    """
    order = None
    count = 0
    while count < tries:
        order = get_order(conn, order_id)
        status = order.get("orderStatus")
        if status in ("TRADED", "REJECTED", "CANCELLED"):
            return order
        time.sleep(delay)
        count = count + 1

    print(f"Order {order_id} still not final after {tries} checks.")
    return order
