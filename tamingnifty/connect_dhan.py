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
# 80 at a time to leave a margin, and fetch_candles() loops to cover the rest.
MAX_DAYS_PER_REQUEST = 80

# Dhan returns candle timestamps as true UTC epoch seconds. Converting them through
# UTC and then to Asia/Kolkata lands the first candle of the session exactly on
# 09:15:00, which is how this was verified. We then drop the timezone, because the
# bots compare these values against a plain datetime.today() and pandas raises a
# TypeError if you compare timezone-aware against timezone-naive.
IST = "Asia/Kolkata"

# NIFTY options are 65 per lot, read straight off the scrip master.
#
# There is deliberately no freeze-quantity constant here any more. The exchange
# freeze limit is per underlying (NIFTY and BANKNIFTY differ) and the exchange
# revises it from time to time, so a single number in a shared library is wrong for
# most callers and goes stale silently. A strategy that can actually reach the limit
# should check it itself, before it sends the first leg - see check_quantity() in
# the credit spread bot.
NIFTY_LOT_SIZE = 65

# Filled in by load_instruments() the first time it is needed, then reused for the
# life of the process so we do not download a 30 MB file inside the trading loop.
_instruments = None


# --------------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------------

# Dhan only lets you generate a token once every 2 minutes. The signal bot and the
# spread bot start at the same time and are separate processes, so the second one to
# start always gets refused. It just has to wait the lockout out.
#
# So the first retry waits 125 seconds - just past that 2 minute window - and the
# second waits 250 more, giving up 375 seconds after the first attempt. Retrying
# sooner than 125 seconds is pointless for two separate reasons:
#
#   1. The token lockout has not expired yet, so the answer will be the same.
#   2. A TOTP code is only valid for 30 seconds, and Dhan rejects a code that has
#      already been used. Retrying a few seconds later sends the SAME code again and
#      comes back as "Invalid TOTP" - which looks like a bad secret but is not one.
#
# Waiting 125 seconds guarantees both a new TOTP window and an expired lockout.
@retry(tries=3, delay=125, backoff=2)
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
def fetch_one_chunk(conn, security_id, exchange_segment, instrument, start, end, interval):
    """
    One request to Dhan for one security over a range already known to be inside the
    90 day limit. fetch_candles() below is what you normally want.
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


def fetch_candles(conn, security_id, exchange_segment, instrument, start, end, interval="min"):
    """
    Candles for any Dhan instrument, as a DataFrame with columns datetime, open,
    high, low, close, volume.

    exchange_segment / instrument is the pair that says what you are asking for:

        index    "IDX_I",   "INDEX"
        option   "NSE_FNO", "OPTIDX"
        equity   "NSE_EQ",  "EQUITY"

    Intraday requests are split into MAX_DAYS_PER_REQUEST windows and stitched back
    together, because Dhan refuses anything wider than 90 days. This matters after
    the bot has been down for a while: the signal doc can hold a start_date months
    back, and without chunking that request would simply fail. Daily candles have no
    such limit, so they go in one request.
    """
    if interval == "day":
        return fetch_one_chunk(
            conn, security_id, exchange_segment, instrument, start, end, "day"
        )

    frames = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST)
        if chunk_end > end:
            chunk_end = end

        part = fetch_one_chunk(
            conn, security_id, exchange_segment, instrument, chunk_start, chunk_end, "min"
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


def fetch_historical_data(conn, exchange, trading_symbol, start, end, interval="min"):
    """
    Candles for an INDEX, looked up by name ("Nifty 50", "India VIX", "Nifty Bank").

    The signature deliberately matches connect_definedge.fetch_historical_data so
    that ta.py works unchanged. `exchange` is accepted and ignored - Dhan works out
    the venue from the segment, but keeping the argument means no call site changes.
    """
    security_id = get_security_id(trading_symbol)
    return fetch_candles(conn, security_id, "IDX_I", "INDEX", start, end, interval)


@retry(tries=5, delay=5, backoff=2)
def get_ltp(conn, security_id, exchange_segment):
    """
    Last traded price for one instrument, in whichever segment it lives in
    ("IDX_I", "NSE_FNO" or "NSE_EQ").

    Note this endpoint also wants the client id in the headers, which the candle
    endpoints do not.
    """
    response = requests.post(
        f"{API_BASE}/marketfeed/ltp",
        headers=build_headers(conn, include_client_id=True),
        json={exchange_segment: [int(security_id)]},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    price = data["data"][exchange_segment][str(security_id)]["last_price"]
    return round(float(price), 2)


def fetch_ltp(conn, exchange, trading_symbol):
    """
    Last traded price for an index, for example "India VIX".

    `exchange` is accepted and ignored, to match the Definedge signature.
    """
    return get_ltp(conn, get_security_id(trading_symbol), "IDX_I")


def get_option_ltp(conn, security_id):
    """Last traded price for a single option contract. Used by the running PnL loop."""
    return get_ltp(conn, security_id, "NSE_FNO")


def get_equity_ltp(conn, security_id):
    """Last traded price for one NSE cash instrument (a share or an ETF)."""
    return get_ltp(conn, security_id, "NSE_EQ")


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


def fetch_equity_data(conn, security_id, start, end, interval="day"):
    """
    Candles for an NSE cash instrument (a share or an ETF).

    One thing worth knowing: Dhan publishes the consolidated DAILY candle for a
    session very late - on 2026-09-25 that day's daily bar still did not exist at
    22:57, long after the minute bars were complete. So a job that needs a session's
    daily close must run the NEXT morning, not the same evening.
    """
    return fetch_candles(conn, security_id, "NSE_EQ", "EQUITY", start, end, interval)


# --------------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------------

def check_dhan_response(response):
    """
    Raise a useful error if Dhan rejected the request.

    requests' own raise_for_status() only says "400 Client Error: for url ..." and
    throws the response body away - but the body is the ONLY place Dhan tells you
    what was actually wrong, in an errorMessage field. Without it a rejected order
    is undebuggable, which is exactly what happened on the first live attempt on
    2026-09-25.

    This raises the same kind of error, with Dhan's own explanation attached.
    """
    if response.status_code < 400:
        return

    # The body is normally JSON, but fall back to raw text if it is not.
    try:
        detail = response.json()
    except Exception:
        detail = response.text

    raise Exception(
        f"Dhan returned HTTP {response.status_code} for {response.url} - {detail}"
    )


def send_order(conn, security_id, transaction_type, quantity,
               exchange_segment, product_type):
    """
    Place a MARKET order and return Dhan's response, which looks like:

        {"orderId": "112111182198", "orderStatus": "PENDING"}

    That only says the order was accepted. Call wait_for_fill() to find out what
    price it actually filled at.

    exchange_segment / product_type is the pair that says what kind of order this
    is. The two wrappers below are what strategies should normally call:

        options   "NSE_FNO", "MARGIN"
        equity    "NSE_EQ",  "CNC"

    Neither uses INTRADAY, on purpose - the broker auto-squares-off an INTRADAY
    position at around 3:20pm, which would silently close a weekly spread meant to
    be held to expiry, or an ETF meant to be held for weeks.
    """
    body = {
        "dhanClientId": conn["client_id"],
        "transactionType": transaction_type,   # "BUY" or "SELL"
        "exchangeSegment": exchange_segment,
        "productType": product_type,
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": str(security_id),
        "quantity": int(quantity),
        "price": 0,
    }

    response = requests.post(
        f"{API_BASE}/orders", headers=build_headers(conn), json=body, timeout=30
    )
    # A 400 here means Dhan did not like something in the body above, so print what
    # we sent as well as what Dhan said. One of the two will name the problem.
    if response.status_code >= 400:
        print(f"Order Dhan rejected: {body}")
    check_dhan_response(response)
    return response.json()


@retry(tries=3, delay=2, backoff=2)
def place_order(conn, security_id, transaction_type, quantity):
    """Market order for an index option contract, held overnight (MARGIN)."""
    return send_order(conn, security_id, transaction_type, quantity,
                      "NSE_FNO", "MARGIN")


def place_equity_order(conn, security_id, transaction_type, quantity):
    """
    Market order for NSE cash delivery (CNC).

    There is deliberately no @retry on this one. A retry can only help if the
    request never reached Dhan, and from here we cannot tell that apart from a
    reply that got lost on the way back - in which case retrying buys the same ETF
    twice. The caller notifies on failure and we look at it by hand instead.
    """
    return send_order(conn, security_id, transaction_type, quantity,
                      "NSE_EQ", "CNC")


@retry(tries=5, delay=2, backoff=2)
def get_order(conn, order_id):
    """
    Fetch the current state of one order. The useful fields are orderStatus and
    averageTradedPrice.
    """
    response = requests.get(
        f"{API_BASE}/orders/{order_id}", headers=build_headers(conn), timeout=30
    )
    check_dhan_response(response)
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
