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
import sys
import time
from datetime import datetime, timedelta
from pymongo import MongoClient
from retry import retry
import os
from dotenv import (  # pip install python-dotenv
    find_dotenv,
    load_dotenv,
)

from tamingnifty import utils as util

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

API_BASE = "https://api.dhan.co/v2"

# Statuses that mean Dhan has finished with an order, one way or another. EXPIRED
# is in here because it is terminal too - leaving it out used to make wait_for_fill
# sit through every one of its polls on an order that was already dead.
#
# PART_TRADED is deliberately NOT in here. Some of it filled and the rest is still
# working, which is precisely the case worth chasing. Note it only ever appears on
# the order book / get-order endpoints, never in the reply to placing an order.
FINAL_ORDER_STATUSES = ("TRADED", "REJECTED", "CANCELLED", "EXPIRED")

# A limit price that is not a multiple of the instrument's tick is rejected outright,
# so every price we calculate gets rounded onto it.
#
# THE TICK IS PER INSTRUMENT, NOT PER SEGMENT. This started life as a per-segment
# table with MCX_COMM = 1.00, read off crude, and that is wrong for most of MCX.
# Counted in the scrip master on 2026-09-29:
#
#   MCX futures alone use FIVE different ticks - Rs 0.05 (copper, zinc, aluminium),
#   Rs 0.10 (natural gas), Rs 0.50, Rs 1.00 (crude, gold, silver), Rs 10.00 (cotton,
#   steel rebar).
#
#   NSE cash uses SIX - and 467 instruments tick coarser than 5 paise, including
#   Maruti Suzuki, Page Industries, Divi's Labs, InterGlobe Aviation and Oracle
#   Financial. A price rounded onto a 5 paise grid is not on a 10 paise grid, so
#   every one of those would have had its reprice rejected.
#
# Rounding COARSER than the real tick is safe when the real tick divides it exactly
# (0.05 is a legal multiple of 0.01), which is why the ETFs the momentum bot holds
# never hit this - all 345 NSE ETFs tick at 0.01 or 0.05. It is only luck.
#
# get_tick_size() below reads the real number per instrument. This table is what it
# falls back to when the scrip master cannot be reached, because a coarse guess that
# usually works beats refusing to reprice at all.
FALLBACK_TICK_SIZE = {
    "NSE_EQ": 0.05,
    "NSE_FNO": 0.05,
    "BSE_EQ": 0.05,
    "BSE_FNO": 0.05,
    "MCX_COMM": 0.05,
}

# Anything unlisted falls back to the finest tick we know of. Too fine gets the order
# rejected and we find out at once; too coarse would silently move the price further
# than intended, which is worse.
DEFAULT_TICK_SIZE = 0.05

# Dhan's API names a segment one way ("NSE_EQ"); its own scrip master names the same
# segment another way, as an EXCH_ID plus a one letter SEGMENT code. This is the
# translation, needed by any lookup that starts from an order and goes to the master.
#
# IDX_I is deliberately absent - indices cannot be traded, so nothing ever needs a
# tick for one, and the code that reads indices already knows their ids.
SEGMENT_TO_MASTER = {
    "NSE_EQ": ("NSE", "E"),
    "NSE_FNO": ("NSE", "D"),
    "NSE_CURRENCY": ("NSE", "C"),
    "BSE_EQ": ("BSE", "E"),
    "BSE_FNO": ("BSE", "D"),
    "BSE_CURRENCY": ("BSE", "C"),
    "MCX_COMM": ("MCX", "M"),
}

AUTH_URL = "https://auth.dhan.co/app/generateAccessToken"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

# ---- The shared access token ------------------------------------------------------
#
# Dhan keeps exactly ONE live access token per account. Logging in again does not
# fail and does not give you a second session - it silently kills the token every
# other process is holding, and they only find out later, mid-session, when an
# unrelated call comes back with:
#
#     {'errorType': 'Order_Error', 'errorCode': 'DH-906', 'errorMessage': 'Invalid Token'}
#
# That error code says "order problem", which is why this took a day to identify.
# Proved by experiment on 2026-09-29: mint A, fetch OK, wait, mint B, and the same
# fetch with A then fails with exactly the message above while B works.
#
# We run four processes against this one account - the credit spread signal and
# executor all session, plus the two momentum cron jobs - so whichever logged in last
# was disabling all the others. The fix is that nobody owns a private token any more:
# one document in Mongo holds the token, every process reads it, and a new one is
# minted only when there isn't a usable one there.
#
# The database name is deliberately NOT configurable. The token belongs to the Dhan
# account, not to a bot, so every process must look in the same place regardless of
# whatever MONGO_DB it uses for its own state.
TOKEN_DB_NAME = "Bots"
TOKEN_COLLECTION_NAME = "dhan_token"

# Dhan's tokens last 24 hours. We re-mint at 20 so a bot that starts late in the day
# is never handed one that expires mid-session.
TOKEN_MAX_AGE_HOURS = 20

# One MongoClient for the life of the process. login_to_dhan() is called on every
# pass of a 10-second loop, and opening a connection each time would be silly.
_token_collection = None

# login_to_dhan() asks for the store more than once per call, and runs every few
# seconds, so the "there is no store" warning has to be said once and then shut up.
_warned_about_missing_store = False

# When Mongo is unreachable, building the client is not cheap to get wrong: an Atlas
# mongodb+srv:// URI resolves DNS at construction, and a DNS timeout took 21 seconds
# on a real outage here. Retrying that on every pass would turn a 10 second loop into
# a 30 second one. So after a failure we stop trying for a while and just use the
# token we already have.
_store_retry_after = None
TOKEN_STORE_RETRY_MINUTES = 5

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

def _get_token_collection():
    """
    The Mongo collection holding the one shared access token, or None.

    Never raises. A broker login must not die because Mongo is unreachable - if it
    is, login_to_dhan() falls back to the old per-process behaviour. That still
    works, it just loses the sharing.
    """
    global _token_collection
    if _token_collection is not None:
        return _token_collection

    global _warned_about_missing_store, _store_retry_after

    # Still inside the cooldown from a failed attempt - do not pay the DNS timeout again.
    if _store_retry_after is not None and datetime.now() < _store_retry_after:
        return None

    connection_string = os.environ.get("CONNECTION_STRING")
    if not connection_string:
        if _warned_about_missing_store == False:
            print("No CONNECTION_STRING set, so the Dhan token cannot be shared between bots.", flush=True)
            _warned_about_missing_store = True
        _store_retry_after = datetime.now() + timedelta(minutes=TOKEN_STORE_RETRY_MINUTES)
        return None
    try:
        _token_collection = MongoClient(connection_string)[TOKEN_DB_NAME][TOKEN_COLLECTION_NAME]
        return _token_collection
    except Exception as e:
        print(f"Could not reach the shared Dhan token store, retrying in "
              f"{TOKEN_STORE_RETRY_MINUTES} minutes: {util.exception_detail(e)}", flush=True)
        _store_retry_after = datetime.now() + timedelta(minutes=TOKEN_STORE_RETRY_MINUTES)
        return None


def _read_shared_token(client_id):
    """
    Return the token the bots are currently sharing, or None if there is not a usable
    one. Never raises, for the same reason as above.
    """
    collection = _get_token_collection()
    if collection is None:
        return None

    try:
        doc = collection.find_one({"_id": client_id})
    except Exception as e:
        print(f"Could not read the shared Dhan token: {util.exception_detail(e)}", flush=True)
        return None

    if not doc or not doc.get("access_token") or not doc.get("minted_at"):
        return None

    age = datetime.now() - doc["minted_at"]
    if age > timedelta(hours=TOKEN_MAX_AGE_HOURS):
        print(f"Shared Dhan token is {round(age.total_seconds() / 3600, 1)} hours old, "
              f"past the {TOKEN_MAX_AGE_HOURS} hour limit. Minting a new one.", flush=True)
        return None

    return doc["access_token"]


def _save_shared_token(client_id, access_token, expiry_time):
    """
    Publish a freshly minted token so every other bot picks it up instead of minting
    its own and killing this one. Never raises.
    """
    collection = _get_token_collection()
    if collection is None:
        return
    try:
        collection.update_one(
            {"_id": client_id},
            {"$set": {
                "access_token": access_token,
                "minted_at": datetime.now(),
                "expiry_time": expiry_time,
                "minted_by": _who_am_i(),
            }},
            upsert=True,
        )
    except Exception as e:
        print(f"Could not publish the new Dhan token: {util.exception_detail(e)}", flush=True)


def _who_am_i():
    """The script that is running, e.g. "credit_spread_signal.py". Used in the mint
    notification so the channel says which bot took the token."""
    return os.path.basename(sys.argv[0]) or "unknown"


def _announce_new_token(expiry_time, slack_channel):
    """
    Post one line to Slack saying a new token was minted. Never raises - a Slack
    outage must not stop a bot logging in.

    A mint is the one event that affects every other bot on the account: the token
    they are holding is dead from this moment until they re-read the store. When a
    bot goes quiet, this line is what tells you why, and who did it.
    """
    channel = slack_channel or os.environ.get("slack_channel") or "niftyweekly"
    try:
        util.notify(
            message=(f"Dhan token minted by {_who_am_i()}, valid till {expiry_time}. "
                     f"Dhan allows one token per account, so this replaces the previous one - "
                     f"the other bots pick it up from Mongo on their next pass."),
            slack_channel=channel,
            slack_client=util.get_slack_client(token=os.environ.get("slack_token")),
        )
    except Exception as e:
        print(f"Could not announce the new Dhan token: {util.exception_detail(e)}", flush=True)


# Dhan only lets you generate a token once every 2 minutes, so two bots starting
# together means the second one is refused and has to wait the lockout out.
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
#
# Note that the retry is now mostly a backstop rather than the plan: since the token
# is shared through Mongo, the usual reason a second bot used to need a login at all
# has gone away. And if it does retry, it re-enters this function from the top and
# re-reads the store, so it will normally find the token the other bot just published
# instead of minting one.
@retry(tries=3, delay=125, backoff=2)
def login_to_dhan(fresh=False, slack_channel=None):
    """
    Log in to Dhan and return a connection dict:

        {"client_id": "1100341129", "access_token": "eyJ0eXAi..."}

    Every other function in this file takes that dict as its first argument, the same
    way the Definedge functions take a ConnectToIntegrate object.

    Dhan keeps ONE live token per account and a new login silently kills the previous
    one, so the token is not private to this process - it lives in Mongo and every bot
    reads the same one. See the comment on TOKEN_DB_NAME for how that was established.

    Pass fresh=True only when a call has just come back with "Invalid Token", which
    means the shared token was killed by something outside our bots (a login from the
    Dhan website, or somebody running a script by hand). Even then this checks the
    store first and will adopt another bot's newer token rather than mint.
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

    shared = _read_shared_token(client_id)

    if fresh == False:
        # The normal path. Read the store every time rather than caching in this
        # process, so that when another bot does mint a new token we pick it up on
        # the next pass instead of spending the session failing on a dead one.
        if shared:
            os.environ["DHAN_ACCESS_TOKEN"] = shared
            return {"client_id": client_id, "access_token": shared}

        # No usable shared token. If we cannot even reach the store, keep using
        # whatever this process already has - minting blind would kill a token the
        # other bots are perfectly happy with.
        if _get_token_collection() is None and os.environ.get("DHAN_ACCESS_TOKEN"):
            return {
                "client_id": client_id,
                "access_token": os.environ["DHAN_ACCESS_TOKEN"],
            }
    else:
        # fresh=True: a call just failed with "Invalid Token". Before minting, check
        # whether another bot has already replaced it. If what is in the store is not
        # the dead token we were holding, they fixed it while we were failing, and
        # minting now would kill their new token, so they would mint again, and the
        # two bots would knock each other over for the rest of the session.
        if shared and shared != os.environ.get("DHAN_ACCESS_TOKEN"):
            print("Another bot has already minted a token, adopting that one instead "
                  "of minting again.", flush=True)
            os.environ["DHAN_ACCESS_TOKEN"] = shared
            return {"client_id": client_id, "access_token": shared}

    # Mint a new 24 hour token. TOTP must be enabled on the Dhan account for this
    # endpoint to work - without it there is no way to log in without a browser.
    totp_now = pyotp.TOTP(totp_secret).now()
    response = requests.post(
        AUTH_URL,
        params={"dhanClientId": client_id, "pin": pin, "totp": totp_now},
        timeout=30,
    )
    check_dhan_response(response)
    data = response.json()

    if "accessToken" not in data:
        raise Exception(f"Dhan login failed: {data}")

    access_token = data["accessToken"]
    os.environ["DHAN_ACCESS_TOKEN"] = access_token

    # Publish before announcing. If Slack is down we still want the other bots to be
    # able to find the token, and that ordering also means the notification is never
    # sent for a token nobody else can reach.
    _save_shared_token(client_id, access_token, data.get("expiryTime"))
    print(f"Login successful. Token valid till {data.get('expiryTime')}", flush=True)
    _announce_new_token(data.get("expiryTime"), slack_channel)

    return {"client_id": client_id, "access_token": access_token}



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


def get_tick_size(security_id, exchange_segment):
    """
    The smallest price increment this one instrument may be priced in, in RUPEES.

    Any limit price that is not a whole multiple of this is rejected by the exchange,
    so every price the chase works out is rounded onto it.

    It has to be looked up per instrument. There is no per-segment answer: MCX
    futures use five different ticks and NSE cash uses six. See the comment on
    FALLBACK_TICK_SIZE for the counts and for who it would have broken.

    TWO THINGS THAT WILL CATCH YOU OUT HERE:

    1. The scrip master reports the tick in PAISE. Crude reads 100.0 meaning Re 1.00,
       a NIFTY option reads 5.0 meaning Rs 0.05. Hence the divide by 100.

    2. A SECURITY ID IS NOT UNIQUE ON ITS OWN. 10,631 of the 188,183 ids in the
       master appear more than once, and the collisions are not obscure - id 13 is
       both Nifty 50 and ABB, id 25 is both Nifty Bank and Adani Enterprises, id 21
       is both India VIX and an SDL bond. Only (id, exchange, segment) is unambiguous,
       and it is: zero triples collide. So this filters on all three, and anything
       that looks an instrument up by id alone is a bug waiting to happen.

    Never raises. A tick we cannot look up must not stop an exit - the caller gets
    the segment fallback and carries on, because a slightly coarse price that the
    exchange accepts is worth far more than a correct one we never sent.
    """
    fallback = FALLBACK_TICK_SIZE.get(exchange_segment, DEFAULT_TICK_SIZE)

    if exchange_segment not in SEGMENT_TO_MASTER:
        return fallback
    exchange, segment = SEGMENT_TO_MASTER[exchange_segment]

    try:
        df = load_instruments()
        match = df[
            (df["SECURITY_ID"].astype(str) == str(security_id))
            & (df["EXCH_ID"].astype(str) == exchange)
            & (df["SEGMENT"].astype(str) == segment)
        ]
        if len(match) == 0:
            print(f"No scrip master entry for security {security_id} in "
                  f"{exchange_segment}, using a {fallback} tick.", flush=True)
            return fallback
        return round(float(match.iloc[0]["TICK_SIZE"]) / 100.0, 4)
    except Exception as e:
        print(f"Could not read the tick size for security {security_id} "
              f"({util.exception_detail(e)}), using {fallback}.", flush=True)
        return fallback


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


@retry(tries=5, delay=5, backoff=2)
def get_commodity_futures_symbol(commodity, min_dte=3):
    """
    Find the front month MCX futures contract for a commodity, for example
    "CRUDEOILM", "NATURALGAS", "GOLDM", "SILVERM".

    Returns (trading_symbol, security_id, expiry_date, lot_size) - the same shape
    get_index_option_symbol returns, so both can be used the same way.

    Commodity futures roll monthly and the contract nearest expiry is the liquid
    one, so this takes the earliest expiry that is still more than `min_dte` days
    out. The cutoff matters more here than it does for options: MCX expiries are
    mid-month rather than weekly, and the last few days of a contract thin out
    badly, which is exactly when a chase cannot find a fill.

    LOT_SIZE is 1 for every one of the 14,961 MCX rows in the scrip master, because
    on MCX the contract itself IS the unit - Dhan's `quantity` is a number of lots.
    One lot of CRUDEOILM is 10 barrels, so a 1 rupee move in crude is Rs 10 on the
    position. That multiplier is a property of the contract, not something the API
    exposes, so a strategy that sizes by risk has to carry it itself.
    """
    df = load_instruments()

    df = df[
        (df["EXCH_ID"].astype(str) == "MCX")
        & (df["INSTRUMENT"].astype(str) == "FUTCOM")
        & (df["UNDERLYING_SYMBOL"].astype(str) == commodity)
    ]

    if len(df) == 0:
        raise Exception(f"No MCX futures contract found for '{commodity}'.")

    df = df.copy()
    df["EXPIRY"] = pd.to_datetime(df["SM_EXPIRY_DATE"], errors="coerce")
    cutoff = datetime.now() + timedelta(days=min_dte)
    df = df[df["EXPIRY"] > cutoff]
    df = df.sort_values(by="EXPIRY", ascending=True)

    if len(df) == 0:
        raise Exception(
            f"No {commodity} futures contract expiring after {cutoff.date()}."
        )

    row = df.iloc[0]
    trading_symbol = row["DISPLAY_NAME"]
    security_id = int(row["SECURITY_ID"])
    expiry = row["EXPIRY"].date()
    lot_size = int(row["LOT_SIZE"])

    print(f"Commodity: {trading_symbol} , Security Id: {security_id} , Expiry: {expiry}")
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
    check_dhan_response(response)
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
    check_dhan_response(response)
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


def get_commodity_ltp(conn, security_id):
    """Last traded price for one MCX futures contract."""
    return get_ltp(conn, security_id, "MCX_COMM")


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


def fetch_commodity_data(conn, security_id, start, end, interval="min"):
    """
    Candles for one MCX futures contract.

    Note the default interval differs from fetch_equity_data's. MCX contracts roll
    every month, so a daily series on a single security id only ever covers that
    contract's own life - a few months at most, and the early part of it is thin.
    Anything wanting a long continuous commodity history has to stitch contracts
    together itself; this returns exactly one contract's candles and nothing more.
    """
    return fetch_candles(conn, security_id, "MCX_COMM", "FUTCOM", start, end, interval)


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

    Used by every Dhan call, not just orders. On 2026-09-28 both credit spread bots
    spent a whole session posting "400 Client Error:  for url: .../charts/intraday"
    with no way to tell a dead token (DH-901) from a bad request (DH-905) from Dhan
    failing to serve data (DH-907) - three different problems, one useless message.

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
    is. The three wrappers below are what strategies should normally call:

        options     "NSE_FNO",  "MARGIN"    place_order()
        equity      "NSE_EQ",   "CNC"       place_equity_order()
        commodity   "MCX_COMM", "MARGIN"    place_commodity_order()

    None uses INTRADAY, on purpose - the broker auto-squares-off an INTRADAY
    position at around 3:20pm, which would silently close a weekly spread meant to
    be held to expiry, an ETF meant to be held for weeks, or an MCX position during
    the session that runs on until 23:30.

    WHAT `quantity` MEANS. Always lots x the contract's LOT_SIZE from the scrip
    master - one rule, but it looks like two because LOT_SIZE differs:

        NSE_EQ      LOT_SIZE 1   -> quantity is a number of SHARES
        NSE_FNO     LOT_SIZE 65  -> one NIFTY lot is quantity=65, not 1
        MCX_COMM    LOT_SIZE 1   -> quantity is a number of LOTS

    MCX reads 1 for all 14,961 of its rows, so quantity=1 is one whole lot - which
    for CRUDEOILM is 10 barrels and about Rs 87,000 of notional on Rs 27,800 of
    margin. Sending 65 there the way you would for NIFTY buys 65 lots.
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


def place_order(conn, security_id, transaction_type, quantity):
    """
    Market order for an index option contract, held overnight (MARGIN).

    There is deliberately no @retry here, for the same reason spelled out under
    place_equity_order below: a retry only helps if the request never reached Dhan,
    and from this side that is indistinguishable from a reply that got lost on the
    way back - in which case the retry sells a second short leg. This one used to
    carry @retry(tries=3); it was removed on 2026-09-29.
    """
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


def place_commodity_order(conn, security_id, transaction_type, quantity):
    """
    Market order for an MCX futures contract, held overnight (MARGIN).

    quantity here is a number of LOTS, because every MCX row in the scrip master
    reads LOT_SIZE 1. That is the same rule as everywhere else - lots x LOT_SIZE -
    it just lands on a different number. One CRUDEOILM lot is 10 barrels, roughly
    Rs 87,000 of notional on about Rs 27,800 of margin, so quantity=65 out of habit
    from the NIFTY path would be 65 lots and around Rs 1.8 crore of exposure.

    MARGIN, not INTRADAY: MCX runs to 23:30 and an INTRADAY position would be
    squared off partway through. No @retry, for the reason under place_order.
    """
    return send_order(conn, security_id, transaction_type, quantity,
                      "MCX_COMM", "MARGIN")


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


def modify_order(conn, order_id, order_type, quantity, price=0):
    """
    Change an order that is already sitting at the exchange. In practice this means
    moving its price, because the two useful changes Dhan's docs imply are possible
    turn out not to be - see the two findings below.

    Always modify rather than cancel-and-replace. Cancelling first leaves a window
    with no working order against a position that is still open, and if the process
    dies in that window nothing is looking after it at all. A modify also cannot
    double the position the way re-placing can: it sets a price on an order id that
    already exists, so if the reply gets lost on the way back, the next poll simply
    shows us the truth. That is the same hazard place_order's comment describes, and
    it is why there is no @retry on that one but this is safe to call.

    DO NOT ADD legName BACK. Dhan's own worked example for this endpoint includes
    `"legName": ""`, and sending it fails EVERY modify with DH-905 "Missing required
    fields, bad values for parameters etc." - which reads like something is absent
    rather than something extra being present, so it is easy to chase the wrong
    field for a long time. The empty string is not a legal value for the enum; the
    field only applies to bracket and cover orders. Omitting it entirely works.
    Proved on MCX 2026-09-29: identical payloads, the only difference being that
    key, gave 400 and 200.

    LIMIT -> MARKET DOES NOT WORK either, whatever the docs imply by listing order
    type as modifiable. Every shape is refused with DH-906, and the message shifts
    with the price - "Invalid Price Value" for any non-zero price, "Basic Validation
    Failed" for zero or absent - so Dhan wants price 0 for a MARKET and still will
    not make the change. Tested six ways on MCX 2026-09-29, with a plain price move
    on the same order succeeding immediately afterwards to prove the order itself
    was still modifiable. This is why chase_the_order prices by hand.

    `quantity` is the WHOLE order, not the part still unfilled. Dhan's docs do not
    actually say which it means, but every exchange works the total-quantity way and
    Dhan's own worked example sends the full number. wait_for_fill checks Dhan's
    reported quantity after each modify and stops if it ever changes, because being
    wrong about this on a part-filled order would add to the position instead of
    finishing it.
    """
    body = {
        "dhanClientId": conn["client_id"],
        "orderId": str(order_id),
        "orderType": order_type,
        "quantity": int(quantity),
        "price": float(price),
        "validity": "DAY",
    }

    response = requests.put(
        f"{API_BASE}/orders/{order_id}", headers=build_headers(conn), json=body, timeout=30
    )
    # Same reasoning as send_order: on a 400, print what we sent as well as what Dhan
    # said, because one of the two will name the problem.
    if response.status_code >= 400:
        print(f"Modify Dhan rejected: {body}")
    check_dhan_response(response)
    return response.json()


def note_if_market_became_limit(order):
    """
    Say so when Dhan rewrites one of our MARKET orders into a LIMIT.

    send_order only ever sends MARKET, so any order that comes back as LIMIT was
    rewritten by Dhan into a protected limit priced off the LTP at that moment.
    Proved on MCX on 2026-09-29: we sent MARKET, Dhan stored LIMIT at 8874.0 against
    an LTP of 8779, a band of about 1% through the market. It is undocumented.

    Usually this is a kindness - the limit sits through the book, so it fills at once
    and caps our slippage. It only bites when the price runs past the band before the
    order rests, which is exactly what happens in a fast move. That is the whole
    reason the chase below exists.

    This line is also how we learn which segments do it. MCX_COMM is confirmed;
    NSE_EQ and NSE_FNO have not been watched yet, and the next live order on either
    will print the answer here without us having to go and test it.
    """
    if order.get("orderType") == "LIMIT":
        print(f"Dhan rewrote MARKET order {order.get('orderId')} as a LIMIT at "
              f"{order.get('price')} on {order.get('exchangeSegment')}.", flush=True)


def chase_the_order(conn, order, attempt, step):
    """
    Push one order that is sitting unfilled towards a fill. Called once per attempt.

    It reprices, widening each time. The obvious move - asking Dhan to turn it back
    into a MARKET order, so it redoes its own protected-limit sum at the price now
    instead of the price when we placed it - is refused by the API; modify_order's
    docstring has the evidence. So we work out the price ourselves.

    We deliberately do NOT price at the LTP: a limit sitting exactly at the LTP only
    joins the queue, which is the problem we are trying to solve. Priced through it,
    the order is marketable and normally fills at the touch - better than the price
    we offered. Repricing to the far side of the LTP is allowed: a BUY limit one
    rupee under crude's LTP was accepted on 2026-09-29, so what bounds us is the
    exchange's circuit band, not how close to the market we dare go.

    Nothing here is direction-specific except one comparison, because an order that
    will not fill is just as bad whichever way round it is. A buy that misses leaves
    the credit spread short with no hedge; a sell that misses leaves the bot certain
    it is flat when it is still holding.
    """
    order_id = order.get("orderId")
    quantity = order.get("quantity")

    # 0.3%, then 0.6%, then 1.2%. Widening covers a bigger move each time, and the
    # hard stop after the last attempt matters more than the sizes do - chasing
    # without a limit in a falling market just guarantees a sale at the bottom.
    offset = step * (2 ** (attempt - 1))

    ltp = get_ltp(conn, order.get("securityId"), order.get("exchangeSegment"))
    if order.get("transactionType") == "BUY":
        price = ltp * (1 + offset)
    else:
        price = ltp * (1 - offset)

    # The tick has to come from the instrument, not the segment. MCX futures alone
    # run from 5 paise to Rs 10 depending on the commodity, and 467 NSE cash names
    # are coarser than 5 paise. See get_tick_size.
    tick = get_tick_size(order.get("securityId"), order.get("exchangeSegment"))
    price = round(round(price / tick) * tick, 2)

    modify_order(conn, order_id, "LIMIT", quantity, price)
    print(f"Order {order_id} still unfilled - repriced to {price} "
          f"({round(offset * 100, 2)}% through an LTP of {ltp}).", flush=True)


def warn_order_never_filled(order, slack_channel):
    """
    Shout about an order that never filled.

    This is the one failure that quietly corrupts everything downstream. The strategy
    carries on as though the trade happened, so the ledger, the P&L and the next
    rebalance are all working from a position that is not the one actually held. It
    has to be loud, and the caller must not record a fill after seeing it.
    """
    message = (f"Order {order.get('orderId')} did NOT fill: "
               f"{order.get('transactionType')} {order.get('filledQty') or 0} of "
               f"{order.get('quantity')} {order.get('tradingSymbol')}, "
               f"status {order.get('orderStatus')}. "
               f"The real position is not what the bot thinks it is - check Dhan by hand.")
    print(message, flush=True)
    try:
        util.notify(
            message=message,
            slack_channel=slack_channel or os.environ.get("slack_channel") or "niftyweekly",
            slack_client=util.get_slack_client(token=os.environ.get("slack_token")),
        )
    except Exception as e:
        print(f"Could not post the unfilled order warning to Slack: "
              f"{util.exception_detail(e)}", flush=True)


def wait_for_fill(conn, order_id, tries=5, delay=1, chases=3, step=0.003,
                  slack_channel=None):
    """
    Poll an order until it reaches a final state, chasing the price if it will not
    fill, and return whatever state it ended in.

    Terminal states on Dhan are TRADED, REJECTED, CANCELLED and EXPIRED - note the
    success value is TRADED, not COMPLETE as it was on Definedge.

    We poll rather than assume, because a MARKET order does not necessarily behave
    like one: Dhan rewrites it into a protected LIMIT (see note_if_market_became_limit),
    and in a fast move that limit can be left behind by the market and never fill. So
    after `tries` polls we chase - up to `chases` times, escalating - and only then
    give up. Worst case is about 20 seconds, which is less than the 30 this used to
    spend waiting passively.

    A REJECTED order is returned immediately and never chased. The reason matters
    (funds, freeze quantity, RMS) and repricing fixes none of them.

    If it still has not filled we return the last state we saw rather than raising,
    so the caller decides what to do - but warn_order_never_filled has already said
    so on Slack by then. The caller must always check orderStatus before using
    averageTradedPrice, and must check filledQty too: chasing makes a part fill a
    much more likely place to end up than it used to be.
    """
    order = get_order(conn, order_id)
    note_if_market_became_limit(order)

    # Dhan is never asked to change this. If it ever reports a different number we
    # have misunderstood what quantity means on a modify, and the safe thing is to
    # stop touching the order rather than risk adding to the position.
    original_quantity = order.get("quantity")

    attempt = 0
    while attempt <= chases:
        if attempt > 0:
            if order.get("quantity") != original_quantity:
                print(f"Order {order_id} quantity changed from {original_quantity} to "
                      f"{order.get('quantity')} after a modify. Not touching it again.",
                      flush=True)
                break
            chase_failed = False
            try:
                chase_the_order(conn, order, attempt, step)
            except Exception as e:
                # A chase can be refused for a reason no amount of repricing fixes.
                # The commonest is the price landing outside the exchange's circuit
                # limit, and that is not a rare corner: on MCX crude the lower
                # circuit sits about 1.5% below the last price, so a sell chased
                # 1.2% down is already close to it, and in the fast fall where this
                # code earns its keep the price IS the circuit.
                #
                # Stop rather than raise. The caller needs the order back so it can
                # see what the position really is, and warn_order_never_filled below
                # is what is supposed to get somebody's attention - an exception
                # thrown from here would skip it.
                print(f"Order {order_id} could not be chased "
                      f"({util.exception_detail(e)}). Giving up on it.", flush=True)
                chase_failed = True
            if chase_failed:
                break

        # Check what we already have before asking again - the order we fetched a
        # moment ago is usually already TRADED, and re-fetching it first would spend
        # an extra round trip on every single order we ever place.
        count = 0
        while count < tries:
            if order.get("orderStatus") in FINAL_ORDER_STATUSES:
                return order
            time.sleep(delay)
            order = get_order(conn, order_id)
            count = count + 1

        attempt = attempt + 1

    warn_order_never_filled(order, slack_channel)
    return order
