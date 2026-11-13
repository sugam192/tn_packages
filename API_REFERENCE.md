# API reference

Source-based reference for tamingnifty functions, including internal helpers and
legacy compatibility entry points. Signatures omit some type annotations, not
argument names or defaults, except the explicitly redacted utility defaults below.
Internal helpers are documented for understanding existing code, not as stable APIs.

## Conventions and safety

- Dhan `conn` means a dictionary containing `client_id` and `access_token`.
  Definedge `conn` means an existing `ConnectToIntegrate` connection.
- Broker, authentication, instrument-download, MongoDB and Slack functions perform
  real external operations. Nothing here is an instruction to run them against a
  live account. Examples below use only local, synthetic data.
- An accepted order is not a filled order. Submission timeouts leave an ambiguous
  outcome: reconcile with the broker rather than blindly submitting again.
  Submission functions deliberately have no automatic retry.
- Dhan candle `datetime` values are **timezone-naive IST wall-clock times**, not
  naive UTC. Do not mix them with timezone-aware timestamps. Definedge history is
  returned as supplied by its SDK; this wrapper does not normalize timezones.
- Unless stated otherwise, exceptions from requests, pandas, SDKs and malformed
  inputs propagate. Retry decorators retry raised exceptions, not unsuccessful
  return values. The common five-attempt policy waits 5, 10, 20 and 40 seconds
  between attempts; nested decorated calls can take longer.
- Install the optional `tamingnifty[definedge]` dependencies before importing
  `connect_definedge` or `tick_streamer`; both import `integrate` immediately.
  Importing `utils` constructs the default Slack `WebClient` used by `notify`,
  but does not post a message. Importing `connect_dhan` also imports `utils`.
  None of these modules logs in, downloads instruments or starts streaming just
  by being imported.
- This reference describes implementation, not a claim that every broker path or
  streaming callback has been validated live. Exchange rules, lot sizes, available
  history and broker responses can change.

## connect_dhan

### connect_dhan._get_token_collection

`_get_token_collection()`

**Internal.** Return the cached Mongo collection `Bots.dhan_token`, or `None`
when unavailable. Reads `CONNECTION_STRING`; constructs and caches a `MongoClient`
and may resolve DNS. Missing configuration or client-construction errors print a
warning and start a five-minute cooldown. Client construction is not a verified
database-health check; later reads/writes may still fail.

### connect_dhan._read_shared_token

`_read_shared_token(client_id)`

**Internal.** Look up the account's token document and return its `access_token`,
or `None` for missing fields, unavailable storage or age over 20 hours. Performs
a Mongo read and may print. Uses naive IST `minted_at`, not `expiry_time`, to
judge age; incompatible timestamp types can raise outside its database-error
handler. This does not validate the token with Dhan.

### connect_dhan._save_shared_token

`_save_shared_token(client_id, access_token, expiry_time)`

**Internal, credential-writing operation.** Upsert the token, naive IST
`minted_at`, supplied expiry and calling script name into shared Mongo storage.
Returns `None`; unavailable storage is skipped and database errors are printed.
There is no cross-process mint lock.

### connect_dhan._who_am_i

`_who_am_i()`

**Internal.** Return the basename of `sys.argv[0]`, or `"unknown"` if empty.
Reads process arguments only; no external effects. Assumes `sys.argv` has its
normal first entry.

### connect_dhan._announce_new_token

`_announce_new_token(expiry_time, slack_channel)`

**Internal, notification operation.** Return `None` after attempting a token-mint
notice through `utils.notify`. Channel precedence is the argument, environment
`slack_channel`, then the built-in fallback. Uses environment `slack_token`;
prints notification failures rather than raising them. Does not include the
access token itself in the notice.

### connect_dhan.login_to_dhan

`login_to_dhan(fresh=False, slack_channel=None)`

**Authentication operation.** Return `{"client_id": ..., "access_token": ...}`.
Loads dotenv and requires `DHAN_CLIENT_ID`, `DHAN_PIN`, `DHAN_TOTP` even when
reusing a token. Normally adopts a usable shared token; with no store, can retain
the process's `DHAN_ACCESS_TOKEN`. `fresh=True` first adopts a different shared
token if available, otherwise mints using PIN/TOTP.

Minting replaces the account's previous token, updates process environment,
attempts Mongo publication and posts a Slack notice. The store is not a lock:
simultaneous mint attempts remain possible. Retries three times with waits of
125 then 250 seconds; missing settings, HTTP errors or missing `accessToken`
eventually raise. Avoid casual refreshes that invalidate other processes.

### connect_dhan.build_headers

`build_headers(conn, include_client_id=False)`

Return a new request-header dictionary with `access-token` and JSON content
type; add `client-id` when the flag equals `True`. Pure local operation; missing
connection keys raise `KeyError`. The returned dictionary contains credentials:
do not log it.

### connect_dhan.get_security_id

`get_security_id(trading_symbol)`

Return the integer ID for exactly `"Nifty 50"`, `"Nifty Bank"` or `"India VIX"`.
Local fixed lookup; unknown names raise `KeyError`. This is not a general
instrument-symbol resolver, and an ID alone is not unique across segments.

### connect_dhan.load_instruments

`load_instruments()`

Download Dhan's detailed CSV master and return a pandas DataFrame, cached for
the process lifetime. Prints download status; retries five times on exceptions.
Returns the cached object itself, not a copy: caller edits alter later lookups.
No automatic daily refresh or explicit download timeout is configured here.

### connect_dhan.get_tick_size

`get_tick_size(security_id, exchange_segment)`

Return a tick size in rupees using security ID **and** exchange/segment from
the master; divides master `TICK_SIZE` by 100. May download the master.
Unsupported segments, absent rows and lookup errors use a segment/default
fallback (currently `0.05`); errors may print. Uses the first match and does not
strictly validate its tick. A fallback is not proof of a tradable tick; execution
uses the stricter `execution_tick` instead.

### connect_dhan.get_index_option_symbol

`get_index_option_symbol(strike, option_type, instrument_name="NIFTY", min_dte=3)`

Return `(display_name, integer_security_id, expiry_date, integer_lot_size)`
for an `OPTIDX` row with exact underlying, numeric strike and option type
(normally `"CE"`/`"PE"`). Select the nearest expiry strictly after current IST
time plus `min_dte` calendar days. Master expiries are midnight timestamps, so
this is not an exchange-session DTE calculation. May download, prints selection
and retries five times; absent eligible contracts raise. There is no additional
exchange filter; verify venue before execution.

### connect_dhan.get_commodity_futures_symbol

`get_commodity_futures_symbol(commodity, min_dte=3)`

Return `(display_name, integer_security_id, expiry_date, integer_lot_size)`
for the nearest MCX `FUTCOM` contract matching `commodity` exactly and expiring
strictly after current IST time plus the calendar-day cutoff. May download,
prints selection and retries five times; no matching contract raises. Returned
master lot size is not the physical price/P&L multiplier; callers must know the
contract's units and verify current specifications.

### connect_dhan.candles_to_dataframe

`candles_to_dataframe(data)`

Convert parallel `timestamp`, `open`, `high`, `low`, `close`, `volume` lists
into those OHLCV columns plus `datetime`. UTC epoch seconds become naive IST.
Missing/empty timestamps return an empty DataFrame with the six standard
columns. Pure local conversion; missing price fields, unequal list lengths or
invalid timestamps raise. Does not sort or fill gaps.

```python
from tamingnifty import connect_dhan

bars = connect_dhan.candles_to_dataframe({
    "timestamp": [0], "open": [100], "high": [102],
    "low": [99], "close": [101], "volume": [20],
})
assert str(bars.loc[0, "datetime"]) == "1970-01-01 05:30:00"
```

### connect_dhan.fetch_one_chunk

`fetch_one_chunk(conn, security_id, exchange_segment, instrument, start, end, interval)`

**Low-level network helper.** Return a standard candle DataFrame for one request.
`start`/`end` need `strftime`; only their dates are sent. Exactly `"day"` selects
daily history; every other interval sends one-minute intraday history. No range
splitting or interval validation here. Uses a 60-second request timeout and
five-attempt retry; HTTP/body-conversion errors propagate after retries.

### connect_dhan.fetch_candles

`fetch_candles(conn, security_id, exchange_segment, instrument, start, end, interval="min")`

Return candle DataFrame for the given ID, segment and instrument type. Typical
pairs are `IDX_I/INDEX`, `NSE_FNO/OPTIDX`, `NSE_EQ/EQUITY`,
`MCX_COMM/FUTCOM`. Daily requests pass through once; other intervals become
one-minute requests split at `start + 80 days`, then advance to the day after
the chunk end. Intraday results are concatenated, timestamp-deduplicated and
sorted; no data gives an empty six-column frame. Daily results do not get that
extra cleanup. Multiple network calls can occur; child calls retry.
Date-only request bounds and broker boundary semantics must be checked for
coverage; this wrapper does not prove complete sessions or offer arbitrary
minute intervals.

### connect_dhan.fetch_historical_data

`fetch_historical_data(conn, exchange, trading_symbol, start, end, interval="min")`

**Compatibility wrapper.** Return `fetch_candles` output for one of the three
fixed index names using `IDX_I/INDEX`. `exchange` is accepted but ignored.
Network operations and range/interval limits are inherited; other names raise
`KeyError` before fetching.

### connect_dhan.get_ltp

`get_ltp(conn, security_id, exchange_segment)`

Return `last_price` as float rounded to two decimals from Dhan's LTP endpoint.
Sends client-ID headers, uses a 30-second timeout and five-attempt retry.
Requires an integer-convertible ID and the expected response keys. A returned
LTP has no freshness, spread or executable-depth guarantee.

### connect_dhan.fetch_ltp

`fetch_ltp(conn, exchange, trading_symbol)`

**Compatibility wrapper.** Return two-decimal float LTP for a fixed index name
through `get_ltp(..., "IDX_I")`. Ignores `exchange`; unknown names raise.
Performs network access with the underlying retry policy.

### connect_dhan.get_option_ltp

`get_option_ltp(conn, security_id)`

Return two-decimal float LTP using `NSE_FNO`. Performs the retried `get_ltp`
request and inherits its errors and freshness limits; does not resolve symbols
or validate that the ID represents an option.

### connect_dhan.get_equity_ltp

`get_equity_ltp(conn, security_id)`

Return two-decimal float LTP using `NSE_EQ` for a cash share/ETF ID.
Performs the retried `get_ltp` request and inherits its errors and freshness
limits.

### connect_dhan.get_commodity_ltp

`get_commodity_ltp(conn, security_id)`

Return two-decimal float LTP using `MCX_COMM`. Performs the retried `get_ltp`
request and inherits its errors and freshness limits; the value is a quoted
price, not a contract-multiplied P&L amount.

### connect_dhan.get_option_price

`get_option_price(conn, security_id, start, end, interval="min")`

Return the last returned candle's `close` as a two-decimal float for
`NSE_FNO/OPTIDX`. Fetches history with underlying chunk retries; raises if no
rows are returned. This candle-close proxy is not a verified executable fill
and does not include fees or slippage.

### connect_dhan.fetch_equity_data

`fetch_equity_data(conn, security_id, start, end, interval="day")`

Return standard candle DataFrame from `fetch_candles` using `NSE_EQ/EQUITY`.
Network errors and interval limitations are inherited. Default is daily,
unlike most other wrappers. A session's consolidated daily candle may arrive
late; do not assume same-evening availability.

### connect_dhan.fetch_commodity_data

`fetch_commodity_data(conn, security_id, start, end, interval="min")`

Return standard candle DataFrame using `MCX_COMM/FUTCOM`; fetches with the
underlying retries and date/interval limits. Covers only the specified
contract, not a continuous back-adjusted series. No rollover stitching or
physical-unit multiplication is performed.

### connect_dhan.check_dhan_response

`check_dhan_response(response)`

Return `None` for HTTP status below 400; otherwise raise a generic `Exception`
containing status, URL and JSON body (or raw text if JSON decoding fails).
Works on an already obtained response; sends no request. Despite historical
docstring wording, this does not raise `requests.HTTPError`. A 2xx application
error is not detected here. Error text may contain sensitive response data.

### connect_dhan.execution_number

`execution_number(value, name)`

**Execution validation helper.** Convert `value` to float and return it only if
finite and greater than zero. `name` labels a `ValueError`; invalid conversions
also raise. Pure local operation; it does not enforce integer quantities.

### connect_dhan.execution_tick

`execution_tick(security_id, exchange_segment)`

**Strict execution helper.** Return a positive finite tick in rupees from exactly
one master row matching ID, exchange and segment. May download the master;
unsupported segments raise `KeyError`, missing/duplicate rows or invalid ticks
raise `ValueError`. Unlike `get_tick_size`, never guesses a fallback.

### connect_dhan.round_execution_price

`round_execution_price(price, tick, side)`

Return float price rounded **up** to the tick for `"BUY"` or **down** for
`"SELL"` using Decimal arithmetic. Requires positive finite price/tick and an
exact uppercase side; invalid inputs raise. Pure local operation. A small SELL
price can round to zero; this function alone is not a full order validator.

### connect_dhan.get_depth

`get_depth(conn, security_id, exchange_segment)`

Return the instrument's quote dictionary, including `depth` and circuit limits,
from the marketfeed quote endpoint. Sleeps to space requests by at least one
second within this module/process; no cross-process rate coordination or lock.
Uses a three-second timeout, requires response `status == "success"` and rejects
elapsed response time over three seconds. No automatic retry. Elapsed-time
checking does not prove exchange-timestamp freshness.

### connect_dhan.depth_limit

`depth_limit(quote, side, quantity, tick, max_price=None, entry=False, best_price_only=False, check_spread=True)`

**Pure execution-pricing helper.** Return `(limit_price, reference_price)`.
Requires positive integral quantity, `"BUY"`/`"SELL"`, positive tick and a quote
with `depth.buy`/`depth.sell` rows containing `quantity`/`price`, plus positive
`lower_circuit_limit`/`upper_circuit_limit`. Sorts bids descending, asks ascending;
ignores zero-quantity levels and rejects invalid quantities, missing sides or
crossed books.

Normally reference is the opposite-side level covering cumulative quantity,
or the last displayed level if depth is insufficient. Adds a 0.1% cushion
(BUY up, SELL down), rounds to tick, then clamps within circuit bounds and
optional `max_price` (an upper cap, including for SELL).
`best_price_only=True` instead references the best opposite-side price with
no cushion. With `entry=True`, rejects spread over 0.75% by default, insufficient
depth and a capped price below reference. Set `check_spread=False` to disable
only the spread rejection; crossed or missing books and the other entry checks
still fail. Raises on invalid data or unusable bounds;
it neither fetches a quote nor guarantees a fill.

### connect_dhan.prepare_order

`prepare_order(conn, security_id, side, quantity, exchange_segment, product_type, entry_deadline=None, correlation_id=None, best_price_first=False, check_spread=True)`

**Network operation, but does not submit an order.** Read strict master tick and
live depth, then return `{"request": body, "policy": policy}` for a LIMIT/DAY
order. Inputs include full broker quantity and exact side/product/segment.
No lot-multiple, freeze-quantity, funds or portfolio-risk checks are added.

Supplying `entry_deadline` activates long-BUY-only entry checks. Deadline is
finite future Unix seconds, capped at 15 seconds from preparation; policy adds
a tick-rounded upper price cap 1% above the original depth reference.
`correlation_id` is copied into the request when supplied; no uniqueness check
is made. `best_price_first=True` starts at best opposite-side price and adds
`initial_quote` with receipt time, quote, remaining quantity, reference and
limit price. It also sets the policy flag used by `wait_for_fill`.
`check_spread=False` disables the spread rejection during both preparation and
subsequent repricing; the choice is saved in `policy`. Older saved policies
without this flag retain the spread check.
Validation/depth/master errors propagate. Returned request contains client ID;
do not publish it.

### connect_dhan.submit_prepared_order

`submit_prepared_order(conn, prepared)`

**Places a real order.** POST `prepared["request"]`, returning raw response JSON
(normally order ID/status, not fill evidence). Rejects a passed policy deadline
before sending; otherwise uses a three-second timeout and HTTP check, with no
retry. Does not re-fetch depth or independently validate arbitrary prepared
payloads. A lost response is ambiguous: reconcile rather than resubmit.

### connect_dhan.order_by_correlation

`order_by_correlation(conn, correlation_id)`

Fetch the broker order associated with a correlation ID; return a dictionary.
Unwraps a one-element list; any other list length or missing `orderId` raises
`ValueError`. Uses a three-second GET, no retry. Intended for reconciliation:
an unresolved result is not permission to submit another order.

### connect_dhan.cancel_order

`cancel_order(conn, order_id)`

**Changes a live order.** Send a three-second DELETE request and return `None`
after checking HTTP status; no retry or terminal-status confirmation. Fills can
race cancellation and partial fills remain positions. Poll/reconcile the order
afterward; a successful HTTP response alone is not proof it is cancelled.

### connect_dhan.send_order

`send_order(conn, security_id, transaction_type, quantity, exchange_segment, product_type)`

**Places a real order.** Prepare depth-priced LIMIT/DAY request then submit it;
return raw submission JSON, not a filled trade. Side is `"BUY"`/`"SELL"` and
quantity is total broker units, not an automatic lot count conversion.
Preparation reads master/depth; submission has no automatic retry, correlation
ID or long-entry deadline in this convenience path. Verify current lot/freeze
limits yourself. An ambiguous timeout must be reconciled, not blindly retried.

### connect_dhan.place_order

`place_order(conn, security_id, transaction_type, quantity)`

**Live-order convenience wrapper.** Return `send_order` submission JSON for
`NSE_FNO/MARGIN`. Quantity is option units (lots times current lot size), not
number of lots. Inherits depth/network errors and no-submission-retry policy;
does not validate the contract or wait for fills.

### connect_dhan.place_equity_order

`place_equity_order(conn, security_id, transaction_type, quantity)`

**Live-order convenience wrapper.** Return `send_order` submission JSON for
`NSE_EQ/CNC` delivery. Quantity is shares/ETF units. Inherits network errors
and no-submission-retry policy; not a paper-trading switch or fill confirmation.

### connect_dhan.place_commodity_order

`place_commodity_order(conn, security_id, transaction_type, quantity)`

**Live-order convenience wrapper.** Return `send_order` submission JSON for
`MCX_COMM/MARGIN`. This implementation treats quantity as lots where master
`LOT_SIZE` is one; verify current contract specifications and physical
multiplier rather than borrowing NSE option quantities. Inherits network
errors/no-submission-retry policy and does not wait for fills.

### connect_dhan.get_order

`get_order(conn, order_id)`

Return raw broker order JSON, unwrapping exactly one list item when needed.
Three-second GET, no retry; other list lengths raise `ValueError`.
Useful fields include `orderStatus`, `filledQty`, `averageTradedPrice`.
Does not validate order identity/quantities; use `checked_order` when needed.

### connect_dhan.modify_order

`modify_order(conn, order_id, order_type, quantity, price=0)`

**Changes a live order.** PUT total order quantity (including already-filled
units), type, price and DAY validity, returning raw JSON. Uses three-second
timeout, no decorator retry; HTTP rejection prints the request body then raises.
Do not pass remaining quantity as total quantity.

Historical broker observations in source found LIMIT-to-MARKET modifications
rejected and empty `legName` invalid, so execution reprices LIMIT orders and
omits that field. These are implementation caveats, not promises about every
current segment. This helper itself does not validate side, tick or quantity
identity; poll after modification.

### connect_dhan.note_if_market_became_limit

`note_if_market_became_limit(order)`

**Legacy diagnostic.** Print a message and return `None` whenever the supplied
dictionary has `orderType == "LIMIT"`. No network. It does not actually know
whether the caller originally sent MARKET; using it on a normal LIMIT order
produces a misleading rewrite message. Current submission already uses LIMIT.

### connect_dhan.chase_the_order

`chase_the_order(conn, order, attempt, step)`

**Internal compatibility entry point; can modify live orders.** Fetch and
validate order against the supplied snapshot, return immediately if terminal,
otherwise get strict tick/depth and price remaining quantity. Re-fetch before
modifying a PENDING/PART_TRADED order using the **whole** quantity.
`attempt` and `step` are ignored, not widening controls. Returns the last
pre-modification snapshot, not confirmation of the new price/fill. Network and
validation errors propagate; no replacement order is placed.

### connect_dhan.warn_order_never_filled

`warn_order_never_filled(order, slack_channel)`

**Internal notification helper.** Return `None`; print and attempt Slack warning
using order fields and environment `slack_token`. Channel uses argument,
environment then built-in fallback. Slack errors are printed and suppressed.
Does not cancel orders or reconcile positions, and warning wording can apply
to a partially filled order too.

### connect_dhan.checked_order

`checked_order(conn, order_id, expected=None)`

**Execution validation helper.** Fetch and return broker order dictionary only
after validating matching ID, integer positive quantity, integer filled quantity
within bounds, valid side and recognized status. Recognized statuses are
TRANSIT, PENDING, PART_TRADED, TRADED, REJECTED, CANCELLED, EXPIRED.
TRADED must be fully filled; any positive fill needs a positive finite average
price. Optional expected snapshot fixes quantity, side, security ID and segment
and prohibits decreasing filled quantity. Network errors or violated checks
raise; a valid nonterminal snapshot is still not a fill.

### connect_dhan.save_execution_quote

`save_execution_quote(collection, key, snapshot)`

**Internal Mongo audit helper.** Return `None`. With `collection=None`, do
nothing; otherwise insert `{"order_key": key, **snapshot}` and require
acknowledgement. Database errors propagate; unacknowledged writes raise
`RuntimeError` to stop repricing. Avoid an `order_key` key inside `snapshot`,
which conflicts with the explicit keyword during dictionary construction.

### connect_dhan.wait_for_fill

`wait_for_fill(conn, order_id, tries=5, delay=1, chases=3, step=0.003, slack_channel=None, prepared=None, audit_collection=None, audit_id=None)`

**Live monitoring/repricing operation.** Return a validated broker order
dictionary, terminal **or nonterminal**. Checks identity against
`prepared["request"]` if supplied, else the first fetched order.
The legacy `tries`, `delay`, `chases`, `step` parameters are accepted but ignored;
nondefault values only print a warning. Monitoring uses a roughly ten-second
monotonic budget and two-second sleeps, plus request/notification time, not
a strict wall-clock timeout.

Reprices the same PENDING/PART_TRADED order from depth, using remaining quantity
for pricing but whole quantity for modification. Never submits a replacement.
Prepared best-price-first policy waits up to three seconds initially. Prepared
long-entry deadline triggers cancellation; depth-pricing `ValueError` also
cancels such entries. Without that policy, pricing errors propagate and generic
BUY orders are **not** automatically cancelled (they may close shorts).

Optional audit collection and ID must be supplied together; snapshots are
inserted before repricing and audit failures propagate. At monitoring timeout,
prints/posts an unfilled warning and returns the last snapshot: the order can
remain working, including after a cancellation request. The caller still owns
reconciliation and all partial positions. Terminal REJECTED/CANCELLED/EXPIRED
does not mean fully filled. HTTP/validation failures can exit with an order
still live; do not treat exceptions as proof that no position exists.

## connect_definedge

This is the legacy Definedge Integrate adapter, not an interchangeable Dhan
connection object. Every module-level function below has a five-attempt,
5-second initial-delay, exponential-backoff decorator. Several function names
overlap the Dhan adapter but signatures and result shapes differ.

### connect_definedge.login_to_integrate

`login_to_integrate(fresh=False)`

**Authentication operation.** Load dotenv and return `ConnectToIntegrate`.
Requires environment `INTEGRATE_API_TOKEN`, `INTEGRATE_API_SECRET`, `TOTP`.
Normally applies environment session keys `INTEGRATE_UID`, `INTEGRATE_ACTID`,
`INTEGRATE_API_SESSION_KEY`, `INTEGRATE_WS_SESSION_KEY`. Missing session keys
or `fresh == True` triggers TOTP login, writes new keys to process environment
and prints success. Does not persist keys to dotenv or check their expiry.
Failures retry; missing required variables raise `KeyError` (message omits
the TOTP requirement). Fresh login is not a harmless local constructor.

### connect_definedge.fetch_historical_data

`fetch_historical_data(conn, exchange, trading_symbol, start, end, interval="min")`

Return `pd.DataFrame(list(history))` from `IntegrateData.historical_data`, using
supplied exchange, symbol and date bounds. `"day"`/`"min"` select SDK constants;
other intervals leave `tf` undefined and raise. Performs retried SDK/network
access. No candle normalization, sorting, timezone conversion, gap checking or
guaranteed six-column empty schema is added.

### connect_definedge.fetch_ltp

`fetch_ltp(conn, exchange, trading_symbol)`

**Legacy candle-close proxy, not a quote lookup.** Fetch five calendar days of
minute history ending at host-local `datetime.today()` and return its final
close rounded to two decimals. Catches exceptions, prints and implicitly
returns `None`, so its own retry decorator does not retry those swallowed
errors; the inner history call still retries. Can return a stale close.

### connect_definedge.get_option_price

`get_option_price(exchange, trading_symbol, start, end, interval="min")`

Return final history `close` scalar without explicit float conversion or
rounding. Unlike the Dhan function, takes **no conn** and calls login internally.
Uses SDK history for `"min"`/`"day"`; other intervals raise due to undefined
`tf`, and empty/missing close data raise. Network/authentication effects and
nested retry delays apply. It is not an executable-price guarantee.

### connect_definedge.get_index_future

`get_index_future(url="https://app.definedgesecurities.com/public/allmaster.zip", instrument_name="NIFTY")`

Download ZIP, parse its **first archive member** as a headerless master CSV,
and return the nearest future's `TRADINGSYM` scalar for exact symbol,
`SEGMENT == "NFO"` and `INSTRUMENT TYPE == "FUTIDX"`. Expiry parsing uses
`%d%m%Y` without explicit zero-padding and selects strictly after host-local
now; expiry-day midnight is excluded. No eligible rows raise `IndexError`.
HTTP/ZIP/parser failures retry; request has no explicit timeout or cache.

### connect_definedge.load_csv_from_zip

`load_csv_from_zip(url="https://app.definedgesecurities.com/public/allmaster.zip", instrument_name="NIFTY")`

Download and parse the first ZIP member; return a DataFrame of NFO/OPTIDX
rows whose `SYMBOL` exactly equals `instrument_name` (an earlier startswith
filter does not broaden this). Zero-pad expiry strings to eight characters,
parse `%d%m%Y`, sort ascending. Does **not** filter expired contracts.
Bad CSV lines are skipped; invalid dates become NaT. Network/ZIP/parser
errors retry; no explicit HTTP timeout or caching.

### connect_definedge.get_index_option_symbol

`get_index_option_symbol(strike=19950, option_type="PE")`

Return `(trading_symbol, expiry_date)` for the first eligible NIFTY option
after downloading the master. Strike matching is a **substring search on
TRADINGSYM**, not numeric equality; option type uses regex `.str.match`.
Select expiry strictly after host-local now with **no three-day minimum**;
expiry-day midnight is excluded. Prints selection; no match raises by indexing
an empty array and retries. Unlike Dhan, returns neither security ID nor lot
size and has no underlying/min-DTE argument.

## utils

### utils.ist_now

`ist_now()`

Return current naive IST `datetime` by adding fixed UTC+05:30 to current UTC
and dropping timezone metadata. Reads system clock only; independent of host
timezone and DST settings. This is a wall-clock convention, not a UTC datetime
for timezone-aware storage.

### utils.round_to_nearest

`round_to_nearest(x, base=0.05)`

Return `round(base * round(float(x) / base), 2)`. Pure local rounding with
Python's ties-to-even behavior, not directional execution rounding.
Conversion errors/zero base raise; always rounds final output to two decimals,
so unsuitable for arbitrary sub-paise tick precision.

### utils.resample_ohlc_data

`resample_ohlc_data(df, frequency)`

Return resampled DataFrame with `datetime`, `open`, `high`, `low`, `close`,
`volume`: first/max/min/last/sum by pandas frequency, dropping missing OHLC
bins, then rounding OHLC to `0.05`. **Mutates input** by converting datetime
and setting it as index in place. Requires all named columns; invalid
frequency/date/data errors propagate. Uses pandas' default bin alignment,
not exchange-session anchoring; fixed `0.05` rounding is not instrument-aware.

```python
import pandas as pd
from tamingnifty import utils

source = pd.DataFrame({
    "datetime": ["2026-01-01 09:15", "2026-01-01 09:16"],
    "open": [100, 101], "high": [102, 103],
    "low": [99, 100], "close": [101, 102], "volume": [10, 20],
})
bars = utils.resample_ohlc_data(source.copy(), "5min")
assert bars.loc[0, "volume"] == 30
```

### utils.get_public_ip

`get_public_ip()`

Make a real HTTPS request to ipify with ten-second timeout. Return stripped
response text, or `"unknown (<error>)"` after caught exceptions. Does not
validate IP format or call `raise_for_status`, so an HTTP error body can be
returned as if it were an address. Not a local-only startup check.

### utils.get_slack_client

`get_slack_client(token=<redacted source default>)`

The source contains a literal credential-like default, intentionally not
reproduced here. **Use an explicit caller-supplied token.** Construct and
return Slack SDK `WebClient`; construction does not post or verify credentials.
SDK construction/configuration errors can propagate. This constructor is also
evaluated while importing utils to create `notify`'s default client.

### utils.exception_detail

`exception_detail(e)`

Return a compact string containing exception type/message and traceback
location, plus the deepest frame outside a path containing `"site-packages"`
when different. With no traceback, return just type/message. No printing or
network effects; nested formatting helper is not a module-level API.
This path heuristic is not guaranteed to identify user code, and unlike the
motivating docstring discussion, the function does **not** extract HTTP
response bodies or print a full traceback. Exception text may contain secrets:
sanitize before publishing.

### utils.notify

`notify(message="This is just a stupid notification!", slack_channel="niftyweekly", slack_client=get_slack_client())`

Print one timestamped line with flush, then call `post_to_slack`; return `None`.
Timestamp uses host-local `datetime.now()`, not `ist_now()`. Supply explicit
message, channel and a caller-created client; do not rely on the import-time
default client, whose credential-like constructor default is intentionally
not reproduced above. Prints even if Slack fails; post exceptions propagate
after retries. Channel argument is the bare name, without leading `#`.

### utils.post_to_slack

`post_to_slack(message, slack_channel, slack_client)`

**Internal notification helper; prefer `notify`.** Call
`slack_client.chat_postMessage` with `"#" + slack_channel`, supplied text and
fixed bot display fields. Return `None`, discarding the Slack response.
Retries five times on exceptions with 5-second initial/exponential delays;
ambiguous delivery failures can cause duplicate posts. No input/channel
normalization or credential validation.

## tick_streamer

`TickStreamer` wraps the optional Definedge `IntegrateWebSocket`, not Dhan.
There are no module-level functions in this module. Methods below include
constructor and SDK callbacks; all require an instance as `self`.
The shared latest-value attributes have no locks, timestamp/freshness checks
or copy-on-read. There is no stop/join/reconnect API supplied by this wrapper.
No live reliability guarantee is made for callback delivery.

### tick_streamer.TickStreamer.__init__

`__init__(self, conn, exchange, symbol)`

Construct a streamer for an existing connection, exact segment/exchange and
trading symbol. Resolve token from `conn.symbols`, construct SDK WebSocket,
attach callbacks, initialize `latest_tick={}`, `latest_ltp=None`,
`on_tick_callback=None`, `_thread=None`. Returns `None` (normal constructor);
does not call connect/start itself. Token lookup prints and raises if absent;
SDK constructor errors propagate.

### tick_streamer.TickStreamer._get_token_for_symbol

`_get_token_for_symbol(self, exchange, symbol)`

**Internal lookup helper.** Return `(exchange, token)` for the first
`conn.symbols` entry matching `segment` and `trading_symbol`. Prints the
resolved token or an error. Missing/falsy token raises generic `Exception`;
malformed symbol records can raise `KeyError`. No download, refresh or
duplicate-match validation is performed here.

### tick_streamer.TickStreamer._on_login

`_on_login(self, iws)`

**Internal SDK callback, network operation.** Print and subscribe `iws` to
`[self.token]` using `conn.SUBSCRIPTION_TYPE_TICK`; return `None`.
Subscription exceptions are not caught. A subscription request is not
confirmation that ticks will arrive.

### tick_streamer.TickStreamer._on_tick_update

`_on_tick_update(self, iws, tick)`

**Internal SDK callback.** Store the supplied tick object directly; if it has
`"lp"`, convert to float, replace `latest_ltp` and print. If a user function is
assigned to `on_tick_callback`, call it as `(latest_ltp, tick)` even when this
tick lacks `lp` (price can then be old or `None`). Returns `None`.
`iws` is unused. Bad price conversion or user callback errors propagate to the
SDK callback context; there is no isolation, copy or symbol validation.

### tick_streamer.TickStreamer._setup_callbacks

`_setup_callbacks(self)`

**Internal wiring helper.** Assign the bound `_on_login` and `_on_tick_update`
methods onto the SDK socket; return `None`. Overwrites any callbacks previously
assigned to those SDK attributes; does not connect or subscribe.

### tick_streamer.TickStreamer.start

`start(self)`

**Starts network activity.** Create/start a daemon thread running
`self.iws.connect(ssl_verify=False)`; return `None`. TLS certificate verification
is explicitly disabled by this implementation. No wait-for-login/first-tick
confirmation, thread-error forwarding or already-started guard is provided;
repeated calls can launch multiple threads. Daemon thread does not keep the
process alive.

### tick_streamer.TickStreamer.get_latest_ltp

`get_latest_ltp(self)`

Return cached float LTP, initially `None`, without network access.
No freshness guarantee, waiting or synchronization; a later tick without
`lp` leaves the previous price in place.

### tick_streamer.TickStreamer.get_latest_tick

`get_latest_tick(self)`

Return cached tick object, initially `{}`, without network access or copying.
Caller mutation changes the cached object; concurrent updates and stale ticks
are not guarded. Reading it does not prove successful connection/subscription.

## Technical analysis: `tamingnifty.ta`

These are the current implementations, not claims of equivalence to a vendor's
charts or proof of a profitable signal. The runnable README example is the
recommended starting point for Renko and P&F. Percentage sizes are supplied as
`0.1` for 0.1%, not as decimal fractions. Price steps round to two decimals;
callers must reject empty, nonfinite, nonpositive and zero-step inputs before
calling the converters. A zero rounded step can make a conversion loop forever.
Use sorted data and `.copy()`; no converter automatically filters live candles.
Unless stated otherwise, indicator inputs use lower-case column names.

### ta.atr

`atr(DF, period)`

Use for a volatility series from `high`, `low`, `close`. Copies the frame and
returns an index-aligned Series named `ATR`, rounded to two decimals. True range
uses the maximum of high-low and previous-close gaps with `skipna=False`, so the
first missing previous close stays missing. Smoothing is pandas
`ewm(com=period, min_periods=period).mean()` with default adjustment; it is not
an assertion of exact Wilder/TradingView equivalence. Does not mutate `DF`.

### ta.supertrend

`supertrend(df, period, multiplier)`

Adds ATR-derived trend bands and returns the same mutated OHLC frame with `ST`
and `signal` (`Bullish`/`Bearish`). Intermediate bands/ATR are removed before
return. Use a clean zero-based RangeIndex: the implementation uses integer
labels and chained assignment. Warm-up `ST` values can remain zero; do not
treat every initial row as a fully initialized signal. Requires high/low/close.

### ta.ema_channel

`ema_channel(df, period=21)`

Mutates a high/low/close frame with `ema_low`, `ema_high` and `trend`. Both EWMs
use **fixed `com=10`**; `period` controls only `min_periods`, not the smoothing
span. Drops warm-up rows in place. Trend is Bearish when close is at or below
`ema_low`, otherwise Bullish; it does not require close above `ema_high`.

### ta.tma

`tma(df)`

Adds rounded `ema_20`, `ema_30`, `ema_40` to the supplied close-price frame,
using spans 20/30/40 with `adjust=False`. Returns that same frame. Despite the
name, this means three EMAs, **not a triangular moving average**.

### ta.tma_custom

`tma_custom(df, p1, p2, p3)`

The configurable three-EMA version; positive spans become columns `ema_p1`,
`ema_p2`, `ema_p3` using `adjust=False`, rounded to two decimals. Mutates and
returns the supplied close-price frame; column names are literal, not the
numeric span values.

### ta.rs

`rs(df, nifty)`

Returns a copy of the stock frame with `close = stock_close / nifty_close`
scaled by 10000 if the minimum stock close is below 100, otherwise by 1000;
rounds to two decimals. Drops stock `open`, `high`, `low`, `volume`, so supply
those columns even though the ratio itself uses closes. Alignment is by pandas
**index**, not a datetime merge. Match the benchmark sessions/index explicitly;
missing/zero benchmark prices are not repaired. Not RSI and not an unscaled
percentage return.

### ta.sma

`sma(df, period)`

Adds a two-decimal rolling arithmetic average of `close` in `sma`; returns
the same mutated frame. Initial values are NaN until a full window exists.
`period` must be a valid positive pandas rolling window.

### ta.straddle_chart

`straddle_chart(call_df, put_df)`

Converts each input's `datetime` column in place, inner-joins on that timestamp,
and returns new `datetime`, `open`, `high`, `low`, `close` columns formed by
summing the corresponding call/put fields. Missing timestamps are dropped by
the join; duplicate timestamps can multiply rows. Sum-of-leg highs/lows are
not proof that those leg extremes happened simultaneously.

### ta.up_step

`up_step(brick_size=0.1, value=10.0)`

Returns `round(value * brick_size / 100, 2)`: the next upward percentage-box
increment, not the next price itself. For example, `up_step(1, 100)` is `1.0`.
No positive/finite/zero-step validation is performed internally.

### ta.down_step

`down_step(brick_size=0.1, value=10.0)`

Returns `round(value - round(value * 100 / (100 + brick_size), 2), 2)`.
This is the inverse-percentage downward increment, not simply the same
percentage subtracted from value. For example, `down_step(1, 100)` is `0.99`.
Invalid/zero-progress sizes are not guarded; `brick_size=-100` divides by zero.

### ta.renko

`renko(conn, exchange, trading_symbol, start, end, interval='min', brick_size=0.1, last_high=0, last_low=0, initial_color='', initial_datetime=None)`

Fetches index candles via `connect_dhan.fetch_historical_data`, then constructs
percentage Renko using successive closes. Requires a configured Dhan connection;
unlike `convert_to_renko`, this performs network calls. Returns `datetime`,
`low`, `high`, `color`, `close`, including a seed row. With both seed boundaries
zero it anchors at the first open. Otherwise it uses the supplied boundaries,
color (default green) and optional seed datetime. Callers own seed consistency,
history overlap and candle completion; passing a seed is not automatic
incremental-state persistence. See the README rounding/timestamp warnings.

### ta.convert_to_renko

`convert_to_renko(brick_size=0.1, df=pd.DataFrame)`

Pure DataFrame conversion; **always pass an actual `df` explicitly**. The
default is the DataFrame class, not an empty frame or data source. Requires
nonempty `datetime`, `close`; first `open` is the anchor if the column exists,
otherwise first close. Converts input datetime in place, then returns new
`datetime`, `low`, `high`, `color`, `close` columns.

Every additional up brick steps from the previous high; down bricks step from
the previous low. Reversal therefore skips across the prior brick's width
before a new opposite brick forms (roughly a two-box change from the previous
directional close). There is no configurable reversal argument. Keeps the
initial zero-height green seed and can produce many rows with the same candle
timestamp. Only closes drive expansion; input high/low are not traversed.
No empty-data, finite-price, positive-size or nonzero-step validation is built in.

### ta.convert_to_heiken_ashi

`convert_to_heiken_ashi(df, interval='1min')`

Copies `datetime`, open/high/low/close data, resamples with pandas, and keeps only
buckets whose row count equals `int(Timedelta(interval) / Timedelta('1min'))`.
Calculates HA OHLC plus `color` (green only when close > open) and `body_height`,
rounded to two decimals. Returns a new frame with datetime as a column.

The completeness rule assumes one valid row per source minute: it is **a count
test**, not a wall-clock, deduplication or missing-minute validation. Use fixed
timedelta-compatible intervals; empty input/no qualifying buckets can raise
IndexError. It is not safe to infer real candle completion from row count alone.

### ta.heiken_ashi

`heiken_ashi(conn, exchange, trading_symbol, start, end, interval='3min')`

Fetches minute index candles via Dhan, resamples to the requested pandas
interval and returns HA OHLC, datetime, `color`, `body_height`. Unlike the
offline HA converter it retains a running/incomplete final bucket. Requires
a configured connection; empty resampled data can raise IndexError.

### ta.rsi

`rsi(data, period=14, sma=20)`

Copies a close-price frame; returns it with two-decimal `rsi` and `rsi_sma`.
Initial gains/losses use rolling means with `min_periods=1`; from `period`
onward they use Wilder-style recurrence. RSI's overlay is a rolling SMA with
`min_periods=1`. Flat/zero-loss inputs can yield NaN/100 through division;
the helper does not apply an explicit missing-value policy.

### ta.rsi_avg

`rsi_avg(data, period=14, ema=9)`

Copies a high/low frame and runs the same gain/loss approach on
`(high + low) / 2`, **not close**. Returns added `rsi` and `rsi_ema`, rounded
to two decimals; the overlay uses span `ema` and `adjust=False`. Removes
temporary average/delta fields. Use positive windows and inspect warm-up/NaNs.

### ta.bullish_reversal

`bullish_reversal(brick_size=0.1, value=10.0, reversal=3)`

Returns the projected price after repeatedly adding `up_step` `reversal`
times, rounded to two decimals. Supply a nonnegative integer count and valid
positive price/size. It does not construct a chart or detect a pattern.

### ta.bearish_reversal

`bearish_reversal(brick_size=0.1, value=10.0, reversal=3)`

Returns the projected price after repeatedly subtracting `down_step` the
requested number of times, rounded to two decimals. Same caller-validation
requirements as the bullish projection.

### ta.detect_double_top_buy

`detect_double_top_buy(df)`

Returns a **list of integer row positions**, despite the DataFrame return
annotation. From position 2, flags a green current row whose high exceeds the
green row two positions earlier. Reads `color`, `high`; does not mutate input.
Use P&F columns, not raw candles, and do not assume further pattern validation.

### ta.detect_double_bottom_sell

`detect_double_bottom_sell(df)`

Returns integer row positions where a red row's low is lower than the red row
two positions earlier. Reads `color`, `low`; no input mutation. Return is a
list, not the annotated DataFrame.

### ta.detect_bull_trap

`detect_bull_trap(df)`

Returns integer positions matching green/red/green/red: the second green
makes a higher high and the final red makes a lower low than the preceding red.
Reads `color`, `high`, `low`; no mutation. Identification needs all four columns,
not just the timestamp of the earlier breakout.

### ta.detect_bear_trap

`detect_bear_trap(df)`

Returns integer positions matching red/green/red/green: the second red makes a
lower low and final green makes a higher high than the preceding green. Reads
color/high/low without mutation. Return is a list of positions.

### ta.detect_high_pole

`detect_high_pole(df, brick_size=0.05)`

Returns positions of a **previous green column** (`i-1`) when current data meets
the implementation's higher-high, at-least-five-percentage-box extension and
half-previous-count retracement tests. Requires color/high/count fields.
Size is a percentage and defaults to 0.05, independently of converter defaults.
The prior-column flag is only known after the subsequent retracement; avoid
look-ahead when using it in a backtest.

### ta.detect_low_pole

`detect_low_pole(df, brick_size=0.05)`

Mirror low-pole test: returns prior-red-column positions after lower-low,
five-box percentage extension and half-count retracement checks. Requires
color/low/count. This also marks an earlier column using later information;
returns a list, not a DataFrame.

### ta.detect_bearish_turtle_breakout

`detect_bearish_turtle_breakout(df, n=5)`

Filters red columns and, starting at same-color offset `n`, compares the
current low against the preceding **`n-1` red lows**. Suppresses a breakout
immediately following a breakout in the same-color sequence. Returns **index
labels**, unlike the simple pattern detectors' row positions. Requires
color/low and a unique index; does not mutate input. Use an integer `n >= 2`.

### ta.detect_bullish_turtle_breakout

`detect_bullish_turtle_breakout(df, n=5)`

The green/high counterpart: compares against preceding `n-1` green highs,
starts at same-color offset `n`, suppresses the immediately following
same-color breakout and returns matching **index labels**. No mutation.

### ta.convert_to_pnf

`convert_to_pnf(brick_size=0.1, df=pd.DataFrame)`

Broker-free, close-driven percentage P&F conversion. Pass a real, nonempty
DataFrame with `datetime`, `open`, `close`. Returns one row per column with
`datetime`, `color`, `low`, `high`, `count`, `close` and the eight boolean
pattern fields listed in the README. Green means X/up; red means O/down.
The final **developing column is included**. Flat data can yield an uncolored
zero-count seed. Three-box reversal is hard-coded; no reversal argument exists.

Mutates input datetime and, if **any close < 10**, multiplies input open/close
by 10 internally. Returned high/low are divided back and rounded to two decimals.
There is no general repeated 100x scaling or input validation. The `count` is
the implementation's box counter, not a number of source candles. Timestamps
update on extension and are not immutable column-start times; pattern flags
can require later columns. See the README for zero-step/causality warnings.

### ta.pnf

`pnf(conn, exchange, trading_symbol, start, end, interval='min', brick_size=0.1, last_high=0, last_low=0, initial_color='')`

Fetches index candles via Dhan and runs the P&F builder with the same output
fields/three-box reversal and developing-column behavior. Both zero boundaries
seed from first open; otherwise use explicit high/low/color. Requires a real
connection and can make network requests. Seeded count starts at zero and the
seed datetime comes from supplied history, so saved high/low/color alone are
**not a full previous-column snapshot or a guaranteed incremental replay**.
Caller owns overlap, seed units (especially low-price scaling), validation
and data freshness.

### ta.xo_zone

`xo_zone(df, n=5)`

Adds `xo_zone` to the same P&F frame: sum green `count` minus red `count` over
the last `n` columns. Initial `n-1` values are None. Requires color/count,
a valid positive integer `n` and a zero-based RangeIndex because `.at[i]`
uses integer labels. Returns the mutated frame; not added automatically by
the P&F converters.

## Accounting: `tamingnifty.charges`

These are **estimates from the encoded schedule**, not a contract-note service
or tax advice. The source identifier is `arrow-chart-2026-10-26`; check the
current broker/exchange schedule before relying on it. Turnover is premium for
options and full contract notional for futures. Slippage, funding, rollovers
and settlement/exercise costs are not automatically included.

Every breakdown has `brokerage`, `stt`, `stamp_duty`, `exchange_txn`, `sebi`,
`ipft`, `gst`, `dp`, `total`. Amounts are rupees; rates are applied to turnover.
The current encoded rates are:

| Segment | Brokerage | Sell STT | Buy stamp | Exchange transaction rate |
|---|---|---|---|---|
| NSE options | 20/order | 0.15% | 0.003% | 0.0355299% |
| BSE options | 20/order | 0.15% | 0.003% | 0.0325% |
| NSE futures | min(20, 0.03% turnover) | 0.05% | 0.002% | 0.0018299% |
| NSE equity-classified ETF | 0 | 0.001% | 0.015% | 0.0030699% |
| NSE STT-exempt ETF | 0 | 0 | 0.015% | 0.0030699% |

SEBI is 0.0001% and IPFT is 0.0000001% of turnover in this implementation.
GST is 18% of brokerage + exchange + SEBI; BSE adds IPFT, ETF delivery adds
IPFT and provisional DP to that base. ETF sells add provisional DP 12.50
before GST per call. STT/stamp round to whole rupees, other amounts to paise.
Do not charge DP repeatedly for split orders on one symbol/day without
reconciling the broker's actual debit policy.

### charges.money

`money(value, whole_rupees=False)`

Returns a float rounded with Decimal `ROUND_HALF_UP`, to paise by default or
whole rupees when requested. Use for monetary totals; not exchange tick sizing.
Decimal conversion/quantization errors propagate.

### charges.order_charges

`order_charges(segment, side, price, quantity, exchange='NSE')`

Estimates one executed order; returns the breakdown above. `segment` is
`options`, `futures`, `etf_equity` or `etf_exempt`; side is uppercase BUY/SELL.
Requires finite nonnegative price and a positive whole quantity. BSE is
supported only for options. Unsupported inputs raise ValueError.
No broker or database side effects. Does not decide an ETF's tax category.

### charges.etf_tax_segment

`etf_tax_segment(symbol)`

Returns `etf_exempt` for the encoded exempt set, `etf_equity` for its explicit
equity ETF set. Unknown/case-mismatched symbols raise ValueError rather than
assuming a tax status. The lists are not a complete or automatically updated
exchange universe.

### charges.etf_order_charges

`etf_order_charges(symbol, side, turnover)`

Classifies the symbol and estimates a full-position delivery order using
`price=turnover, quantity=1`. Input is already **total turnover**, not a unit
price. Returns an itemized breakdown, including one provisional DP on a sale.
Use the lower-level function with a verified category when appropriate.

### charges.sum_charges

`sum_charges(*breakdowns)`

Adds all required itemized fields, rounds and recomputes total. Returns a new
breakdown; no mutation/network calls. Missing component keys raise KeyError.
It sums the fields, not the supplied `total` values, so does not validate an
inconsistent input total.

### charges.silver_micro_estimated_order_charges

`silver_micro_estimated_order_charges(side, price, quantity)`

Estimates one Silver Micro execution using the research scenario: Rs20
brokerage, sell-side 0.01% CTT, buy-side 0.002% stamp, 0.0026% exchange
fee, 0.0001% SEBI, a 0.0000001% IPFT proxy and 18% GST on brokerage,
exchange, SEBI and IPFT. `quantity` is a whole number of Silver Micro lots;
the scenario uses one kg per lot. `stt` stores commodity transaction tax.
This is not a verified Dhan tariff or contract-note calculation. Settlement,
financing and slippage are not included. Other instruments and charge
schedules must not be estimated with this function.

### charges.option_spread_charges

`option_spread_charges(quantity, long_entry, long_exit, short_entry, short_exit)`

Estimates four NSE option executions: long BUY/SELL plus short SELL/BUY at
the supplied premiums and **the same quantity**. Returns their summed
breakdown. Useful for an ordinary fully closed four-order spread, not a
partial/multi-fill/eight-order trade or BSE spread without adaptation.

## Trading safeguards: `tamingnifty.safety`

These functions deliberately use Mongo and a caller-supplied notification
callback. They do not create an official exchange holiday feed, flatten broker
positions or make order submission exactly-once. They are **not prerequisites
for the offline chart converters**.

### safety.alert

`alert(notify, message)`

Prints the message with flush and calls `notify(message)`; returns None.
Callback exceptions are printed, not re-raised. Trading failure paths still
stop independently; successful return is not proof Slack delivered the alert.

### safety.validate_calendar

`validate_calendar(document, day)`

Validates a supplied annual NSE calendar dictionary for a `datetime.date`.
Requires `_id=NSE:<year>`, exchange/year, status verified, full Jan1-Dec31
coverage, parseable `verified_at`, nonempty HTTPS source URLs and a nonempty
list of unique in-year `{date, name}` holidays. Returns a mapping from date
objects to holiday names. Invalid/missing fields raise; no network/write.
This validates schema, **not the truth or completeness of a source calendar**.

### safety.check_trading_day

`check_trading_day(client, notify, now=None)`

Reads `client['Bots']['exchange_calendars']` for the current IST year and
validates it. Default clock is naive IST; an explicit `now` must also be a naive
IST datetime. Returns True for a regular weekday not listed as a holiday,
False with a notification for holidays/weekends. Missing/invalid/unavailable
calendars alert and raise; there is no missing-year fallback. Normal jobs
skip special weekend sessions too. Call before login, data or trading writes.

### safety.assert_execution_clear

`assert_execution_clear(client, scope, notify)`

Reads `Bots.execution_guards` by stable strategy/account scope. Returns None
if absent or status `complete`; alerts and raises on any other status or read
failure. Call outside an outer retry/polling exception handler. Does not clear,
expire or reconcile a pending/failed attempt.

### safety.guarded_execution

`guarded_execution(client, scope, notify, action, function, *args, **kwargs)`

Atomically reserves an acknowledged per-scope pending attempt before calling
the function, then acknowledges completion before returning its result.
Uses the unique Mongo `_id` and attempt token to reject an unresolved or
concurrent reservation. Wrap the **entire order plus persistence operation**,
not only a single broker request.

On Exception/SystemExit it tries to persist failed status, alerts and raises
SystemExit(1), so ordinary `except Exception` polling cannot silently retry it.
A process crash can leave pending status, which intentionally blocks restart.
No TTL, automatic retry, cancellation or rollback exists. Complete guards may
be reused for later operations; this is not exactly-once execution. Use a
single executor per account/strategy and recheck position state inside the
guard when concurrency matters. Reconcile broker working orders/partial fills,
positions and Mongo state before an operator manually marks that exact guard
complete; retain attempt/error evidence.
