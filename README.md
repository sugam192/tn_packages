# TamingNifty

A Python library for percentage Renko and Point & Figure (P&F) charts,
technical indicators, Dhan execution, optional Definedge integration and
trading infrastructure.

**Public documentation:** [PyPI project page](https://pypi.org/project/tamingnifty/).
The published page contains this guide **and the complete API reference below
it**, including broker, utility, charge and safety functions. In a source
checkout, the reference is also stored as `API_REFERENCE.md`.
Version 2.4.5 adds an opt-out for the entry spread check. The
default remains unchanged, and the choice is saved with the order policy for
repricing and recovery. It does not remove other entry protections.

## Installation

```bash
python -m pip install tamingnifty
```

Python 3.11 is used by the release checks. Metadata permits Python 3.7+, but
compatibility with every currently resolved dependency on older Python versions
is not verified. Definedge-only modules additionally require:

```bash
python -m pip install "tamingnifty[definedge]"
```

## Start here: convert your own candles without a broker

The two offline conversion functions need no account, token, MongoDB, Slack
connection or network request:

- `ta.convert_to_renko(brick_size=..., df=...)`: one row per Renko brick,
  **including an initial zero-height seed row**.
- `ta.convert_to_pnf(brick_size=..., df=...)`: one row per P&F column,
  including the final developing column and eight pattern flags.

### Runnable Renko and P&F example

These are synthetic prices, not a backtest or a trading recommendation.
`brick_size=1` means **1%**, not one rupee. For 0.1%, pass `0.1`, not `0.001`.
Both algorithms use successive **close prices**, with the first open as the
anchor; they do not reconstruct an intrabar high/low path.

```python
import pandas as pd
from tamingnifty import ta

candles = pd.DataFrame({
    "datetime": pd.date_range("2026-01-05 09:15", periods=7, freq="min"),
    "open": [100, 100, 102, 105, 101, 97, 100],
    "close": [100, 102, 105, 101, 97, 100, 104],
})
candles = candles.sort_values("datetime").reset_index(drop=True)

renko = ta.convert_to_renko(brick_size=1, df=candles.copy())
print("Renko: last three rows")
print(renko.tail(3)[["datetime", "low", "high", "color", "close"]].to_string(index=False))

pnf = ta.convert_to_pnf(brick_size=1, df=candles.copy())
print("P&F: all columns")
print(pnf[["datetime", "color", "low", "high", "count", "close"]].to_string(index=False))
```

Expected output:

```text
Renko: last three rows
           datetime    low   high color  close
2026-01-05 09:21:00 100.00 101.00 green 101.00
2026-01-05 09:21:00 101.00 102.01 green 102.01
2026-01-05 09:21:00 102.01 103.03 green 103.03
P&F: all columns
           datetime color    low   high  count  close
2026-01-05 09:17:00 green 100.00 104.06      4 104.06
2026-01-05 09:19:00   red  97.06 103.03      7  97.06
2026-01-05 09:21:00 green  98.03 103.03      6 103.03
```

### Inputs, outputs and interpretation

| Detail | Renko | P&F |
|---|---|---|
| Required input columns | `datetime`, `close`; first `open` is used when present, otherwise first `close` | `datetime`, `open`, `close` |
| Box size | Percentage, compounded from the previous box boundary | Same percentage stepping; hard-coded three-box reversal |
| Output columns | `datetime`, `low`, `high`, `color`, `close` | Those columns plus `count` and eight boolean pattern columns |
| Direction | `green` up, `red` down | `green` X column, `red` O column; a flat seed can have an empty color |
| Final state | Includes its seed row; no separate partial-brick row | Includes the current developing column, not just completed columns |
| `close` | Up brick's high or down brick's low | Up column's high or down column's low |
| Timestamp | Source candle that produced the brick; several bricks can share one timestamp | Updated when a column extends; do not treat it as an immutable column-open time |

P&F pattern columns are `double_top_buy`, `double_bottom_sell`, `bull_trap`,
`bear_trap`, `high_pole`, `low_pole`, `bullish_turtle`, `bearish_turtle`.
They are algorithmic flags, not independently validated trading signals.
Pole flags are attached to a previous column after a later retracement, so
their row timestamps alone do not prove when a signal became available.

### Important chart limitations

- Pass a **non-empty**, chronologically sorted DataFrame with finite positive
  prices and consistent timestamp conventions. The converters do not sort,
  deduplicate or fully validate inputs. Pass a copy: datetime conversion and
  P&F's low-price scaling modify the supplied frame.
- Percentage steps are rounded to two decimals. They are not fixed-point
  bricks, an ATR size, exchange tick rounding or a promise to match another
  charting platform's anchoring/rounding.
- Choose a positive size that produces **nonzero rounded up/down steps at
  every price reached**. Zero/negative sizes, tiny sizes or very low prices
  can produce zero-progress loops; these inputs are not guarded internally.
  P&F's internal 10x scaling when any close is below 10 reduces some rounding
  effects but does not make all low-price inputs safe.
- Neither converter filters incomplete source candles or proves data freshness.
  Several synthetic bricks from one candle do not reveal their actual intrabar
  times or executable fill prices.
- A P&F column may change as more data arrives. Recompute prefixes or retain
  signal-availability timestamps for causal research; do not backtest final
  column/pattern labels as though all were known at the start of the column.
- Use the pure `convert_to_*` functions for CSV/DataFrame input. The separate
  `renko(...)` and `pnf(...)` functions **fetch index data through Dhan** first;
  they require a configured connection and are not the offline examples above.

Other chart helpers include Heiken Ashi conversion, ATR, Supertrend, moving
averages, RSI, relative strength and P&F pattern/XO-zone calculations. Exact
signatures, return shapes and implementation-specific caveats are in the API
reference appended to the public PyPI description.

## Charge accounting (2.3.2)

`tamingnifty.charges` provides plain functions for itemized NSE executed-order
charges and four-order option spreads. Derivative rates follow the user's
Arrow charge chart dated 2026-10-26; the canonical source and product-specific
exceptions live in the vault's `Trading Charges - Arrow and Dhan Reference`.
Amounts are estimates until reconciled with contract notes. Do not confuse
option premium turnover with futures notional, or apply a stock-delivery tax
schedule to ETFs without checking their tax classification.

Run `python test_charges.py` to verify rates, tax sides and rounding.

## License

MIT

## Shared trading safety (2.4.3)

All NSE bot jobs call `safety.check_trading_day(client, notify)` before broker
login, data requests, state writes or polling. It reads `Bots.exchange_calendars`
with `_id=NSE:<IST year>`. Weekends and listed holidays exit normally, including
weekend Muhurat sessions (these jobs are for regular sessions only).
Missing, invalid, unverified or unavailable calendars alert and raise: no trading.
This is the NSE equity/F&O calendar, not an MCX/currency/settlement calendar.

The reviewed `nse_calendar_2026.json` includes 16 weekday closures, cross-checked
against the linked Groww and Angel One calendars. Direct NSE circular retrieval
timed out; the snapshot records that limitation. Review official ad-hoc notices,
amend the JSON and re-seed as needed. A fresh verified annual calendar is required
before January 1; there is deliberately no next-year fallback.

From this repository, preview without connecting to Mongo:

```powershell
python seed_exchange_calendar.py nse_calendar_2026.json --env-file <bot-env-file>
```

After review, add `--apply` to replace this single year document using
`CONNECTION_STRING`. The script validates full-year coverage, requires an
acknowledged write and checks exact read-back. Do not put credentials in the JSON.

For executors without their own durable intent recovery, use
`safety.assert_execution_clear(client, scope, notify)` outside startup polling
catches, then wrap the entire order-and-persistence operation with
`safety.guarded_execution(client, scope, notify, action, function, *args)`.
Use one stable scope per strategy/account. It atomically reserves a unique
`Bots.execution_guards` document before submission, requires acknowledged
completion, and exits nonzero on any operation exception or SystemExit.
Failed or crash-pending attempts block restarts and concurrent execution.
This prevents blind retries; it does not roll back orders, flatten positions,
guarantee alert delivery or replace broker reconciliation.

Never auto-expire/delete a guard. On an alert, inspect broker order IDs,
working/partial fills, actual positions and Mongo order/trade records. Cancel or
manage outstanding orders/positions and repair accounting first. Only after
manual reconciliation may an operator change that exact guard's `status` to
`complete`, retaining its attempt/error fields and recording the reconciliation.
The next operation gets a new attempt ID. Interceptor retains its existing
persisted-intent recovery rather than an incompatible blanket latch.

Offline checks: `python -m pytest test_safety.py -q`.

## Depth-based Dhan execution (2.4.0)

### Opt-in best-price-first and quote audit (2.4.2)

`prepare_order(..., best_price_first=True)` starts BUY at the best ask and
SELL at the best bid, without the depth cushion. Existing entry spread,
displayed-size, fixed price-cap and deadline checks still apply. Persist the
whole returned object before submission: `initial_quote` contains the full
broker quote (all returned bids/asks and quantities), receipt time, selected
reference, requested quantity and limit price.

`wait_for_fill` gives that order up to three seconds before depth repricing,
limited by an entry deadline. A remaining quantity then uses the existing
depth-priced, 0.1%-cushioned policy; it modifies the same total-quantity order,
never submits another one. BUY entry caps/deadlines remain in force; SELL has
no fixed floor and still prioritizes closure over a bounded price.

Pass `audit_collection` and `audit_id` together to `wait_for_fill` to insert
each subsequent full quote and pricing evaluation as a separate audit document
before any modification. Records include `order_key`, broker `order_id`, receipt time,
remaining/filled quantity, previous order price and candidate reference/limit,
or the validation error. This avoids unbounded arrays inside an order document.
A candidate may be unused if a fill races the quote.
Audit-write failures stop repricing and surface to the caller; the existing
order can remain working. Broker fetch failures cannot supply a depth
snapshot and still surface explicitly. REST provides only its returned
depth levels, not the entire exchange order book or proof of source freshness.

Only opted-in consumers use best-price-first. Existing generic callers keep
their execution behavior. Neither this policy nor a lower sell limit guarantees
a fill, a better fill or a bounded loss.

All Dhan order wrappers now submit explicit LIMIT orders from current bid/ask
depth, not MARKET orders subject to broker conversion. BUY uses asks; SELL uses
bids. Prices include a 0.1% cushion, rounded outward with Decimal to the exact
instrument tick and clipped to current quote circuit bounds. Missing ticks,
invalid/empty/crossed books and slow quote responses fail explicitly.
Generic orders may take a partial fill when displayed depth is insufficient.

`wait_for_fill` checks and reprices the same order roughly every two seconds for
a ten-second monitoring batch. It returns the actual terminal or pending state,
never assumes a fill, and warns if still pending. Rejection, quote failure and
modification failure must be handled by the caller. Legacy timing/percentage
arguments remain accepted but no longer control execution; non-default values
produce a notice. `chase_the_order` also uses depth rather than escalating LTP.
There is no automatic replacement or indefinite widening.

For long-option entry, use `prepare_order(..., entry_deadline=epoch_seconds,
correlation_id=...)`, persist the returned request/policy **before submission**,
then `submit_prepared_order`. Pass that saved `prepared` value on every
`wait_for_fill` call, including after restart. The entry policy rejects spreads
over 0.75%, requires displayed size, fixes a BUY cap 1% above initial depth price
and expires after fifteen seconds or the supplied deadline, whichever comes first.
It cancels on expiry or invalid/excessive entry pricing, then polls for terminal
status and final filled quantity. Network latency can delay confirmation; a
cancel acknowledgement is not proof of no fill. Callers must abort zero fills
or unwind partial entry fills as appropriate.

Generic BUY orders are not treated as entries: they may close a short or hedge
a spread. Existing call signatures and product/quantity conventions are
preserved. Generic callers retain ownership of pending/partial/rejected orders;
the swing executor adds persisted entry cancellation and partial-unwind policy.
`order_by_correlation` supports recovery after lost submission replies. It does
not make submission idempotent and unresolved lookup must never trigger resubmit.

Quotes are throttled to one request/second **per process**. Coordinate separate
processes sharing an account. Snapshot age is not supplied by this REST depth
response; we reject responses taking over three seconds, but cannot prove that
the broker book itself is current. No fill, maximum exit loss or broker uptime
is guaranteed. An exit has no fixed initial-price floor and may accept substantial
slippage while following bids within exchange bounds.

Offline checks (no broker/Mongo calls):

```powershell
python test_execution.py
python test_execution.py --swing
python test_charges.py
```

`--swing` additionally loads the sibling Interceptor executor for integration checks.
The publish workflow runs library-only checks, so it needs no sibling checkout.
Publication is triggered by a matching version tag. Changing source or
requirements does not update already deployed images. Documentation-only
version 2.4.4 does not require existing consumers pinned to 2.4.3 to redeploy.
Definedge 1.x consumers are not automatically migrated by a library release.