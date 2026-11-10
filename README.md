# @TamingNifty

A Python library for Definedge trading strategies.

## Installation

```bash
pip install tamingnifty
```

## Usage

```python
from tamingnifty import ta
```

## Features

- Connect to Definedge API
- Utilities for trading strategies
- Conversion to Renko or Point and Figure charts
- Tick streaming

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

`--swing` additionally loads the sibling swing executor for integration checks.
The publish workflow runs library-only checks, so it needs no sibling checkout.
Publish version 2.4.0 before building consumers pinned to it; changing source or
requirements does not update already deployed images. Definedge 1.x consumers
are not migrated by this release.