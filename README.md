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

## Depth-based Dhan execution (2.4.0)

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