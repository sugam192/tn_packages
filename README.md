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