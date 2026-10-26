"""Itemized NSE charges. Source: vault charge reference dated 2026-10-26."""
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite

CHARGE_SCHEDULE = "arrow-chart-2026-10-26"
CHARGE_FIELDS = ("brokerage", "stt", "stamp_duty", "exchange_txn",
                 "sebi", "ipft", "gst", "dp")
# Owner-approved provisional delivery-sale DP fee, before GST.
ETF_DP_FEE = 12.50
STT_EXEMPT_ETFS = {"GOLDIETF", "LIQUIDCASE", "LTGILTBEES"}
EQUITY_ETFS = {
    "PVTBANIETF", "ITBEES", "FMCGIETF", "MOREALTY", "OILIETF", "HDFCSML250",
    "GROWWPOWER", "MID150BEES", "ALPHA", "METALIETF", "INFRAIETF", "NIFTYBEES",
    "JUNIORBEES", "ICICIB22", "MODEFENCE", "PSUBNKBEES", "PHARMABEES",
    "MOCAPITAL", "LOWVOLIETF", "CPSEETF", "BANKBEES", "AUTOBEES",
}


def money(value, whole_rupees=False):
    unit = "1" if whole_rupees else "0.01"
    return float(Decimal(str(value)).quantize(Decimal(unit), rounding=ROUND_HALF_UP))


def order_charges(segment, side, price, quantity):
    """One executed NSE order; price is premium for options, not index notional."""
    price = float(price)
    qty = int(quantity)
    if not isfinite(price) or price < 0 or qty <= 0 or qty != float(quantity):
        raise ValueError("Charges require a finite non-negative price and positive whole quantity")
    if side not in ("BUY", "SELL"):
        raise ValueError("Charge side must be BUY or SELL")
    turnover = Decimal(str(price)) * qty
    if segment == "options":
        brokerage = 20.0
        stt_rate = "0.0015" if side == "SELL" else "0"
        stamp_rate = "0.00003"
        exchange_rate = "0.000355299"
    elif segment == "futures":
        brokerage = min(Decimal("20"), turnover * Decimal("0.0003"))
        stt_rate = "0.0005" if side == "SELL" else "0"
        stamp_rate = "0.00002"
        exchange_rate = "0.000018299"
    elif segment in ("etf_equity", "etf_exempt"):
        brokerage = 0.0
        stt_rate = "0.00001" if segment == "etf_equity" and side == "SELL" else "0"
        stamp_rate = "0.00015"
        exchange_rate = "0.000030699"
    else:
        raise ValueError("Unsupported charge segment: " + str(segment))
    charges = {
        "brokerage": money(brokerage),
        "stt": money(turnover * Decimal(stt_rate), whole_rupees=True),
        "stamp_duty": money(turnover * Decimal(stamp_rate), whole_rupees=True) if side == "BUY" else 0.0,
        "exchange_txn": money(turnover * Decimal(exchange_rate)),
        "sebi": money(turnover * Decimal("0.000001")),
        "ipft": money(turnover * Decimal("0.000000001")),
        "dp": ETF_DP_FEE if segment.startswith("etf_") and side == "SELL" else 0.0,
    }
    # The supplied chart excludes IPFT from its GST base.
    gst_base = sum(Decimal(str(charges[field])) for field in ("brokerage", "exchange_txn", "sebi"))
    if segment.startswith("etf_"):
        gst_base += Decimal(str(charges["ipft"])) + Decimal(str(charges["dp"]))
    charges["gst"] = money(Decimal("0.18") * gst_base)
    charges["total"] = money(sum(charges.values()))
    return charges


def etf_tax_segment(symbol):
    if symbol in STT_EXEMPT_ETFS:
        return "etf_exempt"
    if symbol in EQUITY_ETFS:
        return "etf_equity"
    raise ValueError("ETF tax status is unverified: " + str(symbol))


def etf_order_charges(symbol, side, turnover):
    """Full-position NSE delivery orders; one DP debit per symbol/day on sale."""
    return order_charges(etf_tax_segment(symbol), side, turnover, 1)


def sum_charges(*breakdowns):
    combined = {}
    for field in CHARGE_FIELDS:
        combined[field] = money(sum(charges[field] for charges in breakdowns))
    combined["total"] = money(sum(combined.values()))
    return combined


def option_spread_charges(quantity, long_entry, long_exit, short_entry, short_exit):
    return sum_charges(
        order_charges("options", "BUY", long_entry, quantity),
        order_charges("options", "SELL", long_exit, quantity),
        order_charges("options", "SELL", short_entry, quantity),
        order_charges("options", "BUY", short_exit, quantity),
    )
