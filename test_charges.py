"""Run with python test_charges.py. No broker, Mongo or Slack connection."""
import math

from tamingnifty.charges import (etf_order_charges, etf_tax_segment, money,
                                option_spread_charges, order_charges, sum_charges)


def test_options_four_executed_orders():
    charges = option_spread_charges(65, 100, 80, 200, 150)
    assert charges == dict(brokerage=80.0, stt=28.0, stamp_duty=0.0,
                          exchange_txn=12.24, sebi=0.04, ipft=0.0,
                          gst=16.60, dp=0.0, total=136.88), charges


def test_futures_notional_and_sides():
    buy = order_charges("futures", "BUY", 24000, 65)
    sell = order_charges("futures", "SELL", 25000, 65)
    assert buy["stamp_duty"] == 31 and buy["stt"] == 0
    assert sell["stamp_duty"] == 0 and sell["stt"] == 813
    assert buy["total"] == 90.13, buy
    assert sell["total"] == 873.62, sell
    assert sum_charges(buy, sell)["total"] == 963.75


def test_brokerage_cap_and_rounding():
    assert order_charges("futures", "BUY", 100, 1)["brokerage"] == 0.03
    assert order_charges("options", "BUY", 100, 1)["brokerage"] == 20
    assert money(1.005) == 1.01
    assert money(2.5, whole_rupees=True) == 3
    assert order_charges("futures", "BUY", 18936, 65)["gst"] == 7.88


def test_invalid_inputs_do_not_silently_become_zero_costs():
    cases = [("wrong", "BUY", 100, 1), ("options", "bad", 100, 1),
             ("options", "BUY", -1, 1), ("options", "BUY", math.nan, 1),
             ("options", "BUY", math.inf, 1), ("options", "BUY", 100, 0),
             ("options", "BUY", 100, 1.5)]
    for args in cases:
        try:
            order_charges(*args)
        except ValueError:
            continue
        raise AssertionError("Invalid charge input accepted: " + str(args))


def test_itemized_totals_reconcile():
    for segment in ("options", "futures"):
        for side in ("BUY", "SELL"):
            charges = order_charges(segment, side, 23321.55, 65)
            assert charges["total"] == money(sum(
                value for key, value in charges.items() if key != "total"))


def test_etf_stt_exemptions_and_provisional_dp():
    equity_buy = etf_order_charges("PHARMABEES", "BUY", 100000)
    equity_sell = etf_order_charges("PHARMABEES", "SELL", 110000)
    assert equity_buy["stt"] == 0 and equity_buy["total"] == 18.74
    assert equity_sell["stt"] == 1 and equity_sell["dp"] == 12.50
    assert equity_sell["gst"] == 2.88 and equity_sell["total"] == 19.87
    for symbol in ("LIQUIDCASE", "GOLDIETF", "LTGILTBEES"):
        assert etf_order_charges(symbol, "SELL", 1000000)["stt"] == 0
    for symbol in ("MON100", "HNGSNGBEES", "SILVERBEES", "UNKNOWN"):
        try:
            etf_tax_segment(symbol)
        except ValueError:
            continue
        raise AssertionError("Unverified ETF tax status accepted: " + symbol)


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(len(tests), "passed")
