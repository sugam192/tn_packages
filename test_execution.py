import importlib.util
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tamingnifty import connect_dhan as dhan

def quote():
    return dict(depth=dict(buy=[dict(price=100, quantity=65)],
                           sell=[dict(price=100.2, quantity=30),
                                 dict(price=100.25, quantity=35)]),
                lower_circuit_limit=50, upper_circuit_limit=150)


def order(status="PENDING", filled=0, side="BUY", price=100.4):
    return dict(orderId="123", orderStatus=status, filledQty=filled, quantity=65,
                securityId="101", exchangeSegment="NSE_FNO", transactionType=side,
                averageTradedPrice=100.25, price=price)


def prepared(deadline=None, cap=101.25, side="BUY"):
    policy = dict(tick=0.05)
    if deadline is not None:
        policy.update(deadline=deadline, max_price=cap)
    return dict(request=order(side=side), policy=policy)


def collection(doc=None):
    docs = {} if doc is None else {doc["_id"]: deepcopy(doc)}
    result = Mock()
    result.name = "orders_nifty_interceptor_test"
    result.database = {"order_quotes_nifty_interceptor_test": Mock()}
    result.find_one.side_effect = lambda query: deepcopy(docs.get(query["_id"]))

    def insert(document):
        assert document["_id"] not in docs
        docs[document["_id"]] = deepcopy(document)
        return Mock(acknowledged=True)

    def update(query, changes):
        document = docs[query["_id"]]
        for field, value in query.items():
            if field == "_id":
                continue
            expected = value.get("$in") if isinstance(value, dict) else [value]
            if document.get(field) not in expected:
                return
        document.update(deepcopy(changes.get("$set", {})))
        for field, value in changes.get("$max", {}).items():
            document[field] = max(document.get(field, value), value)
        for field, value in changes.get("$min", {}).items():
            document[field] = min(document.get(field, value), value)
        return Mock(acknowledged=True, matched_count=1)

    result.insert_one.side_effect = insert
    result.update_one.side_effect = update
    return result, docs


def raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError("Expected " + error_type.__name__)


def test_depth_uses_cumulative_quantity_and_directional_ticks():
    price, reference = dhan.depth_limit(quote(), "BUY", 65, 0.05, entry=True)
    assert reference == 100.25 and price == 100.4
    price, reference = dhan.depth_limit(quote(), "SELL", 65, 0.05)
    assert reference == 100 and price == 99.9
    assert dhan.round_execution_price(100.327, 0.05, "BUY") == 100.35
    assert dhan.round_execution_price(100.327, 0.05, "SELL") == 100.3
    for tick in (0.01, 0.05, 0.1, 0.5, 1, 10):
        for side in ("BUY", "SELL"):
            price = dhan.round_execution_price(100.327, tick, side)
            assert abs(price / tick - round(price / tick)) < 1e-8


def test_invalid_depth_spread_cap_and_exchange_bounds():
    boundary = quote()
    boundary["depth"]["buy"][0]["price"] = 99.625
    boundary["depth"]["sell"] = [dict(price=100.375, quantity=65)]
    assert dhan.depth_limit(boundary, "BUY", 65, 0.05, entry=True)[1] == 100.375
    boundary["depth"]["sell"][0]["price"] = 100.376
    raises(ValueError, dhan.depth_limit, boundary, "BUY", 65, 0.05, entry=True)
    bad = quote()
    bad["depth"]["sell"] = []
    raises(ValueError, dhan.depth_limit, bad, "BUY", 65, 0.05)
    bad = quote()
    bad["depth"]["buy"][0]["price"] = 102
    raises(ValueError, dhan.depth_limit, bad, "BUY", 65, 0.05)
    bad = quote()
    bad["depth"]["sell"][0]["price"] = 102
    bad["depth"]["sell"][1]["price"] = 102
    raises(ValueError, dhan.depth_limit, bad, "BUY", 65, 0.05, entry=True)
    raises(ValueError, dhan.depth_limit, quote(), "BUY", 130, 0.05, entry=True)
    raises(ValueError, dhan.depth_limit, quote(), "BUY", 65, 0.05, 100.2, True)
    bad = quote()
    bad["upper_circuit_limit"] = 100.32
    assert dhan.depth_limit(bad, "BUY", 65, 0.05)[0] == 100.3
    bad["lower_circuit_limit"] = float("nan")
    raises(ValueError, dhan.depth_limit, bad, "SELL", 65, 0.05)


def test_best_price_first_saves_all_depth_and_preserves_entry_cap():
    book = quote()
    book["depth"]["buy"] = [dict(price=98, quantity=100), dict(price=100, quantity=30)]
    for side, expected in (("BUY", 100.2), ("SELL", 100)):
        with patch.object(dhan, "execution_tick", return_value=0.05), \
                patch.object(dhan, "get_depth", return_value=book), \
                patch.object(dhan.time, "time", return_value=1000):
            plan = dhan.prepare_order({"client_id": "test"}, "101", side, 65, "NSE_FNO", "MARGIN",
                                      1010 if side == "BUY" else None,
                                      best_price_first=True)
        assert plan["request"]["price"] == expected
        assert plan["policy"]["best_price_first"] is True
        snapshot = plan["initial_quote"]
        assert snapshot["quote"] == book
        assert snapshot["received_at"] == 1000
        assert snapshot["remaining_quantity"] == 65
        assert snapshot["reference"] == expected
        assert snapshot["limit_price"] == expected
        if side == "BUY":
            assert plan["policy"]["max_price"] == 101.25
            assert plan["policy"]["deadline"] == 1010
        else:
            assert "max_price" not in plan["policy"]


def test_best_price_entry_still_rejects_thin_or_wide_books():
    book = quote()
    raises(ValueError, dhan.depth_limit, book, "BUY", 130, 0.05,
           entry=True, best_price_only=True)
    book["depth"]["buy"][0]["price"] = 98
    raises(ValueError, dhan.depth_limit, book, "BUY", 65, 0.05,
           entry=True, best_price_only=True)


def test_spread_opt_out_keeps_other_entry_checks():
    book = quote()
    book["depth"]["buy"][0]["price"] = 98
    with patch.object(dhan, "execution_tick", return_value=0.05), \
            patch.object(dhan, "get_depth", return_value=book), \
            patch.object(dhan.time, "time", return_value=1000):
        raises(ValueError, dhan.prepare_order, {"client_id": "test"}, "101", "BUY", 65,
               "NSE_FNO", "INTRADAY", entry_deadline=1010, best_price_first=True)
        plan = dhan.prepare_order({"client_id": "test"}, "101", "BUY", 65,
                                  "NSE_FNO", "INTRADAY", entry_deadline=1010,
                                  best_price_first=True, check_spread=False)
    assert plan["request"]["price"] == 100.2
    assert plan["policy"]["check_spread"] is False
    assert plan["policy"]["max_price"] == 101.25
    assert plan["policy"]["deadline"] == 1010
    raises(ValueError, dhan.depth_limit, book, "BUY", 130, 0.05, entry=True, check_spread=False)
    raises(ValueError, dhan.depth_limit, book, "BUY", 65, 0.05, 100.2, True, check_spread=False)
    book["depth"]["buy"][0]["price"] = 103
    raises(ValueError, dhan.depth_limit, book, "BUY", 65, 0.05, entry=True, check_spread=False)
    raises(ValueError, dhan.depth_limit, quote(), "BUY", 65, 0.05, entry=True, check_spread="false")


def test_spread_opt_out_survives_saved_policy_and_repricing():
    book = quote()
    book["depth"]["buy"][0]["price"] = 98
    plan = prepared(1010)
    plan["policy"]["check_spread"] = False
    initial = order(price=100.2)
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=True)
    with patch.object(dhan, "get_order",
                      side_effect=[initial, initial, order("TRADED", 65)]), \
            patch.object(dhan, "get_depth", return_value=book), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "modify_order") as modify, \
            patch.object(dhan, "cancel_order") as cancel:
        result = dhan.wait_for_fill({}, "123", prepared=plan,
                                    audit_collection=audit, audit_id="trade:BUY")
    assert result["filledQty"] == 65
    assert audit.insert_one.call_args.args[0]["quote"] == book
    cancel.assert_not_called()
    modify.assert_called_once_with({}, "123", "LIMIT", 65, 100.4)


def test_best_price_fill_gets_three_seconds_without_repricing():
    for side, deadline, expected_wait in (("BUY", 1010, 3), ("SELL", None, 3),
                                          ("BUY", 1002, 2)):
        plan = prepared(deadline, side=side)
        plan["policy"]["best_price_first"] = True
        with patch.object(dhan, "get_order",
                          side_effect=[order(side=side), order("TRADED", 65, side)]), \
                patch.object(dhan.time, "time", return_value=1000), \
                patch.object(dhan.time, "sleep") as sleep, \
                patch.object(dhan, "get_depth") as depth, \
                patch.object(dhan, "modify_order") as modify:
            result = dhan.wait_for_fill({}, "123", prepared=plan)
        assert result["filledQty"] == 65
        sleep.assert_called_once_with(expected_wait)
        depth.assert_not_called()
        modify.assert_not_called()


def test_best_price_partial_reprices_remaining_and_persists_before_modify():
    for side in ("BUY", "SELL"):
        plan = prepared(1010 if side == "BUY" else None, side=side)
        plan["policy"]["best_price_first"] = True
        partial = order("PART_TRADED", 30, side, price=100.2 if side == "BUY" else 100)
        snapshots = []
        audit = Mock()

        def save(record):
            snapshots.append(deepcopy(record))
            return Mock(acknowledged=True)

        audit.insert_one.side_effect = save
        key = "trade:" + side

        def modify(conn, order_id, order_type, quantity, price):
            assert quantity == 65
            assert price == (100.4 if side == "BUY" else 99.9)
            assert len(snapshots) == 1
            assert snapshots[0]["order_key"] == key
            assert snapshots[0]["order_id"] == "123"
            assert snapshots[0]["quote"] == quote()
            assert snapshots[0]["remaining_quantity"] == 35
            assert snapshots[0]["limit_price"] == price
            assert snapshots[0]["filled_quantity"] == 30
            assert snapshots[0]["received_at"] == 1000

        with patch.object(dhan, "get_order",
                          side_effect=[partial, partial, partial, order("TRADED", 65, side)]), \
                patch.object(dhan, "get_depth", return_value=quote()), \
                patch.object(dhan.time, "time", return_value=1000), \
                patch.object(dhan.time, "sleep") as sleep, \
                patch.object(dhan, "modify_order", side_effect=modify) as changed:
            result = dhan.wait_for_fill({}, "123", prepared=plan,
                                        audit_collection=audit, audit_id=key)
        assert result["filledQty"] == 65
        assert [call.args[0] for call in sleep.call_args_list] == [3, 2]
        changed.assert_called_once()


def test_audit_failure_blocks_reprice_and_invalid_book_is_saved():
    pending = order(price=99)
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=False)
    with patch.object(dhan, "get_order", return_value=pending), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan, "modify_order") as modify:
        raises(RuntimeError, dhan.wait_for_fill, {}, "123", prepared=prepared(),
               audit_collection=audit, audit_id="trade:BUY")
        modify.assert_not_called()
    bad = quote()
    bad["depth"]["buy"] = []
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=True)
    with patch.object(dhan, "get_order", return_value=order(side="SELL")), \
            patch.object(dhan, "get_depth", return_value=bad), \
            patch.object(dhan, "modify_order") as modify:
        raises(ValueError, dhan.wait_for_fill, {}, "123", prepared=prepared(side="SELL"),
               audit_collection=audit, audit_id="trade:SELL")
        modify.assert_not_called()
    snapshot = audit.insert_one.call_args.args[0]
    assert snapshot["quote"] == bad and "Missing" in snapshot["error"]


def test_audit_arguments_and_best_price_entry_expiry():
    raises(ValueError, dhan.wait_for_fill, {}, "123", audit_id="missing")
    plan = prepared(1001)
    plan["policy"]["best_price_first"] = True
    with patch.object(dhan, "get_order",
                      side_effect=[order(), order(), order("CANCELLED")]), \
            patch.object(dhan.time, "time", return_value=1001), \
            patch.object(dhan.time, "sleep") as sleep, \
            patch.object(dhan, "cancel_order") as cancel, \
            patch.object(dhan, "modify_order") as modify:
        result = dhan.wait_for_fill({}, "123", prepared=plan)
    assert result["orderStatus"] == "CANCELLED"
    assert sleep.call_args_list[0].args == (0,)
    cancel.assert_called_once()
    modify.assert_not_called()


def test_best_bid_then_deeper_exit_when_no_units_fill():
    book = quote()
    book["depth"]["buy"] = [dict(price=98, quantity=100), dict(price=100, quantity=30)]
    with patch.object(dhan, "execution_tick", return_value=0.05), \
            patch.object(dhan, "get_depth", return_value=book), \
            patch.object(dhan.time, "time", return_value=1000):
        plan = dhan.prepare_order({"client_id": "test"}, "101", "SELL", 130,
                                  "NSE_FNO", "MARGIN", best_price_first=True)
    assert plan["request"]["price"] == 100
    pending = dict(order(side="SELL", price=100), quantity=130)
    filled = dict(order("TRADED", 130, "SELL"), quantity=130, averageTradedPrice=99)
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=True)
    with patch.object(dhan, "get_order", side_effect=[pending, pending, pending, filled]), \
            patch.object(dhan, "get_depth", return_value=book), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "modify_order") as modify:
        result = dhan.wait_for_fill({}, "123", prepared=plan,
                                    audit_collection=audit, audit_id="trade:SELL")
    assert modify.call_args.args == ({}, "123", "LIMIT", 130, 97.9)
    assert audit.insert_one.call_args.args[0]["reference"] == 98
    assert result["averageTradedPrice"] == 99


def test_audited_entry_invalid_depth_cancels_without_losing_snapshot():
    book = quote()
    book["depth"]["sell"] = [dict(price=102, quantity=65)]
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=True)
    with patch.object(dhan, "get_order", side_effect=[order(), order("CANCELLED")]), \
            patch.object(dhan, "get_depth", return_value=book), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan, "cancel_order") as cancel, \
            patch.object(dhan, "modify_order") as modify:
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010),
                                    audit_collection=audit, audit_id="trade:BUY")
    assert result["orderStatus"] == "CANCELLED"
    snapshot = audit.insert_one.call_args.args[0]
    assert snapshot["quote"] == book and "spread" in snapshot["error"]
    cancel.assert_called_once()
    modify.assert_not_called()


def test_entry_quote_validation_failure_still_cancels_and_records_error():
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=True)
    with patch.object(dhan, "get_order", side_effect=[order(), order("CANCELLED")]), \
            patch.object(dhan, "get_depth", side_effect=ValueError("Depth response too old")), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan, "cancel_order") as cancel:
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010),
                                    audit_collection=audit, audit_id="trade:BUY")
    assert result["orderStatus"] == "CANCELLED"
    cancel.assert_called_once()
    snapshot = audit.insert_one.call_args.args[0]
    assert snapshot["error"] == "Depth response too old"
    assert "quote" not in snapshot


def test_tick_is_exact_master_lookup_without_fallback():
    master = pd.DataFrame([
        dict(SECURITY_ID="101", EXCH_ID="NSE", SEGMENT="D", TICK_SIZE=5),
        dict(SECURITY_ID="101", EXCH_ID="NSE", SEGMENT="E", TICK_SIZE=10)])
    with patch.object(dhan, "load_instruments", return_value=master):
        assert dhan.execution_tick("101", "NSE_FNO") == 0.05
        assert dhan.execution_tick("101", "NSE_EQ") == 0.1
        raises(ValueError, dhan.execution_tick, "missing", "NSE_FNO")


def test_prepare_entry_fixes_cap_and_deadline_and_sends_limit():
    with patch.object(dhan, "execution_tick", return_value=0.05), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", return_value=1000):
        plan = dhan.prepare_order({"client_id": "test"}, "101", "BUY", 65,
                                  "NSE_FNO", "MARGIN", 1060, "intent")
    assert plan["policy"] == dict(tick=0.05, deadline=1015, max_price=101.25)
    with patch.object(dhan, "execution_tick", return_value=0.05), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", return_value=1000):
        limited = dhan.prepare_order({"client_id": "test"}, "101", "BUY", 65,
                                     "NSE_FNO", "MARGIN", 1008)
    assert limited["policy"]["deadline"] == 1008
    assert plan["request"]["orderType"] == "LIMIT"
    assert plan["request"]["price"] == 100.4
    assert plan["request"]["correlationId"] == "intent"
    flat = quote()
    flat["depth"]["buy"][0]["price"] = 99.9
    flat["depth"]["sell"] = [dict(price=100, quantity=65)]
    with patch.object(dhan, "execution_tick", return_value=0.05), \
            patch.object(dhan, "get_depth", return_value=flat), \
            patch.object(dhan.time, "time", return_value=1000):
        exact = dhan.prepare_order({"client_id": "test"}, "101", "BUY", 65,
                                   "NSE_FNO", "MARGIN", 1060)
    assert exact["policy"]["max_price"] == 101
    assert exact["request"]["price"] == 100.1
    at_cap = quote()
    at_cap["depth"]["buy"][0]["price"] = 100.9
    at_cap["depth"]["sell"] = [dict(price=101, quantity=65)]
    assert dhan.depth_limit(at_cap, "BUY", 65, 0.05,
                            exact["policy"]["max_price"], entry=True)[0] == 101
    at_cap["depth"]["sell"][0]["price"] = 101.05
    raises(ValueError, dhan.depth_limit, at_cap, "BUY", 65, 0.05,
           exact["policy"]["max_price"], True)
    response = Mock(status_code=200)
    response.json.return_value = dict(orderId="123")
    with patch.object(dhan.time, "time", return_value=1016), \
            patch.object(dhan.requests, "post") as post:
        raises(ValueError, dhan.submit_prepared_order, {}, plan)
        post.assert_not_called()
    with patch.object(dhan, "prepare_order", return_value=plan), \
            patch.object(dhan, "submit_prepared_order", return_value={"orderId": "123"}) as submit:
        assert dhan.place_equity_order({}, "101", "BUY", 65)["orderId"] == "123"
        submit.assert_called_once()


def test_partial_reprice_keeps_total_quantity_and_stops_at_fill():
    partial = order("PART_TRADED", filled=30)
    partial["price"] = 100
    with patch.object(dhan, "get_order", side_effect=[partial, partial, order("TRADED", 65)]), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan.time, "sleep") as sleep, \
            patch.object(dhan, "modify_order") as modify:
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010))
    assert result["filledQty"] == 65
    assert modify.call_args.args[3] == 65
    sleep.assert_called_once_with(2)


def test_entry_deadline_cancel_confirms_partial_fill():
    partial = order("PART_TRADED", 30)
    with patch.object(dhan, "get_order", side_effect=[partial, order("CANCELLED", 30)]), \
            patch.object(dhan.time, "time", return_value=1011), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "cancel_order") as cancel, \
            patch.object(dhan, "modify_order") as modify, \
            patch.object(dhan, "warn_order_never_filled"):
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010))
    assert result["orderStatus"] == "CANCELLED" and result["filledQty"] == 30
    cancel.assert_called_once_with({}, "123")
    modify.assert_not_called()
    with patch.object(dhan, "get_order", return_value=partial), \
            patch.object(dhan.time, "time", return_value=1011), \
            patch.object(dhan.time, "monotonic", side_effect=[0, 0, 11]), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "cancel_order") as cancel, \
            patch.object(dhan, "warn_order_never_filled"):
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010))
    assert result["orderStatus"] == "PART_TRADED"
    cancel.assert_called_once()


def test_fill_races_cancel_or_modify_are_not_replaced():
    for expired in (False, True):
        with patch.object(dhan, "get_order", side_effect=[order(), order("TRADED", 65)]), \
                patch.object(dhan, "get_depth", return_value=quote()), \
                patch.object(dhan.time, "time", return_value=1011 if expired else 1000), \
                patch.object(dhan.time, "sleep"), \
                patch.object(dhan, "cancel_order"), \
                patch.object(dhan, "modify_order") as modify:
            assert dhan.wait_for_fill({}, "123", prepared=prepared(1010))["filledQty"] == 65
            modify.assert_not_called()


def test_fixed_cap_failure_cancels_instead_of_chasing():
    moved = quote()
    for level in moved["depth"]["sell"]:
        level["price"] = 101.3
    moved["depth"]["buy"][0]["price"] = 101.2
    with patch.object(dhan, "get_order", side_effect=[order(), order("CANCELLED")]), \
            patch.object(dhan, "get_depth", return_value=moved), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan, "cancel_order") as cancel, \
            patch.object(dhan, "warn_order_never_filled"):
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010))
    assert result["filledQty"] == 0
    cancel.assert_called_once()


def test_rejected_modify_and_order_identity_errors_surface():
    pending = order(price=99)
    with patch.object(dhan, "get_order", return_value=pending), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan, "execution_tick", return_value=0.05), \
            patch.object(dhan, "modify_order", side_effect=RuntimeError("RMS rejected")):
        raises(RuntimeError, dhan.wait_for_fill, {}, "123")
    with patch.object(dhan, "get_order", return_value=dict(pending, securityId="wrong")):
        raises(ValueError, dhan.wait_for_fill, {}, "123", prepared=prepared())
    with patch.object(dhan, "get_order", return_value=order("TRADED", 30)):
        raises(ValueError, dhan.wait_for_fill, {}, "123")
    with patch.object(dhan, "get_order", return_value=order("CANCELLED", 0)):
        raises(ValueError, dhan.checked_order, {}, "123", order("PART_TRADED", 30))


def test_generic_buy_is_not_mistaken_for_entry_and_exit_alerts_pending():
    for side in ("BUY", "SELL"):
        pending = order(side=side)
        with patch.object(dhan, "get_order", return_value=pending), \
                patch.object(dhan.time, "monotonic", side_effect=[0, 11]), \
                patch.object(dhan, "cancel_order") as cancel, \
                patch.object(dhan, "warn_order_never_filled") as warn:
            result = dhan.wait_for_fill({}, "123")
        assert result["orderStatus"] == "PENDING"
        cancel.assert_not_called()
        warn.assert_called_once()


def test_swing_ambiguous_submit_recovers_saved_order_without_resubmit():
    plan = prepared(1010)
    stored = dict(_id="trade:BUY", simulated=False, orderStatus="SUBMITTING",
                  prepared=plan, correlationId="intent")
    orders, docs = collection(stored)
    with patch.object(dhan, "order_by_correlation", return_value=order()), \
            patch.object(dhan, "wait_for_fill", return_value=order("TRADED", 65)), \
            patch.object(dhan, "submit_prepared_order") as submit:
        result = swing.live_order({}, orders, "trade:BUY", dict(security_id="101"),
                                  65, "BUY", datetime(2026, 11, 5, 10), stored)
    assert result["filledQty"] == 65
    assert docs["trade:BUY"]["prepared"]["policy"]["deadline"] == 1010
    submit.assert_not_called()


def test_swing_best_price_and_quote_are_saved_before_submission():
    now = datetime(2026, 11, 9, 10)
    plan = prepared(1010)
    plan["policy"]["best_price_first"] = True
    plan["initial_quote"] = dict(quote=quote(), reference=100.2, limit_price=100.2,
                                received_at=1000, remaining_quantity=65)
    orders, docs = collection()
    orders.name = "orders_nifty_interceptor_test"
    audit = Mock()
    orders.database = {"order_quotes_nifty_interceptor_test": audit}

    def submit(conn, prepared_order):
        assert docs["trade:BUY"]["prepared"] == plan
        return dict(orderId="123")

    with patch.object(swing, "now_ist", return_value=now), \
            patch.object(dhan, "prepare_order", return_value=plan) as prepare, \
            patch.object(dhan, "submit_prepared_order", side_effect=submit), \
            patch.object(dhan, "wait_for_fill", return_value=order("TRADED", 65)) as wait:
        result = swing.live_order({}, orders, "trade:BUY",
                                  dict(security_id="101", symbol="TEST"), 65, "BUY", now, None)
    assert result["filledQty"] == 65
    assert prepare.call_args.kwargs["best_price_first"] is True
    assert prepare.call_args.kwargs["check_spread"] is False
    assert wait.call_args.kwargs["audit_collection"] is audit
    assert wait.call_args.kwargs["audit_id"] == "trade:BUY"
    assert docs["trade:BUY"]["prepared"]["initial_quote"]["quote"] == quote()


def test_swing_unacknowledged_intent_never_submits():
    now = datetime(2026, 11, 9, 10)
    orders = Mock()
    orders.insert_one.return_value = Mock(acknowledged=False)
    with patch.object(swing, "now_ist", return_value=now), \
            patch.object(dhan, "prepare_order", return_value=prepared(1010)), \
            patch.object(dhan, "submit_prepared_order") as submit:
        raises(RuntimeError, swing.live_order, {}, orders, "trade:BUY",
               dict(security_id="101", symbol="TEST"), 65, "BUY", now, None)
    submit.assert_not_called()


def test_swing_buy_and_sell_use_intraday_product_with_limit_depth_policy():
    now = datetime(2026, 11, 9, 10)
    for side in ("BUY", "SELL"):
        orders, docs = collection()
        key = "trade:" + side

        def submit(conn, plan):
            request = plan["request"]
            assert request["productType"] == "INTRADAY"
            assert request["orderType"] == "LIMIT"
            assert request["transactionType"] == side
            assert plan["policy"]["best_price_first"] is True
            assert plan["initial_quote"]["quote"] == quote()
            assert docs[key]["prepared"] == plan
            return dict(orderId="123")

        with patch.object(swing, "now_ist", return_value=now), \
                patch.object(dhan, "execution_tick", return_value=0.05), \
                patch.object(dhan, "get_depth", return_value=quote()), \
                patch.object(dhan.time, "time", return_value=1000), \
                patch.object(dhan, "submit_prepared_order", side_effect=submit), \
                patch.object(dhan, "wait_for_fill",
                             return_value=dict(order("TRADED", 65, side),
                                               productType="INTRADAY")):
            result = swing.live_order({"client_id": "test"}, orders, key,
                                      dict(security_id="101", symbol="TEST"),
                                      65, side, now, None)
        assert result["productType"] == "INTRADAY"


def test_swing_paper_buy_and_sell_record_intraday_product():
    now = datetime(2026, 11, 9, 10)
    for side in ("BUY", "SELL"):
        orders, _ = collection()
        with patch.object(swing, "now_ist", return_value=now), \
                patch.object(dhan, "get_option_ltp", return_value=100):
            result = swing.execute_order({}, orders, "trade:" + side,
                                         dict(security_id="101", symbol="TEST"),
                                         65, side, now, True)
        assert result["productType"] == "INTRADAY"


def test_swing_unresolved_submit_never_replaces_order():
    stored = dict(_id="trade:BUY", simulated=False, prepared=prepared(1010),
                  correlationId="intent")
    orders, _ = collection(stored)
    with patch.object(dhan, "order_by_correlation", side_effect=ValueError("not found")), \
            patch.object(dhan, "submit_prepared_order") as submit:
        raises(ValueError, swing.live_order, {}, orders, "trade:BUY",
               dict(security_id="101"), 65, "BUY", datetime(2026, 11, 5, 10), stored)
        submit.assert_not_called()


def test_swing_partial_entry_is_recorded_at_actual_size_then_closed_with_actual_charges():
    now = datetime(2026, 11, 5, 10)
    trade = dict(_id="trade", strategy_state="pending", quantity=65, simulated=False,
                 security_id="101", long_option_symbol="TEST", entry_signal="2026-11-05 09:59:00")
    strategies, docs = collection(trade)
    orders, _ = collection(dict(_id="trade:BUY"))
    bought = dict(order("CANCELLED", 30), fill_timestamp=str(now))
    sold = dict(order("TRADED", 30, "SELL"), quantity=30, fill_timestamp=str(now),
                averageTradedPrice=100)
    with patch.object(swing, "execute_order", return_value=bought):
        swing.open_pending({}, strategies, orders, trade, now)
    actual = docs["trade"]
    assert actual["quantity"] == 30 and actual["requested_quantity"] == 65
    assert actual["strategy_state"] == "closing"
    assert actual["total_investment"] == 3007.5
    with patch.object(swing, "execute_order", return_value=sold) as execute:
        swing.close_trade({}, strategies, orders, actual.copy(), actual["exit_reason"], now)
    assert execute.call_args.args[4] == 30
    assert docs["trade"]["strategy_state"] == "closed"
    assert docs["trade"]["pnl"] == -7.5
    assert docs["trade"]["net_pnl"] == swing.money(-7.5 - docs["trade"]["charges"]["total"])


def test_swing_zero_fill_entry_aborts_without_position():
    now = datetime(2026, 11, 5, 10)
    trade = dict(_id="trade", strategy_state="pending", quantity=65, simulated=False,
                 security_id="101", long_option_symbol="TEST", entry_signal="2026-11-05 09:59:00")
    strategies, docs = collection(trade)
    orders, _ = collection(dict(_id="trade:BUY"))
    with patch.object(swing, "execute_order", return_value=order("CANCELLED")):
        swing.open_pending({}, strategies, orders, trade, now)
    assert docs["trade"]["strategy_state"] == "aborted"
    assert "long_option_cost" not in docs["trade"]


def test_swing_invalid_entry_preparation_consumes_daily_slot_without_retry():
    now = datetime(2026, 11, 5, 10)
    trade = dict(_id="trade", strategy_state="pending", quantity=65, simulated=False,
                 security_id="101", long_option_symbol="TEST", entry_signal="2026-11-05 09:59:00")
    strategies, docs = collection(trade)
    orders, _ = collection()
    with patch.object(swing, "entry_is_current", return_value=True), \
            patch.object(swing, "execute_order", side_effect=ValueError("Entry spread exceeds 0.75%")):
        raises(ValueError, swing.open_pending, {}, strategies, orders, trade, now)
    assert docs["trade"]["strategy_state"] == "aborted"
    assert "spread exceeds" in docs["trade"]["exit_reason"]


def test_swing_submission_persists_policy_before_network_and_does_not_repeat():
    now = datetime(2026, 11, 5, 10)
    orders, docs = collection()
    plan = prepared(1010)

    def submit(conn, value):
        assert docs["trade:BUY"]["prepared"] == plan
        assert docs["trade:BUY"]["orderStatus"] == "SUBMITTING"
        return dict(orderId="123", orderStatus="PENDING")

    with patch.object(swing, "now_ist", return_value=now), \
            patch.object(dhan, "prepare_order", return_value=plan), \
            patch.object(dhan, "submit_prepared_order", side_effect=submit) as send, \
            patch.object(dhan, "wait_for_fill", return_value=order("TRADED", 65)) as wait:
        contract = dict(security_id="101", symbol="TEST")
        result = swing.live_order({}, orders, "trade:BUY", contract, 65, "BUY", now, None)
        swing.live_order({}, orders, "trade:BUY", contract, 65, "BUY", now, result)
    send.assert_called_once()
    wait.assert_called_once()


def test_quote_throttle_and_slow_or_failed_responses():
    response = Mock(status_code=200)
    response.json.return_value = dict(status="success", data={"NSE_FNO": {"101": quote()}})
    with patch.object(dhan, "_last_depth_request", 100), \
            patch.object(dhan.time, "monotonic", side_effect=[100.2, 101, 101.1]), \
            patch.object(dhan.time, "sleep") as sleep, \
            patch.object(dhan.requests, "post", return_value=response):
        assert dhan.get_depth({"client_id": "test", "access_token": "test"}, "101", "NSE_FNO") == quote()
        assert abs(sleep.call_args.args[0] - 0.8) < 1e-8
    with patch.object(dhan, "_last_depth_request", 0), \
            patch.object(dhan.time, "monotonic", side_effect=[100, 100, 104]), \
            patch.object(dhan.requests, "post", return_value=response):
        raises(ValueError, dhan.get_depth, {"client_id": "test", "access_token": "test"},
               "101", "NSE_FNO")


def test_http_limit_payload_cancel_and_correlation_recovery():
    conn = dict(client_id="test", access_token="test")
    plan = prepared()
    response = Mock(status_code=200)
    response.json.return_value = [order("TRADED", 65)]
    with patch.object(dhan.requests, "get", return_value=response) as get:
        assert dhan.order_by_correlation(conn, "intent")["filledQty"] == 65
        assert get.call_args.args[0].endswith("/orders/external/intent")
    response.json.return_value = dict(orderId="123")
    with patch.object(dhan.requests, "post", return_value=response) as post:
        dhan.submit_prepared_order(conn, plan)
        assert post.call_args.kwargs["json"] == plan["request"]
    with patch.object(dhan.requests, "delete", return_value=response) as delete:
        dhan.cancel_order(conn, "123")
        assert delete.call_args.args[0].endswith("/orders/123")


def test_deadline_is_rechecked_after_quote_before_modify():
    pending = order(price=99)
    with patch.object(dhan, "get_order", side_effect=[pending, pending, order("CANCELLED")]), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", side_effect=[1000, 1011, 1011]), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "cancel_order") as cancel, \
            patch.object(dhan, "modify_order") as modify, \
            patch.object(dhan, "warn_order_never_filled"):
        assert dhan.wait_for_fill({}, "123", prepared=prepared(1010))["filledQty"] == 0
    cancel.assert_called_once()
    modify.assert_not_called()


def test_swing_main_uses_interceptor_trades_and_orders_with_existing_signal():
    client = MagicMock()
    strategies = Mock()
    strategies.count_documents.return_value = 0
    strategies.find_one.return_value = None
    database = client["Bots"]
    database.__getitem__.side_effect = lambda name: strategies if name == "nifty_interceptor_test" else Mock()
    config = dict(user="test", mongo="offline", slack="offline", simulated=True)
    with patch.object(swing, "load_dotenv"), patch.object(swing, "find_dotenv", return_value=""), \
            patch.object(swing, "settings", return_value=config), \
            patch.object(swing, "MongoClient", return_value=client), \
            patch.object(swing, "WebClient"), patch.object(swing, "notify"), \
            patch.object(swing, "check_trading_day", return_value=True), \
            patch.object(swing.util, "get_public_ip", return_value="offline"), \
            patch.object(swing, "now_ist", return_value=datetime(2026, 11, 7, 15, 16)), \
            patch.dict(swing.os.environ, {"quantity": "65"}):
        swing.main()
    names = [call.args[0] for call in database.__getitem__.call_args_list]
    assert names == ["nifty_interceptor_test", "orders_nifty_interceptor_test", "supertrend"]
    client.close.assert_called_once()


if __name__ == "__main__":
    include_swing = "--swing" in sys.argv
    if include_swing:
        spec = importlib.util.spec_from_file_location(
            "swing_executor", ROOT.parent / "nifty-interceptor" / "executor" / "executor.py")
        swing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(swing)
    tests = [value for name, value in list(globals().items())
             if name.startswith("test_") and (include_swing or not name.startswith("test_swing_"))]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(str(len(tests)) + " execution checks passed")
