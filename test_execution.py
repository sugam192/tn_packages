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


def test_swing_phase_alerts_preserve_quote_submission_and_fill_timing():
    now = datetime(2026, 11, 23, 9, 33, 10)
    orders, docs = collection()
    alerts = Mock()
    plan = prepared(1010)

    def submit(conn, value):
        assert alerts.call_count == 0
        assert docs["trade:BUY"]["prepared"] == plan
        return dict(orderId="123", orderStatus="PENDING")

    def wait(*args, **kwargs):
        assert alerts.call_count == 0
        kwargs["on_event"]("Order not fully filled; remaining 65")
        assert alerts.call_count == 0
        kwargs["on_event"]("Requesting LIMIT modification")
        assert alerts.call_count == 0
        return order("TRADED", 65)

    with patch.object(swing, "now_ist", return_value=now), \
            patch.object(dhan, "prepare_order", return_value=plan), \
            patch.object(dhan, "submit_prepared_order", side_effect=submit), \
            patch.object(dhan, "wait_for_fill", side_effect=wait):
        swing.live_order({}, orders, "trade:BUY", dict(security_id="101", symbol="TEST"),
                         65, "BUY", now, None, report=alerts)
    messages = [call.args[0] for call in alerts.call_args_list]
    assert len(messages) == 1
    assert "Requesting fresh depth" in messages[0]
    stages = ["Fresh depth prepared", "placing LIMIT order now", "Order submitted",
              "monitoring same order"]
    positions = [messages[0].index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "Order not fully filled" in messages[0]
    assert "Requesting LIMIT modification" in messages[0]
    assert "Broker result TRADED" in messages[0] and "full fill confirmed" in messages[0]


def test_swing_monitor_alert_is_immediate_and_timeline_waits_for_persisted_fill():
    now = datetime(2026, 11, 23, 9, 33, 10)
    stored = dict(_id="trade:BUY", simulated=False, prepared=prepared(1010), **order())
    orders, docs = collection(stored)
    alerts = Mock()

    def wait(*args, **kwargs):
        kwargs["on_event"]("Fresh repricing quote saved")
        assert alerts.call_count == 0
        kwargs["on_alert"]("Still not fully filled after 2 accepted repricing requests")
        assert alerts.call_count == 1
        assert docs["trade:BUY"]["orderStatus"] == "PENDING"
        kwargs["on_event"]("Broker-confirmed LIMIT changed")
        assert alerts.call_count == 1
        return order("TRADED", 65)

    def report(message):
        if "execution update:" in message:
            assert docs["trade:BUY"]["orderStatus"] == "TRADED"
            assert docs["trade:BUY"]["fill_timestamp"] == str(now)
        alerts(message)

    with patch.object(swing, "now_ist", return_value=now), \
            patch.object(swing, "ORDER_NOTIFICATION_STATE", {}), \
            patch.object(dhan, "wait_for_fill", side_effect=wait):
        swing.live_order({}, orders, "trade:BUY", dict(security_id="101"),
                         65, "BUY", now, stored, report=report)
    assert alerts.call_count == 2
    assert "2 accepted repricing requests" in alerts.call_args_list[0].args[0]
    assert "Fresh repricing quote saved" in alerts.call_args_list[1].args[0]
    assert "full fill confirmed" in alerts.call_args_list[1].args[0]
    assert not swing.ORDER_NOTIFICATION_STATE


def test_swing_quote_429_retries_join_the_batched_timeline_and_never_alert_early():
    now = datetime(2026, 11, 23, 9, 33, 10)
    orders, docs = collection()
    alerts = Mock()
    plan = prepared(1010)

    def prepare(*args, **kwargs):
        kwargs["on_event"]("Quote request got HTTP 429; asking again after a short pause")
        assert alerts.call_count == 0
        return plan

    with patch.object(swing, "now_ist", return_value=now), \
            patch.object(dhan, "prepare_order", side_effect=prepare), \
            patch.object(dhan, "submit_prepared_order", return_value=dict(orderId="123")), \
            patch.object(dhan, "wait_for_fill", return_value=order("TRADED", 65)):
        swing.live_order({}, orders, "trade:BUY", dict(security_id="101", symbol="TEST"),
                         65, "BUY", now, None, report=alerts)
    assert alerts.call_count == 1
    timeline = alerts.call_args.args[0]
    stages = ["Requesting fresh depth", "BUY | Quote request got HTTP 429", "Fresh depth prepared",
              "placing LIMIT order now", "Order submitted", "full fill confirmed"]
    positions = [timeline.index(stage) for stage in stages]
    assert positions == sorted(positions)


def test_swing_failed_quote_reports_no_submission_and_retains_pending_trade():
    now = datetime(2026, 11, 23, 9, 33, 10)
    trade = dict(_id="trade", strategy_state="pending", simulated=False,
                 entry_signal="2026-11-23 09:32:00", security_id="101",
                 long_option_symbol="TEST", quantity=65)
    strategies, trade_docs = collection(trade)
    orders, order_docs = collection()
    alerts = Mock()
    failure = RuntimeError("Dhan HTTP 429: Too many requests")
    with patch.object(swing, "now_ist", return_value=now), \
            patch.object(dhan, "prepare_order", side_effect=failure), \
            patch.object(dhan, "submit_prepared_order") as submit:
        raises(RuntimeError, swing.open_pending, {}, strategies, orders, trade, now, report=alerts)
    assert trade_docs["trade"]["strategy_state"] == "pending"
    assert not order_docs
    submit.assert_not_called()
    messages = [call.args[0] for call in alerts.call_args_list]
    assert len(messages) == 1
    assert "Quote/preparation failed" in messages[0] and "429" in messages[0]
    assert "no order submitted" in messages[0]
    assert not any("full fill confirmed" in message for message in messages)


def test_swing_recovery_and_zero_fill_alerts_do_not_claim_success_or_resubmit():
    now = datetime(2026, 11, 23, 9, 33, 10)
    stored = dict(_id="trade:BUY", simulated=False, orderStatus="SUBMITTING",
                  prepared=prepared(1010), correlationId="intent")
    orders, _ = collection(stored)
    alerts = Mock()
    with patch.object(dhan, "order_by_correlation", return_value=order()), \
            patch.object(dhan, "wait_for_fill", return_value=order("CANCELLED")), \
            patch.object(dhan, "submit_prepared_order") as submit:
        result = swing.live_order({}, orders, "trade:BUY", dict(security_id="101"),
                                  65, "BUY", now, stored, report=alerts)
    assert result["filledQty"] == 0
    submit.assert_not_called()
    messages = "\n".join(call.args[0] for call in alerts.call_args_list)
    assert "Reconciling saved order" in messages and "Looking up submission" in messages
    assert "Broker result CANCELLED" in messages and "filled 0/65" in messages
    assert "placing LIMIT" not in messages and "full fill confirmed" not in messages


def test_swing_submission_and_monitoring_failures_keep_uncertainty_explicit():
    now = datetime(2026, 11, 23, 9, 33, 10)
    for failing_stage in ("submission", "monitoring"):
        orders, docs = collection()
        alerts = Mock()
        error = RuntimeError("Response unavailable")
        with patch.object(swing, "now_ist", return_value=now), \
                patch.object(dhan, "prepare_order", return_value=prepared(1010)), \
                patch.object(dhan, "submit_prepared_order",
                             side_effect=error if failing_stage == "submission" else None,
                             return_value=dict(orderId="123")) as submit, \
                patch.object(dhan, "wait_for_fill", side_effect=error) as wait:
            raises(RuntimeError, swing.live_order, {}, orders, "trade:BUY",
                   dict(security_id="101", symbol="TEST"), 65, "BUY", now, None, report=alerts)
        assert "trade:BUY" in docs
        submit.assert_called_once()
        messages = "\n".join(call.args[0] for call in alerts.call_args_list)
        assert "full fill confirmed" not in messages
        if failing_stage == "submission":
            wait.assert_not_called()
            assert "Submission failed or unconfirmed" in messages and "do not send another order" in messages
        else:
            wait.assert_called_once()
            assert "Fill monitoring failed" in messages and "may still be working" in messages


def test_monitor_initial_wait_accounts_for_notification_time_before_buy_deadline():
    plan = prepared(1002)
    plan["policy"]["best_price_first"] = True
    clock = {"epoch": 1000}

    def event(message):
        if "not fully filled at initial LIMIT" in message:
            clock["epoch"] = 1001.5

    with patch.object(dhan, "get_order", side_effect=[order(), order("TRADED", 65)]), \
            patch.object(dhan.time, "time", side_effect=lambda: clock["epoch"]), \
            patch.object(dhan.time, "sleep") as sleep:
        result = dhan.wait_for_fill({}, "123", prepared=plan, on_event=event)
    assert result["filledQty"] == 65
    sleep.assert_called_once_with(0.5)


def test_monitor_events_report_partial_repricing_then_broker_confirmed_change():
    plan = prepared(1010)
    plan["policy"]["best_price_first"] = True
    partial = order("PART_TRADED", 30, price=100.2)
    filled = order("TRADED", 65, price=100.4)
    events = []
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=True)

    def modify(*args):
        assert "Requesting LIMIT modification" in events[-1]
        assert audit.insert_one.call_count == 1
        assert args[3] == 65

    with patch.object(dhan, "get_order", side_effect=[partial, partial, partial, filled]), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "modify_order", side_effect=modify) as changed:
        result = dhan.wait_for_fill({}, "123", prepared=plan, audit_collection=audit,
                                    audit_id="trade:BUY", on_event=events.append)
    changed.assert_called_once()
    assert result["filledQty"] == 65
    combined = "\n".join(events)
    stages = ["not fully filled at initial LIMIT", "requesting fresh depth",
              "Fresh repricing depth received", "Repricing quote saved",
              "Requesting LIMIT modification", "Modification response received",
              "Broker-confirmed LIMIT changed", "Broker status TRADED"]
    positions = [combined.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "remaining 35" in combined
    assert "Rs100.20 -> Rs100.40" in combined
    assert "total Qty 65 unchanged" in combined


def test_monitor_fill_race_reports_skipped_modification_not_a_modified_order():
    events = []
    with patch.object(dhan, "get_order", side_effect=[order(price=99), order("TRADED", 65)]), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan, "modify_order") as modify:
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010), on_event=events.append)
    assert result["filledQty"] == 65
    modify.assert_not_called()
    combined = "\n".join(events)
    assert "became TRADED while evaluating quote | modification skipped" in combined
    assert "Requesting LIMIT modification" not in combined
    assert "Broker-confirmed LIMIT changed" not in combined


def test_monitor_quote_and_modification_errors_report_unconfirmed_state():
    for failure in ("quote", "modify"):
        events = []
        alerts = []
        state = dict(repricings=0, warned=False)
        error = RuntimeError("HTTP 429" if failure == "quote" else "Modify response unavailable")
        with patch.object(dhan, "get_order", return_value=order(price=99)), \
                patch.object(dhan, "execution_tick", return_value=0.05), \
                patch.object(dhan, "get_depth",
                             side_effect=error if failure == "quote" else None,
                             return_value=quote()), \
                patch.object(dhan, "modify_order", side_effect=error) as modify:
            raises(RuntimeError, dhan.wait_for_fill, {}, "123", on_event=events.append,
                   on_alert=alerts.append, notification_state=state)
        combined = "\n".join(events)
        assert "Broker status TRADED" not in combined and "Broker-confirmed LIMIT changed" not in combined
        if failure == "quote":
            modify.assert_not_called()
            assert "Repricing quote request failed" in combined and "HTTP 429" in combined
            assert "no modification sent" in combined
        else:
            modify.assert_called_once()
            assert "Modification failed/unconfirmed" in combined
            assert "Modification response received" not in combined
        assert len(alerts) == 1 and alerts[0] in combined
        assert state == dict(repricings=0, warned=False)


def test_monitor_two_accepted_reprices_alert_once_across_passes_unless_filled():
    for fully_filled in (False, True):
        state = dict(repricings=0, warned=False)
        alerts = []
        first = order(price=99)
        second = order("PART_TRADED", 30, price=100.4)
        last = order("TRADED" if fully_filled else "PART_TRADED",
                     65 if fully_filled else 30, price=100.65)
        for before, after, price in ((first, second, 100.4), (second, last, 100.65)):
            def modify(*args):
                assert not alerts
                assert args[3] == 65

            with patch.object(dhan, "get_order", side_effect=[before, before, after]), \
                    patch.object(dhan, "get_depth", return_value=quote()), \
                    patch.object(dhan, "depth_limit", return_value=(price, price)), \
                    patch.object(dhan.time, "monotonic", side_effect=[0, 0, 0, 0, 11]), \
                    patch.object(dhan.time, "sleep"), \
                    patch.object(dhan, "modify_order", side_effect=modify) as changed, \
                    patch.object(dhan, "warn_order_never_filled"):
                result = dhan.wait_for_fill({}, "123", prepared=prepared(),
                                            on_event=lambda message: None,
                                            on_alert=alerts.append, notification_state=state)
            changed.assert_called_once()
        assert state["repricings"] == 2
        assert len(alerts) == (0 if fully_filled else 1)
        if fully_filled:
            assert result["orderStatus"] == "TRADED" and not state["warned"]
        else:
            assert "filled 30/65" in alerts[0] and "remaining 35" in alerts[0]
            assert "LIMIT Rs100.65" in alerts[0] and state["warned"]
            with patch.object(dhan, "get_order", return_value=last), \
                    patch.object(dhan.time, "monotonic", side_effect=[0, 11]), \
                    patch.object(dhan, "warn_order_never_filled"):
                dhan.wait_for_fill({}, "123", prepared=prepared(),
                                   on_alert=alerts.append, notification_state=state)
            assert len(alerts) == 1


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


def quote_reply(status):
    response = Mock(status_code=status, url="https://api.dhan.co/v2/marketfeed/quote")
    if status == 200:
        response.json.return_value = dict(status="success", data={"NSE_FNO": {"101": quote()}})
    else:
        response.json.return_value = {
            "data": {"805": "Too many requests. Further requests may result in the user being blocked."},
            "status": "failed"}
    return response


def test_429_raises_a_retryable_error_and_every_other_failure_does_not():
    def failure(response):
        try:
            dhan.check_dhan_response(response)
        except Exception as error:
            return error
        raise AssertionError("Expected an error")

    limited = failure(quote_reply(429))
    assert isinstance(limited, dhan.DhanRateLimited) and isinstance(limited, Exception)
    assert str(limited).startswith("Dhan returned HTTP 429 for https://api.dhan.co/v2/marketfeed/quote - ")
    assert "Too many requests" in str(limited)
    for status in (400, 401, 500):
        other = failure(quote_reply(status))
        assert not isinstance(other, dhan.DhanRateLimited)
        assert f"HTTP {status}" in str(other)
    assert dhan.check_dhan_response(quote_reply(200)) is None


def test_depth_asks_again_after_two_then_four_seconds_on_429_only():
    conn = {"client_id": "test", "access_token": "test"}
    clock = {"now": 1000.0}

    def tick():
        clock["now"] += 1
        return clock["now"]

    for replies, attempts, sleeps in (
            ([429, 200], 2, [2]), ([429, 429, 200], 3, [2, 4]), ([429, 429, 429], 3, [2, 4])):
        events = []
        posts = [quote_reply(status) for status in replies]
        with patch.object(dhan, "_last_depth_request", 0), \
                patch.object(dhan.time, "monotonic", side_effect=tick), \
                patch.object(dhan.time, "sleep") as sleep, \
                patch.object(dhan.requests, "post", side_effect=posts) as post:
            if replies[-1] == 200:
                assert dhan.get_depth(conn, "101", "NSE_FNO", on_event=events.append) == quote()
            else:
                raises(dhan.DhanRateLimited, dhan.get_depth, conn, "101", "NSE_FNO",
                       on_event=events.append)
        assert post.call_count == attempts
        assert [call.args[0] for call in sleep.call_args_list] == sleeps
        assert len(events) == replies.count(429)
        assert all("HTTP 429" in message and "asking again" in message for message in events)
    # Nothing but a 429 is retried: a bad request or a stale quote fails at once.
    for reply, error in ((quote_reply(400), Exception), (quote_reply(200), ValueError)):
        stale = iter([2000, 2000, 2010]) if error is ValueError else None
        with patch.object(dhan, "_last_depth_request", 0), \
                patch.object(dhan.time, "monotonic",
                             side_effect=(lambda: next(stale)) if stale else tick), \
                patch.object(dhan.time, "sleep") as sleep, \
                patch.object(dhan.requests, "post", return_value=reply) as post:
            raises(error, dhan.get_depth, conn, "101", "NSE_FNO")
        post.assert_called_once()
        assert [call.args[0] for call in sleep.call_args_list if call.args[0] >= 2] == []


def test_prepare_order_and_repricing_pass_the_event_hook_to_depth():
    events = []
    with patch.object(dhan, "execution_tick", return_value=0.05), \
            patch.object(dhan, "get_depth", return_value=quote()) as depth:
        dhan.prepare_order({"client_id": "test"}, "101", "SELL", 65, "NSE_FNO", "INTRADAY",
                           on_event=events.append)
    assert depth.call_args.kwargs["on_event"] == events.append
    with patch.object(dhan, "get_order", side_effect=[order(price=99), order(price=99), order("TRADED", 65)]), \
            patch.object(dhan, "get_depth", return_value=quote()) as depth, \
            patch.object(dhan.time, "time", return_value=1000), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "modify_order"):
        dhan.wait_for_fill({}, "123", prepared=prepared(1010), on_event=events.append)
    assert depth.call_args.kwargs["on_event"] == events.append


def test_every_dhan_request_prints_a_millisecond_timestamp_for_later_correlation():
    conn = {"client_id": "test", "access_token": "test"}
    ltp = Mock(status_code=200)
    ltp.json.return_value = dict(data={"NSE_FNO": {"101": {"last_price": 100.2}}})
    candles = Mock(status_code=200)
    moment = datetime(2026, 11, 23, 9, 33, 19, 123456)
    with patch.object(dhan.util, "ist_now", return_value=moment), \
            patch.object(dhan, "_last_depth_request", 0), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "candles_to_dataframe", return_value="frame"), \
            patch.object(dhan.requests, "post", side_effect=[quote_reply(200), ltp, candles]), \
            patch("builtins.print") as printed:
        dhan.get_depth(conn, "101", "NSE_FNO")
        assert dhan.get_ltp(conn, "101", "NSE_FNO") == 100.2
        assert dhan.fetch_one_chunk(conn, "101", "NSE_FNO", "OPTIDX", datetime(2026, 11, 23),
                                    datetime(2026, 11, 23), "min") == "frame"
    lines = [call.args[0] for call in printed.call_args_list]
    assert lines == ["Dhan quote request at 09:33:19.123 IST", "Dhan LTP request at 09:33:19.123 IST",
                     "Dhan candles request at 09:33:19.123 IST"]


def test_monitor_quote_evaluations_without_modification_do_not_trigger_alert():
    clock = {"monotonic": 0}
    state = dict(repricings=0, warned=False)
    alerts = []
    audit = Mock()
    audit.insert_one.return_value = Mock(acknowledged=True)

    def sleep(seconds):
        clock["monotonic"] += seconds

    with patch.object(dhan, "get_order", return_value=order()), \
            patch.object(dhan, "get_depth", return_value=quote()) as depth, \
            patch.object(dhan.time, "monotonic", side_effect=lambda: clock["monotonic"]), \
            patch.object(dhan.time, "sleep", side_effect=sleep), \
            patch.object(dhan, "modify_order") as modify, \
            patch.object(dhan, "warn_order_never_filled"):
        result = dhan.wait_for_fill({}, "123", prepared=prepared(),
                                    audit_collection=audit, audit_id="trade:BUY",
                                    on_alert=alerts.append, notification_state=state)
    assert depth.call_count >= 2 and audit.insert_one.call_count == depth.call_count
    modify.assert_not_called()
    assert not alerts and state == dict(repricings=0, warned=False)
    assert result["orderStatus"] == "PENDING"


def test_monitor_rechecks_buy_deadline_after_immediate_modification_notice():
    events = []
    clock = {"epoch": 1000}

    def event(message):
        events.append(message)
        if "Requesting LIMIT modification" in message:
            clock["epoch"] = 1011

    with patch.object(dhan, "get_order",
                      side_effect=[order(price=99), order(price=99), order("CANCELLED")]), \
            patch.object(dhan, "get_depth", return_value=quote()), \
            patch.object(dhan.time, "time", side_effect=lambda: clock["epoch"]), \
            patch.object(dhan.time, "monotonic", return_value=0), \
            patch.object(dhan.time, "sleep"), \
            patch.object(dhan, "cancel_order") as cancel, \
            patch.object(dhan, "modify_order") as modify:
        result = dhan.wait_for_fill({}, "123", prepared=prepared(1010), on_event=event)
    assert result["orderStatus"] == "CANCELLED"
    modify.assert_not_called()
    cancel.assert_called_once()
    combined = "\n".join(events)
    assert "Entry deadline reached during notification" in combined
    assert combined.index("requesting cancellation") < combined.index("Broker status CANCELLED")


def test_monitor_notification_quote_age_boundary_prevents_stale_repricing():
    for age in (3, 3.01):
        events = []
        clock = {"monotonic": 0}

        def event(message):
            events.append(message)
            if "Requesting LIMIT modification" in message:
                clock["monotonic"] = age
            if "depth expired during notifications" in message:
                clock["monotonic"] = 11

        pending = order(side="SELL", price=101)
        filled = order("TRADED", 65, side="SELL", price=99.9)
        with patch.object(dhan, "get_order", side_effect=[pending, pending, filled]), \
                patch.object(dhan, "execution_tick", return_value=0.05), \
                patch.object(dhan, "get_depth", return_value=quote()), \
                patch.object(dhan.time, "monotonic", side_effect=lambda: clock["monotonic"]), \
                patch.object(dhan.time, "sleep"), \
                patch.object(dhan, "modify_order") as modify, \
                patch.object(dhan, "warn_order_never_filled"):
            result = dhan.wait_for_fill({}, "123", on_event=event)
        combined = "\n".join(events)
        if age == 3:
            modify.assert_called_once()
            assert result["orderStatus"] == "TRADED"
        else:
            modify.assert_not_called()
            assert result["orderStatus"] == "PENDING"
            assert "depth expired during notifications" in combined
            assert "same-order recovery required" in combined


def test_monitor_failed_cancellation_is_not_reported_as_cancelled():
    events = []
    alerts = []
    with patch.object(dhan, "get_order", return_value=order()), \
            patch.object(dhan.time, "time", return_value=1011), \
            patch.object(dhan, "cancel_order", side_effect=RuntimeError("Response unavailable")):
        raises(RuntimeError, dhan.wait_for_fill, {}, "123", prepared=prepared(1010),
               on_event=events.append, on_alert=alerts.append)
    combined = "\n".join(events)
    assert "Cancellation failed/unconfirmed" in combined
    assert "Cancellation request returned" not in combined and "Broker status CANCELLED" not in combined
    assert len(alerts) == 2 and "requesting cancellation" in alerts[0]
    assert "Cancellation failed/unconfirmed" in alerts[1]


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


def test_swing_live_startup_refuses_library_without_monitor_events_before_external_work():
    with patch.object(swing, "load_dotenv"), patch.object(swing, "find_dotenv", return_value=""), \
            patch.object(swing, "settings", return_value=dict(simulated=False)), \
            patch.object(swing, "dhan", Mock(spec=["prepare_order"])), \
            patch.object(swing, "MongoClient") as mongo, \
            patch.object(swing, "WebClient") as slack:
        raises(RuntimeError, swing.main)
    mongo.assert_not_called()
    slack.assert_not_called()
    old = Mock(spec=["execution_event", "wait_for_fill", "prepare_order"])
    old.wait_for_fill = lambda conn, order_id, on_event=None: None
    old.prepare_order = lambda conn, on_event=None: None
    with patch.object(swing, "load_dotenv"), patch.object(swing, "find_dotenv", return_value=""), \
            patch.object(swing, "settings", return_value=dict(simulated=False)), \
            patch.object(swing, "dhan", old), patch.object(swing, "MongoClient") as mongo:
        raises(RuntimeError, swing.main)
    mongo.assert_not_called()
    # A library with the notification hooks but a prepare_order that cannot report 429s.
    partly = Mock(spec=["execution_event", "wait_for_fill", "prepare_order"])
    partly.wait_for_fill = lambda conn, order_id, on_alert=None, notification_state=None: None
    partly.prepare_order = lambda conn: None
    with patch.object(swing, "load_dotenv"), patch.object(swing, "find_dotenv", return_value=""), \
            patch.object(swing, "settings", return_value=dict(simulated=False)), \
            patch.object(swing, "dhan", partly), patch.object(swing, "MongoClient") as mongo:
        raises(RuntimeError, swing.main)
    mongo.assert_not_called()


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
