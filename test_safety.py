import copy
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pymongo.errors import DuplicateKeyError

from tamingnifty.safety import check_trading_day, assert_execution_clear, guarded_execution


def calendar_client():
    document = json.loads((Path(__file__).parent / "nse_calendar_2026.json").read_text())
    collection = Mock()
    collection.find_one.return_value = document
    return {"Bots": {"exchange_calendars": collection}}, collection


@pytest.mark.parametrize("day,expected", [
    ("2026-11-10", False), ("2026-11-11", True), ("2026-11-24", False),
    ("2026-12-25", False), ("2026-11-08", False), ("2026-11-14", False)])
def test_calendar_regular_sessions(day, expected):
    client, collection = calendar_client()
    notify = Mock()
    assert check_trading_day(client, notify, datetime.fromisoformat(day)) == expected
    collection.find_one.assert_called_once_with({"_id": "NSE:2026"})
    assert notify.call_count == (0 if expected else 1)


@pytest.mark.parametrize("failure", [
    "missing", "year", "status", "coverage", "duplicate", "bad_date", "sources",
    "empty", "database", "new_year"])
def test_calendar_fails_closed_and_alerts(failure):
    client, collection = calendar_client()
    doc = collection.find_one.return_value
    now = datetime(2026, 11, 11)
    if failure == "missing":
        collection.find_one.return_value = None
    elif failure == "year":
        doc["year"] = 2025
    elif failure == "status":
        doc["status"] = "draft"
    elif failure == "coverage":
        doc["valid_to"] = "2026-11-10"
    elif failure == "duplicate":
        doc["holidays"].append(doc["holidays"][0])
    elif failure == "bad_date":
        doc["holidays"][0]["date"] = "2026-02-30"
    elif failure == "sources":
        doc["sources"] = []
    elif failure == "empty":
        doc["holidays"] = []
    elif failure == "database":
        collection.find_one.side_effect = RuntimeError("offline")
    elif failure == "new_year":
        now = datetime(2027, 1, 1)
    notify = Mock()
    with pytest.raises((ValueError, RuntimeError)):
        check_trading_day(client, notify, now)
    notify.assert_called_once()


def guard_client():
    documents = {}
    collection = Mock()

    def find(query):
        return copy.deepcopy(documents.get(query["_id"]))

    def update(query, changes):
        doc = documents.get(query["_id"])
        matched = doc is not None and all(doc.get(key) == value for key, value in query.items())
        if matched:
            doc.update(changes["$set"])
        return SimpleNamespace(acknowledged=True, matched_count=int(matched))

    def insert(doc):
        if doc["_id"] in documents:
            raise DuplicateKeyError("locked")
        documents[doc["_id"]] = dict(doc)
        return SimpleNamespace(acknowledged=True)

    collection.find_one.side_effect = find
    collection.update_one.side_effect = update
    collection.insert_one.side_effect = insert
    return {"Bots": {"execution_guards": collection}}, collection, documents


def test_guard_persists_before_call_and_completes():
    client, collection, documents = guard_client()
    notify = Mock()

    def operation():
        assert documents["bot"]["status"] == "pending"
        return 42

    assert_execution_clear(client, "bot", notify)
    assert guarded_execution(client, "bot", notify, "entry", operation) == 42
    assert documents["bot"]["status"] == "complete"
    assert guarded_execution(client, "bot", notify, "exit", operation) == 42
    assert_execution_clear(client, "bot", notify)
    notify.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("persist failed"), SystemExit(1)])
def test_failure_stops_and_restart_cannot_repeat_orders(error):
    client, collection, documents = guard_client()
    operation = Mock(side_effect=error)
    notify = Mock()
    with pytest.raises(SystemExit):
        guarded_execution(client, "bot", notify, "entry", operation)
    assert documents["bot"]["status"] == "failed"
    with pytest.raises(RuntimeError):
        assert_execution_clear(client, "bot", notify)
    with pytest.raises(SystemExit):
        guarded_execution(client, "bot", notify, "entry", operation)
    operation.assert_called_once()
    assert notify.call_count >= 3


def test_crash_pending_guard_blocks_another_process():
    client, collection, documents = guard_client()
    documents["bot"] = dict(_id="bot", status="pending", attempt_id="other")
    operation = Mock()
    with pytest.raises(SystemExit):
        guarded_execution(client, "bot", Mock(), "entry", operation)
    operation.assert_not_called()
    assert documents["bot"]["attempt_id"] == "other"


def test_database_unavailable_does_not_send_orders():
    client, collection, documents = guard_client()
    collection.update_one.side_effect = RuntimeError("offline")
    operation = Mock()
    with pytest.raises(SystemExit):
        guarded_execution(client, "bot", Mock(), "entry", operation)
    operation.assert_not_called()


def test_lost_completion_write_stops_after_single_order_and_keeps_latch():
    client, collection, documents = guard_client()
    original = collection.update_one.side_effect

    def update(query, changes):
        if changes["$set"]["status"] in ("complete", "failed"):
            raise RuntimeError("write timeout")
        return original(query, changes)

    collection.update_one.side_effect = update
    operation = Mock()
    notify = Mock(side_effect=RuntimeError("Slack offline"))
    with pytest.raises(SystemExit):
        guarded_execution(client, "bot", notify, "exit", operation)
    operation.assert_called_once()
    assert documents["bot"]["status"] == "pending"


def test_unacknowledged_reservation_never_executes():
    client, collection, documents = guard_client()
    collection.insert_one.side_effect = lambda doc: SimpleNamespace(acknowledged=False)
    operation = Mock()
    with pytest.raises(SystemExit):
        guarded_execution(client, "bot", Mock(), "entry", operation)
    operation.assert_not_called()
