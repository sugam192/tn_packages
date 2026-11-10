"""Shared startup calendar and durable fail-stop execution guards."""
from datetime import datetime
from uuid import uuid4

from pymongo.errors import DuplicateKeyError

from tamingnifty.utils import exception_detail, ist_now


def alert(notify, message):
    print(message, flush=True)
    try:
        notify(message)
    except Exception as error:
        print("Safety alert delivery failed: " + exception_detail(error), flush=True)


def validate_calendar(document, day):
    if not isinstance(document, dict):
        raise ValueError("Missing NSE calendar for " + str(day.year))
    expected = "NSE:" + str(day.year)
    if (document.get("_id") != expected or document.get("exchange") != "NSE"
            or document.get("year") != day.year or document.get("status") != "verified"):
        raise ValueError("Unverified or incompatible NSE calendar: " + expected)
    start = datetime.strptime(document["valid_from"], "%Y-%m-%d").date()
    end = datetime.strptime(document["valid_to"], "%Y-%m-%d").date()
    if start != day.replace(month=1, day=1) or end != day.replace(month=12, day=31):
        raise ValueError("NSE calendar does not cover " + str(day))
    datetime.strptime(document["verified_at"], "%Y-%m-%d")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources or not all(
            isinstance(source, str) and source.startswith("https://") for source in sources):
        raise ValueError("NSE calendar needs source URLs")
    holidays = document.get("holidays")
    if not isinstance(holidays, list) or not holidays:
        raise ValueError("NSE calendar needs an explicit holiday list")
    dates = {}
    for holiday in holidays:
        if not isinstance(holiday, dict):
            raise ValueError("Invalid NSE holiday entry")
        date = datetime.strptime(holiday["date"], "%Y-%m-%d").date()
        if date.year != day.year or date in dates or not isinstance(holiday.get("name"), str) or not holiday["name"].strip():
            raise ValueError("Invalid or duplicate NSE holiday")
        dates[date] = holiday["name"]
    return dates


def check_trading_day(client, notify, now=None):
    moment = now if now is not None else ist_now()
    if moment.tzinfo is not None:
        raise ValueError("Calendar clock must be naive IST")
    day = moment.date()
    try:
        document = client["Bots"]["exchange_calendars"].find_one({"_id": "NSE:" + str(day.year)})
        holidays = validate_calendar(document, day)
    except Exception as error:
        alert(notify, "STOPPED: NSE calendar unavailable/invalid; no trading. " + exception_detail(error))
        raise
    reason = holidays.get(day)
    if reason is None and day.weekday() >= 5:
        reason = "weekend; regular-session jobs do not trade special weekend sessions"
    if reason is not None:
        alert(notify, "NSE closed for this job on " + str(day) + ": " + reason + ". Exiting without trading.")
        return False
    return True


def assert_execution_clear(client, scope, notify):
    try:
        document = client["Bots"]["execution_guards"].find_one({"_id": scope})
        if document is not None and document.get("status") != "complete":
            raise RuntimeError("Unresolved execution guard " + scope
                               + "; reconcile broker orders/positions and Mongo before manual clearance")
    except Exception as error:
        alert(notify, "STOPPED: execution safety check failed. " + exception_detail(error))
        raise


def guarded_execution(client, scope, notify, action, function, *args, **kwargs):
    collection = client["Bots"]["execution_guards"]
    attempt = None
    try:
        token = uuid4().hex
        fields = dict(status="pending", attempt_id=token, action=action, started_at=str(ist_now()))
        result = collection.update_one({"_id": scope, "status": "complete"}, {"$set": fields})
        if not result.acknowledged:
            raise RuntimeError("Execution reservation not acknowledged")
        if result.matched_count == 0:
            try:
                inserted = collection.insert_one(dict(_id=scope, **fields))
            except DuplicateKeyError as error:
                raise RuntimeError("Unresolved/concurrent execution: " + scope
                                   + "; manual reconciliation required") from error
            if not inserted.acknowledged:
                raise RuntimeError("Execution reservation insertion not acknowledged")
        attempt = token
        value = function(*args, **kwargs)
        finished = collection.update_one(
            {"_id": scope, "attempt_id": attempt, "status": "pending"},
            {"$set": dict(status="complete", completed_at=str(ist_now()))})
        if not finished.acknowledged or finished.matched_count != 1:
            raise RuntimeError("Execution completion not acknowledged; do not retry")
        return value
    except (Exception, SystemExit) as error:
        if attempt is not None:
            try:
                failed = collection.update_one(
                    {"_id": scope, "attempt_id": attempt},
                    {"$set": dict(status="failed", failed_at=str(ist_now()),
                                  error=exception_detail(error))})
                if not failed.acknowledged or failed.matched_count != 1:
                    raise RuntimeError("Could not acknowledge failed execution guard")
            except Exception as guard_error:
                alert(notify, "Could not persist failure; pending guard must remain unresolved. "
                      + exception_detail(guard_error))
        alert(notify, "STOPPED: " + scope + " " + action + " failed: " + exception_detail(error)
              + ". No automatic retry. MANUAL RECONCILIATION REQUIRED; position may be open.")
        raise SystemExit(1) from error
