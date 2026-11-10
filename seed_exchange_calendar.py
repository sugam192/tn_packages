"""Validate a reviewed calendar JSON; write only with explicit --apply."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

from tamingnifty.safety import validate_calendar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calendar", type=Path)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.calendar.read_text(encoding="utf-8-sig"))
    year = document["year"]
    validate_calendar(document, datetime(year, 1, 1).date())
    validate_calendar(document, datetime(year, 12, 31).date())
    print(json.dumps(document, indent=2))
    if not args.apply:
        print("PREVIEW ONLY: no database connection or write")
        return
    load_dotenv(args.env_file)
    connection = os.environ.get("CONNECTION_STRING")
    if not connection:
        raise ValueError("Missing CONNECTION_STRING")
    with MongoClient(connection, serverSelectionTimeoutMS=15000) as client:
        collection = client["Bots"]["exchange_calendars"]
        result = collection.replace_one({"_id": document["_id"]}, document, upsert=True)
        if not result.acknowledged:
            raise RuntimeError("Calendar write not acknowledged")
        if collection.find_one({"_id": document["_id"]}) != document:
            raise RuntimeError("Calendar read-back differs from supplied calendar")
    print("VERIFIED: calendar saved and exact read-back matches")


if __name__ == "__main__":
    main()
