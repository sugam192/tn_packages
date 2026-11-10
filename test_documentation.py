"""Verify public API coverage and execute the broker-free README example."""
import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re
import socket
from unittest.mock import patch

import pandas as pd

from tamingnifty import ta

ROOT = Path(__file__).resolve().parent
PATTERNS = [
    "double_top_buy", "double_bottom_sell", "bull_trap", "bear_trap",
    "high_pole", "low_pole", "bullish_turtle", "bearish_turtle",
]


def readme_example():
    text = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    code = re.search(r"```python\n(.*?)\n```", text, re.S).group(1)
    expected = re.search(r"Expected output:\n\n```text\n(.*?)\n```", text, re.S).group(1)
    namespace = {}
    output = StringIO()
    with patch.object(socket, "create_connection", side_effect=AssertionError("Example must stay offline")), \
            patch.object(ta.edge, "fetch_historical_data",
                         side_effect=AssertionError("Example must not fetch broker data")), \
            redirect_stdout(output):
        exec(compile(code, "README.md", "exec"), namespace)
    return namespace, output.getvalue(), expected


def test_readme_example_output_matches_the_documentation():
    namespace, actual, expected = readme_example()
    assert actual.split() == expected.split()
    assert namespace["candles"]["close"].tolist() == [100, 102, 105, 101, 97, 100, 104]


def test_renko_example_has_exact_compounded_values_and_seed():
    namespace, actual, expected = readme_example()
    chart = namespace["renko"]
    assert list(chart.columns) == ["datetime", "low", "high", "color", "close"]
    assert len(chart) == 16
    assert chart["close"].tolist() == [
        100, 101, 102.01, 103.03, 104.06, 102.01, 101, 100,
        99.01, 98.03, 97.06, 99.01, 100, 101, 102.01, 103.03,
    ]
    assert chart.iloc[0]["low"] == chart.iloc[0]["high"] == 100
    assert chart["datetime"].duplicated().any()
    fallback = ta.convert_to_renko(brick_size=1, df=namespace["candles"].drop(columns="open"))
    pd.testing.assert_frame_equal(chart, fallback)


def test_pnf_example_has_exact_columns_counts_and_pattern_shapes():
    namespace, actual, expected = readme_example()
    chart = namespace["pnf"]
    assert list(chart.columns) == ["datetime", "color", "low", "high", "count", "close"] + PATTERNS
    assert chart["color"].tolist() == ["green", "red", "green"]
    assert chart["low"].tolist() == [100, 97.06, 98.03]
    assert chart["high"].tolist() == [104.06, 103.03, 103.03]
    assert chart["count"].tolist() == [4, 7, 6]
    assert chart["close"].tolist() == [104.06, 97.06, 103.03]
    assert all(chart[name].dtype == bool for name in PATTERNS)
    assert not chart[PATTERNS].to_numpy().any()
    extra = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-01-05 09:22")], "open": [104], "close": [105],
    })
    extended = ta.convert_to_pnf(brick_size=1, df=pd.concat([namespace["candles"], extra], ignore_index=True))
    assert len(extended) == len(chart)
    assert extended.iloc[-1]["high"] == 104.06
    assert extended.iloc[-1]["datetime"] > chart.iloc[-1]["datetime"]


def test_flat_pnf_can_return_an_uncolored_zero_count_column():
    candles = pd.DataFrame({
        "datetime": pd.date_range("2026-01-05 09:15", periods=3, freq="min"),
        "open": [100, 100, 100], "close": [100, 100, 100],
    })
    chart = ta.convert_to_pnf(brick_size=1, df=candles)
    assert len(chart) == 1
    assert chart.iloc[0]["color"] == ""
    assert chart.iloc[0]["count"] == 0


def test_broker_chart_wrappers_match_offline_examples_without_live_calls():
    namespace, actual, expected = readme_example()
    candles = namespace["candles"]
    start = candles.iloc[0]["datetime"].to_pydatetime()
    end = candles.iloc[-1]["datetime"].to_pydatetime()
    for function, expected_chart in ((ta.renko, namespace["renko"]), (ta.pnf, namespace["pnf"])):
        with patch.object(ta.edge, "fetch_historical_data", return_value=candles.copy()) as fetch:
            chart = function(None, "NSE", "Nifty 50", start, end, brick_size=1)
        fetch.assert_called_once_with(None, "NSE", "Nifty 50", start, end, "min")
        pd.testing.assert_frame_equal(chart, expected_chart)


def test_every_library_defined_function_and_method_has_an_api_heading():
    expected = set()
    for path in (ROOT / "tamingnifty").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                expected.add(path.stem + "." + node.name)
            elif isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        expected.add(path.stem + "." + node.name + "." + method.name)
    text = (ROOT / "API_REFERENCE.md").read_text(encoding="utf-8-sig")
    headings = re.findall(r"^### ([a-zA-Z0-9_]+\.[a-zA-Z0-9_.]+)\s*$", text, re.M)
    assert len(headings) == len(set(headings)), "Duplicate API headings"
    assert expected == set(headings), (
        "Missing: " + str(sorted(expected - set(headings)))
        + "; unexpected: " + str(sorted(set(headings) - expected)))
