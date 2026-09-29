import os
import traceback
from datetime import datetime

import pandas as pd
import requests
from retry import retry
from slack_sdk import WebClient

def round_to_nearest(x, base=0.05):
    """
    Round a number to the nearest specified multiple.
    """
    return round(base * round(float(x)/base), 2)

def resample_ohlc_data(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample OHLC data to specified frequency and return the resulting DataFrame.
    """
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure 'datetime' is in datetime format
    df.set_index('datetime', inplace=True)
    df_resampled = df.resample(frequency).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    for column in ['open', 'high', 'low', 'close']:
        df_resampled[column] = df_resampled[column].apply(lambda x: round_to_nearest(x, base=0.05))
    return df_resampled

def get_public_ip():
    """
    Return the public IP address this process goes out to the internet from.

    Dhan whitelists exactly ONE static IP per account for order placement, and that
    IP cannot be changed for 7 days once it is set. A container has no other way of
    telling us which address it is actually egressing from, so every bot reports this
    when it starts. That is how we confirm the Azure NAT Gateway is doing its job, and
    how we notice if the address ever changes underneath us.

    This never raises. A bot must not fail to start because an IP lookup failed.
    """
    try:
        return requests.get("https://api.ipify.org", timeout=10).text.strip()
    except Exception as e:
        return f"unknown ({e})"

def get_slack_client(token='fgnjf'):
    """
    This function return the webclient to interact with Slack.
    It accepts OAuth Token as a parameter.
    """
    return WebClient(token=token)

def exception_detail(e):
    """
    One line saying WHAT broke, WHERE and WHY. This is what belongs in a Slack message.

    str(e) on its own has repeatedly been useless. requests' HTTPError renders as
    "400 Client Error:  for url: ..." and throws the response body away, so on
    2026-09-28 two bots posted that same string all day and it took four throwaway
    scripts to find out Dhan was actually saying DH-905. A KeyError renders as just
    the missing key, with no clue which line wanted it.

    The traceback is deliberately NOT in here. It goes to stdout instead, so that an
    error repeating every 10 seconds does not flood the channel.
    """
    frames = traceback.extract_tb(e.__traceback__)
    if not frames:
        return f"{type(e).__name__} - {e}"

    def place(frame):
        return f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}()"

    # The deepest frame is where it actually blew up, but that is often inside
    # requests or pandas, which tells you nothing about which line of the bot asked
    # for it - "models.py:1026 in raise_for_status()" is not a debuggable address.
    # So also name the last frame that is our own code, i.e. not an installed package.
    raised_at = place(frames[-1])
    ours = [f for f in frames if "site-packages" not in f.filename]
    if ours and place(ours[-1]) != raised_at:
        return f"{type(e).__name__} at {raised_at}, called from {place(ours[-1])} - {e}"
    return f"{type(e).__name__} at {raised_at} - {e}"

def notify(message = ("This is just a stupid notification!"), slack_channel = 'niftyweekly', slack_client = get_slack_client()):
    # Printed before the post and OUTSIDE the retry below, for two reasons: the log
    # line has to survive a Slack outage, and it has to appear once rather than five
    # times. flush=True because a container's stdout is block-buffered when it is not
    # a terminal, so without it the last lines before a crash can be lost - which is
    # exactly when you need them.
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{slack_channel}] {message}", flush=True)
    post_to_slack(message, slack_channel, slack_client)

@retry(tries=5, delay=5, backoff=2)
def post_to_slack(message, slack_channel, slack_client):
    """Only the network call is retried. Callers should use notify()."""
    slack_client.chat_postMessage(
        channel="#" + slack_channel,
        text=message,
        username="TradeBot",
        icon_emoji=":chart_with_upwards_trend:"
    )