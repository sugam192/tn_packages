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

@retry(tries=5, delay=5, backoff=2)
def notify(message = ("This is just a stupid notification!"), slack_channel = 'niftyweekly', slack_client = get_slack_client()):
    channel = "#" + slack_channel
    print(message)
    slack_client.chat_postMessage(
        channel=channel, 
        text=message, 
        username="TradeBot",
        icon_emoji=":chart_with_upwards_trend:"
    )