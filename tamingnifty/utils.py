import pandas as pd
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