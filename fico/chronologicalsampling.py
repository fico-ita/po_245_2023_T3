"""Provide methods for sampling data chronologically.

This module allows the user to rearrange the data chronologically in terms of volume,
time, and dollar bars.

The module contains the following functions:

- `dollar_bars(dollar_volume, close, traded_dollar_volume)` - Creates dollar bars
    with the OH LC
- `time_bars(close, time_delta)` - Creates time bars with the OHLC
- `volume_bars(volume, close, traded_volume)` - Creates volume bars with the OHLC


"""

import numpy as np
import pandas as pd


def volume_bars(
    volume: pd.Series,
    close: pd.Series,
    traded_volume: float,
) -> pd.DataFrame:
    """Function to create volume bars with the OHLC [open, high, low, close] prices.

    Parameters:
        volume (pd.Series): Volume of a given stock in a given period.
        close (pd.Series): Close prices of a given stock in a given period.
        traded_volume (float): Number of shares traded in each bar.

    Returns:
        volume_bars_ohlc (pd.DataFrame): Dataframe with the volume bars with the OHLC \
        [open, high, low, close] prices.
    """
    # Create a dataframe with volume and close
    volume_bars_df = pd.DataFrame({"volume": volume, "close": close})

    # Create a function to aggregate the bars
    bar_calc = (
        np.int64(np.cumsum(volume_bars_df["volume"]) / traded_volume) * traded_volume
    )

    # Group the dataframe by the bar function and aggregate the close and volume
    volume_bars_ohlc = volume_bars_df.groupby(bar_calc).agg(
        {"close": "ohlc", "volume": "sum"},
    )

    # Set the index to be the date of the first trade in the bar
    volume_bars_ohlc.index = (
        (np.cumsum(volume_bars_df["volume"]) / traded_volume)
        .astype(int)
        .drop_duplicates(keep="first")
        .index
    )

    return volume_bars_ohlc


def dollar_bars(
    volume: pd.Series,
    close: pd.Series,
    dollar_amount: float,
) -> pd.DataFrame:
    """Function to create dollar bars with the OHLC [open, high, low, close] prices.

    Parameters:
        volume (pd.Series): Volume of a given stock in a given period.
        close (pd.Series): Close prices of a given stock in a given period.
        dollar_amount (float): Dollar amount traded in each bar.

    Returns:
        dollar_bars_ohlc (pd.DataFrame): Dataframe with the dollar bars with the OHLC \
        [open, high, low, close] prices.
    """
    # Create a dataframe with volume and close prices
    dollar_bars_df = pd.DataFrame({"volume": volume, "close": close})

    # Calculate the cumulative dollar value
    cumulative_dollar_value = (
        dollar_bars_df["volume"] * dollar_bars_df["close"]
    ).cumsum()

    # Calculate the bar index based on the dollar amount
    bar_index = (cumulative_dollar_value / dollar_amount).astype(int)

    # Group the dataframe by the bar index and aggregate the close and volume
    dollar_bars_ohlc = dollar_bars_df.groupby(bar_index).agg(
        {"close": "ohlc", "volume": "sum"},
    )

    # Set the index to be the date of the first trade in the bar
    dollar_bars_ohlc.index = cumulative_dollar_value.groupby(bar_index).idxmin()

    return dollar_bars_ohlc


def time_bars(close: pd.Series, time_range: str) -> pd.DataFrame:
    """Function to create time bars with the OHLC [open, high, low, close] prices.

    Parameters:
        close (pd.Series): Close prices of a given stock in a given period.
        time_range (str): Time period for the bars (e.g., '3d' for 3 days).

    Returns:
        time_bars_ohlc (pd.DataFrame): Dataframe with the time bars with the OHLC \
        [open, high, low, close] prices.
    """
    # Create a dataframe with close prices
    time_bars_df = pd.DataFrame({"close": close})

    # Group the dataframe by the time range and aggregate the close prices
    return time_bars_df.groupby(pd.Grouper(freq=time_range)).agg({"close": "ohlc"})
