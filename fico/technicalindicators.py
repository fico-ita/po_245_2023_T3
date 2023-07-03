"""Provide methods for labeling strategies for financial time series.

This module allows the user to label financial time series with the technical indicators

The module contains the following functions:

- `bolingerbands(close_prices, window, no_of_stdev)` - Creates indicators for the \
    Bollinger Bands strategy.
- `bollingerbandslabeling(close_prices, window, no_of_stdev)` - Creates labels for the \
    Bollinger Bands strategy.
- `bollingerbandssignal(close_prices, window, no_of_stdev)` - Creates dataframe for the\
    Bollinger Bands strategy.
- `trendfollowsignal(close_prices, fast_window, slow_window)` - Creates dataframe for\
    the Trend Following strategy.

"""

import numpy as np
import pandas as pd


def bollingerbands(
    close_prices: pd.Series,
    window: int,
    no_of_stdev: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands for a given dataframe of prices.

    Parameters:
        close_prices (pd.Series): Close prices of a given stock.
        window (int): Rolling window for the moving average.
        no_of_stdev (float): Number of standard deviations for the bands.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple containing: \\
            - rolling_mean (pd.Series): Rolling mean of the close prices. \\
            - upper_band (pd.Series): Upper band of the Bollinger Bands. \\
            - lower_band (pd.Series): Lower band of the Bollinger Bands.

    """  # noqa: D301
    # Compute rolling mean
    rolling_mean = close_prices.ewm(span=window).mean()

    # Compute rolling standard deviation
    rolling_std = close_prices.ewm(span=window).std()

    # Compute upper and lower bands
    upper_band = rolling_mean + (rolling_std * no_of_stdev)
    lower_band = rolling_mean - (rolling_std * no_of_stdev)

    return rolling_mean, upper_band, lower_band


def bollingerbandslabeling(
    close_prices: pd.Series,
    window: int,
    no_of_stdev: float,
) -> pd.Series:
    """Bollinger Bands side labels for a given dataframe of prices.

    Parameters:
        close_prices (pd.Series): Close prices of a given stock.
        window (int): Rolling window for the moving average.
        no_of_stdev (float): Number of standard deviations for the bands.

    Returns:
        (pd.Series): A Series containing the side labels for the Bollinger Bands.
    """
    label_series = pd.DataFrame(close_prices)
    label_series.columns = ["close"]

    # compute bollinger bands
    _, label_series["upper"], label_series["lower"] = bollingerbands(
        close_prices,
        window,
        no_of_stdev,
    )

    # compute sides
    label_series["label"] = np.nan
    long_signals = label_series.close <= label_series.lower
    short_signals = label_series.close >= label_series.upper
    label_series.loc[long_signals, "label"] = 1
    label_series.loc[short_signals, "label"] = -1

    return label_series.label


def bollingerbandssignal(
    close_prices: pd.Series,
    window: int,
    no_of_stdev: float,
) -> pd.DataFrame:
    """Bollinger Bands side signal for a given dataframe of prices.

    On this function we will use the side labels to generate the signals for the
    Bollinger Bands strategy, removing the look ahead bias by lagging the signal.

    Parameters:
        close_prices (pd.Series): Close prices of a given stock.
        window (int): Rolling window for the moving average.
        no_of_stdev (float): Number of standard deviations for the bands.

    Returns:
        (pd.DataFrame): A dataframe containing the side signals for the Bollinger
            Bands series, labels, signals, and side
    """
    df_signals = pd.DataFrame({"close": close_prices})

    # compute bollinger bands
    df_signals["ewm_mean"], df_signals["upper"], df_signals["lower"] = bollingerbands(
        close_prices,
        window,
        no_of_stdev,
    )

    # compute sides
    df_signals["label"] = bollingerbandslabeling(
        close_prices,
        window,
        no_of_stdev,
    )

    # compute sides
    df_signals["side_long"] = np.nan
    long_signals = df_signals.close <= df_signals.lower
    short_signals = df_signals.close >= df_signals.upper
    df_signals.loc[long_signals, "side_long"] = 1
    df_signals.loc[short_signals, "side_long"] = 0

    df_signals["side_long"] = df_signals.side_long.fillna(method="ffill")

    # Remove Look ahead biase by lagging the signal
    df_signals.side_long = df_signals.side_long.diff()
    df_signals.side_long.loc[df_signals.side_long == 0] = np.nan

    # compute sides
    df_signals["side_short"] = np.nan
    long_signals = df_signals.close <= df_signals.lower
    short_signals = df_signals.close >= df_signals.upper
    df_signals.loc[long_signals, "side_short"] = 0
    df_signals.loc[short_signals, "side_short"] = 1

    df_signals["side_short"] = df_signals.side_short.fillna(method="ffill")

    # Remove Look ahead biase by lagging the signal
    df_signals.side_short = df_signals.side_short.diff()
    df_signals.side_short.loc[df_signals.side_short == 0] = np.nan

    # compute side
    df_signals["side"] = df_signals["label"].replace(0, np.nan).ffill().bfill()

    return df_signals


def trendfollowsignal(
    close_prices: pd.Series,
    fast: int,
    slow: int,
) -> pd.DataFrame:
    """Trend following signal for a given dataframe of prices.

    On this function we will use the side labels to generate the signals for the
    Trend Following strategy, removing the look ahead bias by lagging the signal.

    Parameters:
        close_prices (pd.Series): Close prices of a given stock.
        fast (int): Fast moving average.
        slow (int): Slow moving average.

    Returns:
        (pd.DataFrame): A dataframe containing the side signals for the Trend Following
            series, labels and signals.
    """
    df_signals = pd.DataFrame({"close": close_prices})

    # compute exponential moving averages
    df_signals["fast"] = df_signals["close"].ewm(span=fast).mean()
    df_signals["slow"] = df_signals["close"].ewm(span=slow).mean()

    # compute labels
    df_signals["label"] = (df_signals["fast"] > df_signals["slow"]).astype(int).diff()

    # compute sides
    df_signals.loc[df_signals["label"] == -1, "side_short"] = 1
    df_signals.loc[df_signals["label"] == 1, "side_long"] = 1

    # compute side
    df_signals["side"] = df_signals["label"].replace(0, np.nan).ffill().bfill()

    return df_signals
