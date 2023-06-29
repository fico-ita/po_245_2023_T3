"""Provide methods for labeling strategies for financial time series.

This module allows the user to label financial time series with the technical indicators

The module contains the following functions:

- `bolingerbands(close_prices, window, no_of_stdev)` - Creates indicators for the \
    Bollinger Bands strategy.


"""

import pandas as pd


def bollingerbands(
    close_prices: pd.Series,
    window: int,
    no_of_stdev: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    r"""Bollinger Bands indicator for a given dataframe of prices.

    Parameters:
        close_prices (pd.Series): Close prices of a given stock.
        window (int): Rolling window for the moving average.
        no_of_stdev (float): Number of standard deviations for the bands.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple containing: \\
            - rolling_mean (pd.Series): Rolling mean of the close prices. \\
            - upper_band (pd.Series): Upper band of the Bollinger Bands. \\
            - lower_band (pd.Series): Lower band of the Bollinger Bands.

    """
    # Compute rolling mean
    rolling_mean = close_prices.ewm(span=window).mean()

    # Compute rolling standard deviation
    rolling_std = close_prices.ewm(span=window).std()

    # Compute upper and lower bands
    upper_band = rolling_mean + (rolling_std * no_of_stdev)
    lower_band = rolling_mean - (rolling_std * no_of_stdev)

    return rolling_mean, upper_band, lower_band
