"""Tests for bars module."""  # noqa: INP001

import pandas as pd

from fico.chronologicalsampling import dollar_bars, time_bars, volume_bars

data = pd.read_csv("data/data.csv")
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")
close = data["adj_close"]
volume = data["volume"]

time_bars_ohlc = time_bars(close, time_range="3d")
dollar_bars_ohlc = dollar_bars(volume, close, dollar_amount=1e9)
volume_bars_ohlc = volume_bars(volume, close, traded_volume=1.2e7)

print(time_bars_ohlc)
print(dollar_bars_ohlc)
print(volume_bars_ohlc)
