"""Tests to triplebarriers module."""  # noqa: INP001

import numpy as np
import pandas as pd

from fico.triplebarriers import (
    add_vertical_barrier,
    get_bins,
    get_events,
    get_t_events,
)

data = pd.read_csv("data/data.csv")
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")
close = data["adj_close"]
close.name = "close"

stock_df = pd.DataFrame(close)

span0 = 50
daily_vol = np.log(stock_df.close).diff().dropna().ewm(span=span0).std()
daily_vol  # noqa: B018

threshold = daily_vol.mean() * 1
cusum_events = get_t_events(stock_df.close, threshold=threshold)
cusum_events  # noqa: B018

vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=close, num_days=5)
vertical_barriers  # noqa: B018

pt_sl = np.array([1, 2])
min_ret = 0.0005
triple_barrier_events = get_events(
    close=stock_df.close,
    t_events=cusum_events,
    pt_sl=pt_sl,
    target=daily_vol,
    min_ret=min_ret,
    vertical_barriers=vertical_barriers,
)
triple_barrier_events  # noqa: B018

labels = get_bins(triple_barrier_events, stock_df.close, True)
labels.bin.value_counts()
labels.head()
