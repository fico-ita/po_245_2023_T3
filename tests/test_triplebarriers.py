import numpy as np  # noqa: INP001, D100
import pandas as pd

from fico.triplebarriers import (
    add_vertical_barrier,
    get_events,
    get_t_events,
)

data = pd.read_csv("data/data.csv")
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")
close = data["adj_close"]

# determining daily volatility using the last 50 days
span0 = 50
daily_vol = np.log(close).diff().dropna().ewm(span=span0).std()

# creating our event triggers using the CUSUM filter
threshold = daily_vol.mean() * 1
cusum_events = get_t_events(close, threshold=threshold)

# adding vertical barriers with a half day expiration window
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=close, num_days=5)
pd.DataFrame(vertical_barriers)

pt_sl = np.array(
    [1, 2]
)  # setting profit-take and stop-loss at 1 and 2 standard deviations
min_ret = 0.0005  # setting a minimum return of 0.05%

triple_barrier_events = get_events(
    close=close,
    t_events=cusum_events,
    pt_sl=pt_sl,
    target=daily_vol,
    min_ret=min_ret,
    vertical_barriers=vertical_barriers,  # type: ignore
    # side=stock_df.side
)
triple_barrier_events

labels = get_bins(triple_barrier_events, close)
labels.bin.value_counts()