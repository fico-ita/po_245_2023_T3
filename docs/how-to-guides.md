This part of the project documentation focuses on a
**problem-oriented** approach. You'll tackle common
tasks that you might have, with the help of the code
provided in this project.


## Getting started

To get started with FICO *codebase*, take the following steps:


### Installation

```bash
poetry add git+https://github.com/fico-ita/po_245_2023_T3.git
```

### Usage

The package can be imported as follows:

```python
>>> from fico.technicalindicators import bollingerbands
>>> from fico.chronologicalsampling import dollar_bars, time_bars, volume_bars
>>> from fico.triplebarriers import add_vertical_barrier, apply_pt_sl_on_t1, get_events,
   get_t_events, get_volatility
```

## How To Get Bollinger Bands series, labels and signals?

You have a time series of prices and you want to get the Bollinger Bands series.
On the following example, we will use the `bollingerbands()` function from the
`fico.technicalindicators` module. For that on your python console on the same
directory of the repository, run the following commands:

```python
>>> import pandas as pd
>>> from fico.technicalindicators import bollingerbandssignal
>>> data = pd.read_csv("data/data.csv")
>>> close = data["close"]
>>> df = bollingerbandssignal(close, 50, 1)
>>> df.head()
   close   ewm_mean      upper      lower  label  side_long  side_short
0  74.09  74.090000        NaN        NaN    NaN        NaN         NaN
1  77.03  75.589400  77.668294  73.510506    NaN        NaN         NaN
2  78.06  76.446090  78.493703  74.398477    NaN        NaN         NaN
3  79.91  77.364705  79.785135  74.944274   -1.0        NaN         NaN
4  82.22  78.414970  81.465249  75.364691   -1.0        NaN         NaN
>>>
>>> df.loc[(df.side_short == 1) | (df.side_long == 1)].head()
     close   ewm_mean      upper      lower  label  side_long  side_short
12   77.72  81.092143  84.249824  77.934462    1.0        1.0        -1.0
51   78.83  75.213832  77.892723  72.534942   -1.0       -1.0         1.0
108  76.53  80.911830  84.266327  77.557332    1.0        1.0        -1.0
168  81.99  79.154590  81.001386  77.307793   -1.0       -1.0         1.0
300  87.20  91.504157  94.573599  88.434715    1.0        1.0        -1.0
```

## How To Get Change the Sampling Frequency of a Time Series?

You have a time series of prices and you want to change the sampling frequency
of the time series. On the following example, we will use the `dollar_bars()`,
`time_bars()` and `volume_bars()` functions from the `fico.chronologicalsampling`
module. For that on your python console on the same directory of the repository,
run the following commands:

```python
>>> import pandas as pd
, time_bars, volume_bars
>>> from fico.chronologicalsampling import dollar_bars, time_bars, volume_bars
>>> data = pd.read_csv("data/data.csv")
>>> data["date"] = pd.to_datetime(data["date"])
>>> data = data.set_index("date")
>>> close = data["adj_close"]
>>> volume = data["volume"]
>>> time_bars_ohlc = time_bars(close, time_range="3d")
>>> print(time_bars_ohlc)
             close
              open    high     low   close
date
2002-12-31   42.94   44.64   42.94   44.64
2003-01-03   45.24   45.24   45.24   45.24
2003-01-06   46.32   47.65   46.32   46.65
2003-01-09   48.21   48.58   48.21   48.58
2003-01-12   48.49   49.08   48.49   49.08
...            ...     ...     ...     ...
2023-03-13  125.58  125.58  123.28  123.28
2023-03-16  124.70  124.70  123.69  123.69
2023-03-19  125.94  126.57  125.94  126.57
2023-03-22  124.05  125.29  123.37  125.29
2023-03-25  129.31  129.31  129.31  129.31

[2464 rows x 4 columns]
>>>
>>> dollar_bars_ohlc = dollar_bars(volume, close, dollar_amount=1e9)
>>> print(dollar_bars_ohlc)
             close                            volume
              open    high     low   close    volume
2002-12-31   42.94   44.64   42.94   44.64  16459751
2003-01-03   45.24   47.65   45.24   47.65  26976863
2003-01-08   46.65   48.21   46.65   48.21  21150539
2003-01-10   48.58   48.58   48.58   48.58  10413348
2003-01-13   48.49   49.08   48.49   48.53  27420577
...            ...     ...     ...     ...       ...
2023-03-16  124.70  124.70  124.70  124.70   6434528
2023-03-17  123.69  125.94  123.69  125.94  41914268
2023-03-21  126.57  126.57  124.05  124.05   7353969
2023-03-23  123.37  125.29  123.37  125.29   8453160
2023-03-27  129.31  129.31  129.31  129.31   6498029

[2490 rows x 5 columns]
>>>
>>> volume_bars_ohlc = volume_bars(volume, close, traded_volume=1.2e7)
>>> print(volume_bars_ohlc)
             close                            volume
              open    high     low   close    volume
date
2002-12-31   42.94   42.94   42.94   42.94   8233484
2003-01-02   44.64   45.24   44.64   45.24  14462833
2003-01-06   46.32   46.32   46.32   46.32   8285680
2003-01-07   47.65   47.65   47.65   47.65  12454617
2003-01-08   46.65   46.65   46.65   46.65   9946205
...            ...     ...     ...     ...       ...
2023-03-14  124.65  124.65  124.65  124.65   8103883
2023-03-15  123.28  124.70  123.28  124.70  12387764
2023-03-17  123.69  123.69  123.69  123.69  37339906
2023-03-20  125.94  126.57  124.05  124.05  11928331
2023-03-23  123.37  129.31  123.37  129.31  14951189

[2416 rows x 5 columns]
```

## How To Label a Time Series with a Triple Barrier Method?

You have a time series of prices and you want to label the time series with a
triple barrier method. On the following example, we will use the
`add_vertical_barrier()`,`get_bins()`,`get_events()`,`get_t_events()` functions from
the `fico.triplebarriers` module. For that on your python console on the same directory
of the repository, run the following commands:

```python
>>> import numpy as np
>>> import pandas as pd
>>> from fico.triplebarriers import (
...     add_vertical_barrier,
...     get_bins,
...     get_events,
...     get_t_events,
... )
>>>
>>> data = pd.read_csv("data/data.csv")
 pd.to_datetime(data["date"])
data = data.set_inde>>> data["date"] = pd.to_datetime(data["date"])
>>> data = data.set_index("date")
>>> close = data["adj_close"]
>>> close.name = "close"
>>> stock_df = pd.DataFrame(close)
>>> span0 = 50
>>> daily_vol = np.log(stock_df.close).diff().dropna().ewm(span=span0).std()
>>> daily_vol
date
2003-01-02         NaN
2003-01-03    0.018014
2003-01-06    0.012677
2003-01-07    0.010379
2003-01-08    0.023456
                ...
2023-03-21    0.011021
2023-03-22    0.011400
2023-03-23    0.011192
2023-03-24    0.011514
2023-03-27    0.013037
Name: close, Length: 5093, dtype: float64
>>> threshold = daily_vol.mean() * 1
>>> cusum_events = get_t_events(stock_df.close, threshold=threshold)
sum_events
>>> cusum_events
DatetimeIndex(['2003-01-03', '2003-01-06', '2003-01-07', '2003-01-08',
               '2003-01-09', '2003-01-14', '2003-01-16', '2003-01-17',
               '2003-01-22', '2003-01-23',
               ...
               '2023-02-21', '2023-02-28', '2023-03-06', '2023-03-07',
               '2023-03-09', '2023-03-15', '2023-03-20', '2023-03-22',
               '2023-03-24', '2023-03-27'],
              dtype='datetime64[ns]', length=1746, freq=None)
>>> vertical_barriers = add_vertical_barrier(
...     t_events=cusum_events,
...     close=close,
...     num_days=5,
...     )
>>> vertical_barriers
2003-01-03   2003-01-08
2003-01-06   2003-01-13
2003-01-07   2003-01-13
2003-01-08   2003-01-13
2003-01-09   2003-01-14
                ...
2023-03-07   2023-03-13
2023-03-09   2023-03-14
2023-03-15   2023-03-20
2023-03-20   2023-03-27
2023-03-22   2023-03-27
Name: date, Length: 1744, dtype: datetime64[ns]
>>> pt_sl = np.array([1, 2])
>>> min_ret = 0.0005
>>> triple_barrier_events = get_events(
...     close=stock_df.close,
...     t_events=cusum_events,
...     pt_sl=pt_sl,
...     target=daily_vol,
...     min_ret=min_ret,
...     vertical_barriers=vertical_barriers,
... )
events
>>>
>>> triple_barrier_events
                   t1      trgt
2003-01-03 2003-01-06  0.018014
2003-01-06 2003-01-07  0.012677
2003-01-07 2003-01-08  0.010379
2003-01-08 2003-01-09  0.023456
2003-01-09 2003-01-14  0.022007
...               ...       ...
2023-03-15 2023-03-16  0.010597
2023-03-20 2023-03-22  0.011161
2023-03-22 2023-03-27  0.011400
2023-03-24 2023-03-27  0.011514
2023-03-27        NaT  0.013037

[1746 rows x 2 columns]
>>> labels = get_bins(triple_barrier_events, stock_df.close, True)
()
>>> labels.bin.value_counts()
bin
 1    667
-1    556
 0    522
Name: count, dtype: int64
>>> labels.head()
                 ret  bin      trgt
2003-01-03  0.023873    1  0.018014
2003-01-06  0.028713    1  0.012677
2003-01-07 -0.020986   -1  0.010379
2003-01-08  0.033441    1  0.023456
2003-01-09  0.018046    0  0.022007
```
