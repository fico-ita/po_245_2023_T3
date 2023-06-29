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

## How To Get Bollinger Bands Series?

You have a time series of prices and you want to get the Bollinger Bands series.
On the following example, we will use the `bollingerbands()` function from the
`fico.technicalindicators` module. For that on your python console on the same
directory of the repository, run the following commands:

```python
>>> import pandas as pd
>>> data = pd.read_csv("data/data.csv")
>>> close = data["close"]
>>> mean_ewm, upper_band, lower_band = bollingerbands(close, 50, 2)
>>> print(mean_ewm)
0        74.090000
1        75.589400
2        76.446090
3        77.364705
4        78.414970
           ...
5089    132.199143
5090    131.879569
5091    131.545860
5092    131.300533
5093    131.222472
Name: close, Length: 5094, dtype: float64
>>> print(upper_band)
0              NaN
1        79.747188
2        80.541316
3        82.205566
4        84.515527
           ...
5089    146.176307
5090    145.947713
5091    145.733444
5092    145.421881
5093    145.086156
Name: close, Length: 5094, dtype: float64
>>> print(lower_band)
0              NaN
1        71.431612
2        72.350864
3        72.523843
4        72.314413
           ...
5089    118.221979
5090    117.811425
5091    117.358277
5092    117.179184
5093    117.358789
Name: close, Length: 5094, dtype: float64
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
>>> dollar_bars_ohlc = dollar_bars(volume, close, dollar_amount=1e9)
>>> volume_bars_ohlc = volume_bars(volume, close, traded_volume=1.2e7)
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
