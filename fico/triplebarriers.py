"""Provide the functions for calculating the triple barrier labels with meta-labeling.

This module allows to perform the triple barrier method with meta-labeling.

The module contains the following functions:

- add_vertical_barrier(t_events, close, num_days) - Adds a vertical barrier at a
certain number of days after the event.
- apply_pt_sl_on_t1(close, events, pt_sl, molecule) - Applies the stop loss and
profit taking, if it takes place before the vertical barrier.
- barrier_touched(df) - Determine whether the returns have touched a barrier.
- get_bins(events, close) - Labels the events
- get_events(close, t_events, pt_sl, target, min_ret, num_threads,
vertical_barrier_times, side) - Finds the timestamps of the events, when the price
hits the threshold.
- get_t_events(close, threshold) - Finds the timestamps of the events, when the
price hits the threshold.
- get_volatility(close, window_length, delta) - Computes the daily volatility at
intraday estimation points, applying a span of delta days to an exponentially
weighted moving standard deviation.


"""

import numpy as np
import pandas as pd


def get_volatility(
    close: pd.Series,
    window_length: int = 100,
    delta: pd.Timedelta = pd.Timedelta(days=1),  # noqa: B008
) -> pd.Series:
    """Volatility estimation using a rolling window.

    Computes the daily volatility at intraday estimation points, applying a span of
    delta days to an exponentially weighted moving standard deviation.

    Args:
        close (pd.Series): A pandas series of prices.
        window_length (int, optional): The window length used to calculate the \
            volatility.
        delta (pd.Timedelta): The time period used to calculate the volatility.

    Returns:
        volatility (pd.Series): A pandas series of volatility values.

    """
    # Find the prices for the last delta period
    previous_timestamps = close.index.searchsorted(close.index - delta)
    previous_timestamps = previous_timestamps[previous_timestamps > 0]
    previous_timestamps = pd.Series(
        close.index[previous_timestamps - 1],
        index=close.index[close.shape[0] - previous_timestamps.shape[0] :],
    )

    # Calculate the returns
    returns = (
        close.loc[previous_timestamps.index]
        / close.loc[previous_timestamps.to_numpy()].to_numpy()
        - 1
    )

    # Calculate the volatility
    volatility = returns.ewm(span=window_length).std()

    return volatility  # noqa: RET504


def get_t_events(close: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Finds the timestamps of the events using a Symmetric CUSUM filter.

    Lam and Yam [1997] propose an investment strategy whereby alternating buy-sell
    signals are generated when an absolute return h is observed relative to a prior
    high or low. Those authors demonstrate that such strategy is equivalent to the
    so-called “filter trading strategy” studied by Fama and Blume [1966]. Our use of
    the CUSUM filter is different: We will sample a bar t if and only if St ≥ h, at
    which point St is reset.

    Parameters:
        close (pd.Series): Series of close prices.
        threshold (float): When the absolute change is larger than the threshold, the \
            function captures it as an event.

    Returns:
        events_timestamps (pd.DatetimeIndex): Vector of datetimes when the events \
            occurred.
    """
    # Initialize variables
    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    diff = np.log(close).diff().dropna()

    # Get event time stamps for the entire series if the threshold is surpassed
    # accumulatively
    for i in diff.index[1:]:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])

        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)

        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)

    # Return DatetimeIndex
    events_timestamps = pd.DatetimeIndex(t_events)

    return events_timestamps  # noqa: RET504


def add_vertical_barrier(
    t_events: pd.Series,
    close: pd.Series,
    num_days: int = 1,
) -> pd.Series:
    """Adds a vertical barrier at a certain number of days after the event.

    For each index in t_events, it finds the timestamp of the next price bar at or
    immediately after a number of days num_days. This vertical barrier can be passed
    as an optional argument t_vertical_barrier in get_events.

    Parameters:
        t_events (pd.Series): Series of events (symmetric CUSUM filter).
        close (pd.Series): Close prices.
        num_days (int): Maximum number of days a trade can be active.

    Returns:
        t_vertical_barrier (pd.Series): Timestamps of vertical barriers.

    """
    # Find index of next bar
    t_vertical_barrier = close.index.searchsorted(
        t_events + pd.Timedelta(days=num_days),
    )

    # Limit the index within the range of close index
    t_vertical_barrier = t_vertical_barrier[t_vertical_barrier < close.shape[0]]

    # Generate vertical barrier series
    t_vertical_barrier = pd.Series(
        close.index[t_vertical_barrier],
        index=t_events[: t_vertical_barrier.shape[0]],
    )  # NaNs at end

    return t_vertical_barrier


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: np.ndarray,
    molecule: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Labels observations according to the first barrier touched out of three barriers.

    First, we set two horizontal barriers and one vertical barrier. The two horizontal
    barriers are defined by profit-taking and stop-loss limits, which are a dynamic
    function of estimated volatility (whether realized or implied). The third barrier
    is defined in terms of the number of bars elapsed since the position was taken
    (an expiration limit).

    Embedded a meta-labeling method.

    Parameters:
        close (pd.Series): Close prices.
        events (pd.DataFrame): A pandas dataframe with columns: \\
            - t1: The timestamp of vertical barrier. When the value is np.nan, there
            will not be a vertical barrier. \\
            - trgt: The unit width of the horizontal barriers. \\
        pt_sl (np.ndarray): A list of two non-negative float values: \\
            - Element 0 indicates the profit-taking level. \\
            - Element 1 is the stop-loss level. \\
        molecule (pd.DatetimeIndex): A list with the subset of event indices that will
            be processed by a single thread.

    Returns:
        t_touches (pd.DataFrame): Dataframe with timestamps at which each barrier was
            touched.

    """  # noqa: D301
    # apply stop loss/profit taking if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    t_touches = events_[["t1"]].copy(deep=True)

    # Profit taking active
    if pt_sl[0] > 0:  # noqa: SIM108
        pt = pt_sl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if pt_sl[1] > 0:  # noqa: SIM108
        sl = -pt_sl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs

    # Get events' timestamps
    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.loc[loc, "side"]  # path returns
        t_touches.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        t_touches.loc[loc, "pt"] = df0[
            df0 > pt[loc]
        ].index.min()  # earliest profit take

    return t_touches


def get_events(  # noqa: PLR0913
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: np.ndarray,
    target: pd.Series,
    min_ret: float,
    vertical_barriers=False,
    side=None,
) -> pd.DataFrame:
    """Get the time of the first barrier touch.

    Finds the time of the first barrier touch, either horizontal or vertical,
    embedding a meta-labeling method.

    Parameters:
        close (pd.Series): Close prices.
        t_events (pd.DatetimeIndex): Series of t_events. These are timestamps that
        will seed every triple barrier.
        pt_sl (np.ndarray): 2-element array: \\
            - Element 0 indicates the profit-taking level. \\
            - Element 1 is the stop-loss level. \\
              A non-negative float that sets the width of the two barriers. \\
              A value of 0 means that the respective horizontal barrier will be
              disabled. \\
        target (pd.Series): Series of values that are used (in conjunction with pt_sl)
            to determine the width of the barrier. \\
        min_ret (float): The minimum target return required for running a triple
            barrier search. \\
        vertical_barriers (pd.Series): A pandas series with the timestamps of the
            vertical barriers. \\
        side (pd.Series): Side of the bet (long/short) as decided by the primary model.

    Returns:
        events (pd.DataFrame): Dataframe of events: \\
            - events.index is event's starttime \\
            - events['t1'] is event's endtime \\
            - events['trgt'] is event's target \\
            - events['side'] (optional) implies the algo's position side
    """  # noqa: D301
    # 1) Get target
    target = target.loc[target.index.intersection(t_events)]
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barriers is False:
        vertical_barriers = pd.Series(pd.NaT, index=t_events)

    # 3) Form events object, apply stop loss on vertical barrier
    if side is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side.loc[target.index]
        pt_sl_ = pt_sl[:2]

    events = pd.concat({"t1": vertical_barriers, "trgt": target, "side": side_}, axis=1)

    events = events.dropna(subset=["trgt"])

    # Apply Triple Barrier
    df0 = apply_pt_sl_on_t1(
        close=close,
        events=events,
        pt_sl=pt_sl_,
        molecule=target.index,
    )

    events["t1"] = df0.dropna(how="all").min(axis=1)  # pd.min ignores nan

    if side is None:
        events = events.drop("side", axis=1)

    return events


def barrier_touched(df: pd.DataFrame) -> pd.DataFrame:
    """Determine whether the returns have touched a barrier.

    Parameters:
        df (pd.DataFrame): containing the returns and target


    Returns:
        df (pd.DataFrame): containing returns, target, and labels

    """
    # Arrange data for labelling
    store = []
    t = 0.0
    for i in np.arange(len(df)):
        date_time = df.index[i]
        ret = df.loc[date_time, "ret"]
        target = df.loc[date_time, "trgt"]

        if ret > t and ret > target:
            # Top barrier reached
            store.append(1)
        elif ret < t and ret < -target:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)

    df["bin"] = store

    return df


def get_bins(
    triple_barrier_events: pd.DataFrame,
    close: pd.Series,
    vertical_touch: bool = False,
) -> pd.DataFrame:
    """Label the observations.

    Parameters:
        triple_barrier_events: (pd.DataFrame):
            -events.index is event's starttime \\
            -events['t1'] is event's endtime \\
            -events['trgt'] is event's target \\
            -events['side'] (optional) implies the algo's position side \\
            Case 1: ('side' not in events): bin in (-1,1) <-label by price action \\
            Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling) \\

        close (pd.Dataframe): Close prices

        vertical_touch (bool): Whether to use vertical barrier or not and bin the
            observations accordingly (-1, 0, 1)

    Returns:
        out_df (pd.Dataframe): Data frmae of meta-labeled events

    """  # noqa: D301
    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=["t1"])
    prices = events_.index.union(events_["t1"].to_numpy())
    prices = prices.drop_duplicates()
    prices = close.reindex(prices, method="bfill")

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=events_.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df["ret"] = (
        prices.loc[events_["t1"].to_numpy()].to_numpy() / prices.loc[events_.index] - 1
    )

    # Meta labeling: Events that were correct will have pos returns
    if "side" in events_:
        out_df["ret"] = out_df["ret"] * events_["side"]  # meta-labeling

    out_df["bin"] = np.sign(out_df["ret"])

    # Added code: label 0 when vertical barrier reached
    if vertical_touch:
        out_df["trgt"] = events_["trgt"]
        out_df = barrier_touched(out_df)

    # Meta labeling: label incorrect events with a 0
    if "side" in events_:
        out_df.loc[out_df["ret"] <= 0, "bin"] = 0

    return out_df
