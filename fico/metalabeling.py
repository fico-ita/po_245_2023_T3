"""Methods for metalabeling."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fico.triplebarriers import get_bins, get_events
from fico.utils import gap_kfold, modelmetrics


def metalabelingkfold(data, params, exogenous=None):
    """Implements the meta labeling strategy.

    Parameters:
        data (dict): A dictionary containing the following keys:
            - sampling_bars (pd.DataFrame): A DataFrame containing the sampled bars.
            - cusum_events (pd.Series): A Series containing the CUSUM events.
            - close (pd.Series): A Series containing the close prices.
            - daily_vol (pd.Series): A Series containing the daily volatility.
            - vertical_barriers (pd.Series): A Series containing the vertical barriers.
            - df_exogenous (pd.DataFrame): A DataFrame containing the exogenous features
                (optional).

        params (dict): A dictionary containing the following keys:
            - pt_sl (tuple): A tuple containing the profit taking and stop loss values.
            - min_ret (float): The minimum return required for a trade.
            - test_size (float): The proportion of the dataset to include in the test
                split.
            - gap_before (float): The size of the gap before the vertical barrier.
            - gap_after (float): The size of the gap after the vertical barrier.
            - kfold (int): The number of folds for cross-validation.
            - n_estimator (int): The number of trees in the random forest.
            - depth (int): The maximum depth of the trees in the random forest.

        exogenous (str): The type of exogenous features to use (optional). Can be one of
            "bollinger_bands" or "trend_following".

    Returns:
        results (pd.DataFrame): A DataFrame containing the results of the strategy.
    """
    sampling_bars = data["sampling_bars"]
    cusum_events = data["cusum_events"]
    close = data["close"]
    daily_vol = data["daily_vol"]
    vertical_barriers = data["vertical_barriers"]

    pt_sl = params["pt_sl"]
    min_ret = params["min_ret"]
    test_size = params["test_size"]
    gap_before = params["gap_before"]
    gap_after = params["gap_after"]
    kfold = params["kfold"]
    n_estimator = params["n_estimator"]
    depth = params["depth"]
    RANDOM_STATE = 0  # noqa: N806

    # Technical Analysis
    if exogenous is not None:
        df_exogenous = data["df_exogenous"]

        # Triple Barrier Method with Meta Labeling
        tbm_mlabel = get_events(
            close=close,
            t_events=cusum_events,
            pt_sl=pt_sl,
            target=daily_vol,
            min_ret=min_ret,
            vertical_barriers=vertical_barriers,
            side=df_exogenous.side,
        )

        # Labels
        labels = get_bins(tbm_mlabel, close)
        labels.bin = np.where(labels.bin == 1, 1.0, 0.0)

        # Features
        if exogenous == "bollinger_bands":
            features = pd.concat(
                [
                    sampling_bars.loc[labels.index, :].iloc[:, :-1],
                    df_exogenous.loc[
                        labels.index,
                        ["ewm_mean", "upper", "lower", "side"],
                    ],
                ],
                axis=1,
            )
        elif exogenous == "trend_following":
            features = pd.concat(
                [
                    sampling_bars.loc[labels.index, :].iloc[:, :-1],
                    df_exogenous.loc[labels.index, ["fast", "slow", "side"]],
                ],
                axis=1,
            )
    else:
        exogenous = "thm"
        # Triple Barrier Method
        tbm_mlabel = get_events(
            close=close,
            t_events=cusum_events,
            pt_sl=pt_sl,
            target=daily_vol,
            min_ret=min_ret,
            vertical_barriers=vertical_barriers,
        )

        # Labels
        labels = get_bins(tbm_mlabel, close)
        labels.bin = np.where(labels.bin == 1, 1.0, 0.0)

        # Features
        features = sampling_bars.loc[labels.index, :].iloc[:, :-1]

    x = features
    y = labels["bin"]

    # Train and Validation set
    x_tr, x_val, y_tr, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=False,
    )

    results = pd.DataFrame(
        columns=[
            "exogenous",
            "kfold_test",
            "accuracy_ck",
            "precision_ck",
            "recall_ck",
            "f1_score_ck",
            "threashold_ck",
            "accuracy_tr1",
            "precision_tr1",
            "recall_tr1",
            "f1_score_tr1",
            "threashold_tr1",
            "accuracy_tt1",
            "precision_tt1",
            "recall_tt1",
            "f1_score_tt1",
            "threashold_tt1",
            "accuracy_tr2",
            "precision_tr2",
            "recall_tr2",
            "f1_score_tr2",
            "threashold_tr2",
            "accuracy_tt2",
            "precision_tt2",
            "recall_tt2",
            "f1_score_tt2",
            "threashold_tt2",
        ],
    )

    for test_position in range(kfold):
        # splitting data into train and test
        x_train, y_train, x_test, y_test = gap_kfold(
            x_tr,
            y_tr,
            kfold,
            test_position,
            gap_before,
            gap_after,
        )

        # model metrics check with 0 threashold and signals 1
        accuracy_ck, precision_ck, recall_ck, f1_score_ck, threashold_ck = modelmetrics(
            x_train,
            y_train,
            plot=False,
        )

        # Primary model Random Forest
        rf = RandomForestClassifier(
            max_depth=depth,
            n_estimators=n_estimator,
            criterion="entropy",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        )

        # Fitting our model
        rf.fit(x_train, y_train)

        # model metrics for training set
        (
            accuracy_tr1,
            precision_tr1,
            recall_tr1,
            f1_score_tr1,
            threashold_tr1,
        ) = modelmetrics(
            x_train,
            y_train,
            rf,
            True,
            None,
            False,
        )

        # model metrics for test set
        (
            accuracy_tt1,
            precision_tt1,
            recall_tt1,
            f1_score_tt1,
            threashold_tt1,
        ) = modelmetrics(
            x_test,
            y_test,
            rf,
            False,
            threashold_tr1,
            False,
        )

        # concatenating predicted probabilities to the original values
        x_pred_tr = rf.predict_proba(x_train)[:, 1]
        x_train_2 = np.concatenate((x_train, x_pred_tr.reshape(-1, 1)), axis=1)

        x_pred_tt = rf.predict_proba(x_test)[:, 1]
        x_test_2 = np.concatenate((x_test, x_pred_tt.reshape(-1, 1)), axis=1)

        # Secondary model Random Forest
        rf_2 = RandomForestClassifier(
            max_depth=depth,
            n_estimators=n_estimator,
            criterion="entropy",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        )

        # Fitting our model
        rf_2.fit(x_train_2, y_train)

        (
            accuracy_tr2,
            precision_tr2,
            recall_tr2,
            f1_score_tr2,
            threashold_tr2,
        ) = modelmetrics(
            x_train_2,
            y_train,
            rf_2,
            False,
            None,
            False,
        )

        (
            accuracy_tt2,
            precision_tt2,
            recall_tt2,
            f1_score_tt2,
            threashold_tt2,
        ) = modelmetrics(
            x_test_2,
            y_test,
            rf_2,
            False,
            None,
            False,
        )

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [
                        {
                            "exogenous": exogenous,
                            "kfold_test": test_position + 1,
                            "accuracy_ck": accuracy_ck,
                            "precision_ck": precision_ck,
                            "recall_ck": recall_ck,
                            "f1_score_ck": f1_score_ck,
                            "threashold_ck": threashold_ck,
                            "accuracy_tr1": accuracy_tr1,
                            "precision_tr1": precision_tr1,
                            "recall_tr1": recall_tr1,
                            "f1_score_tr1": f1_score_tr1,
                            "threashold_tr1": threashold_tr1,
                            "accuracy_tt1": accuracy_tt1,
                            "precision_tt1": precision_tt1,
                            "recall_tt1": recall_tt1,
                            "f1_score_tt1": f1_score_tt1,
                            "threashold_tt1": threashold_tt1,
                            "accuracy_tr2": accuracy_tr2,
                            "precision_tr2": precision_tr2,
                            "recall_tr2": recall_tr2,
                            "f1_score_tr2": f1_score_tr2,
                            "threashold_tr2": threashold_tr2,
                            "accuracy_tt2": accuracy_tt2,
                            "precision_tt2": precision_tt2,
                            "recall_tt2": recall_tt2,
                            "f1_score_tt2": f1_score_tt2,
                            "threashold_tt2": threashold_tt2,
                        },
                    ],
                ),
            ],
            ignore_index=True,
        )

    return results


def metalabelingval(data, params, exogenous=None):
    """Implements the meta labeling strategy.

    Args:
        data (dict): A dictionary containing the following keys:
            - sampling_bars (pd.DataFrame): A DataFrame containing the sampled bars.
            - cusum_events (pd.Series): A Series containing the CUSUM events.
            - close (pd.Series): A Series containing the close prices.
            - daily_vol (pd.Series): A Series containing the daily volatility.
            - vertical_barriers (pd.Series): A Series containing the vertical barriers.
            - df_exogenous (pd.DataFrame): A DataFrame containing the exogenous features
                (optional).

        params (dict): A dictionary containing the following keys:
            - pt_sl (tuple): A tuple containing the profit taking and stop loss values.
            - min_ret (float): The minimum return required for a trade.
            - test_size (float): The proportion of the dataset to include in the test
                split.
            - n_estimator (int): The number of trees in the random forest.
            - depth (int): The maximum depth of the trees in the random forest.

        exogenous (str): The type of exogenous features to use (optional). Can be one of
            "bollinger_bands" or "trend_following".

    Returns:
        model (sklearn.ensemble.RandomForestClassifier): The trained model.
        x_train (array like): The training set.
        y_train (array like): The training labels.
        x_val (array like): The validation set.
        y_val (array like): The validation labels.
    """
    sampling_bars = data["sampling_bars"]
    cusum_events = data["cusum_events"]
    close = data["close"]
    daily_vol = data["daily_vol"]
    vertical_barriers = data["vertical_barriers"]

    pt_sl = params["pt_sl"]
    min_ret = params["min_ret"]
    test_size = params["test_size"]
    n_estimator = params["n_estimator"]
    depth = params["depth"]
    RANDOM_STATE = 0  # noqa: N806

    # Technical Analysis
    if exogenous is not None:
        df_exogenous = data["df_exogenous"]

        # Triple Barrier Method with Meta Labeling
        tbm_mlabel = get_events(
            close=close,
            t_events=cusum_events,
            pt_sl=pt_sl,
            target=daily_vol,
            min_ret=min_ret,
            vertical_barriers=vertical_barriers,
            side=df_exogenous.side,
        )

        # Labels
        labels = get_bins(tbm_mlabel, close)
        labels.bin = np.where(labels.bin == 1, 1.0, 0.0)

        # Features
        if exogenous == "bollinger_bands":
            features = pd.concat(
                [
                    sampling_bars.loc[labels.index, :].iloc[:, :-1],
                    df_exogenous.loc[
                        labels.index,
                        ["ewm_mean", "upper", "lower", "side"],
                    ],
                ],
                axis=1,
            )
        elif exogenous == "trend_following":
            features = pd.concat(
                [
                    sampling_bars.loc[labels.index, :].iloc[:, :-1],
                    df_exogenous.loc[labels.index, ["fast", "slow", "side"]],
                ],
                axis=1,
            )
    else:
        # Triple Barrier Method
        tbm_mlabel = get_events(
            close=close,
            t_events=cusum_events,
            pt_sl=pt_sl,
            target=daily_vol,
            min_ret=min_ret,
            vertical_barriers=vertical_barriers,
        )

        # Labels
        labels = get_bins(tbm_mlabel, close)
        labels.bin = np.where(labels.bin == 1, 1.0, 0.0)

        # Features
        features = sampling_bars.loc[labels.index, :].iloc[:, :-1]

    x = features
    y = labels["bin"]

    # Train and Validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=False,
    )

    # Primary model Random Forest
    rf = RandomForestClassifier(
        max_depth=depth,
        n_estimators=n_estimator,
        criterion="entropy",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )

    # Fitting our model
    rf.fit(x_train, y_train)

    # concatenating predicted probabilities to the original values
    x_pred_tr = rf.predict_proba(x_train)[:, 1]
    x_train_2 = np.concatenate((x_train, x_pred_tr.reshape(-1, 1)), axis=1)

    x_pred_tt = rf.predict_proba(x_val)[:, 1]
    x_val_2 = np.concatenate((x_val, x_pred_tt.reshape(-1, 1)), axis=1)

    # Secondary model Random Forest
    model = RandomForestClassifier(
        max_depth=depth,
        n_estimators=n_estimator,
        criterion="entropy",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )

    # Fitting our model
    model.fit(x_train_2, y_train)

    return model, x_train, y_train, x_val_2, y_val, features
