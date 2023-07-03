"""Utility functions for the FICO project.

Methods for cross-validation and plotting performance metrics.

The module contains the following functions:

- gap_kfold(x, y, kfold, test_position, gap_before, gap_after) - Performs gap K-fold
    cross-validation by splitting the data into train and test sets, with a gap around
    the test set.
- modelmetrics(x, y, rf_model=None, best_recall=False, plot=True) - Plot metrics of
    a random forest model, including the ROC curve and confusion matrix. And print the
    accuracy and classification report.

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def gap_kfold(x, y, kfold, test_position, gap_before, gap_after):  # noqa: PLR0913
    """Split the data into train and test sets, with a gap around the test set.

    Performs gap K-fold cross-validation by splitting the data into train and test
    sets, with a gap around the test set.

    Parameters:
        x (array-like): Input features.
        y (array-like): Target variable.
        kfold (int): Number of folds for cross-validation.
        test_position (int): Position of the test fold.
        gap_before (int): Number of folds before the test fold to exclude.
        gap_after (int): Number of folds after the test fold to exclude.

    Returns:
        (tuple): A tuple containing the train and test data (x_train, y_train, x_test,
            y_test).
    """
    fold_size = int(x.shape[0] / kfold)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Determine the positions to exclude for the gap
    gap_positions = list(
        range(
            0 if test_position - gap_before < 0 else test_position - gap_before,
            kfold
            if test_position + gap_after > kfold
            else test_position + gap_after + 1,
        ),
    )

    # Remove the test position from the gap positions
    gap_positions.pop(gap_positions.index(test_position))

    # Split the data into train and test sets
    for i in range(kfold):
        if i == test_position:
            x_test.append(x[i * fold_size : i * fold_size + fold_size])
            y_test.append(y[i * fold_size : i * fold_size + fold_size])
        elif i in gap_positions:
            continue
        else:
            x_train.append(x[i * fold_size : i * fold_size + fold_size])
            y_train.append(y[i * fold_size : i * fold_size + fold_size])

    # Concatenate the train and test sets
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    return x_train, y_train, x_test, y_test


def modelmetrics(  # noqa: PLR0913
    x,
    y,
    rf_model=None,
    best_recall=False,
    train_threshold=None,
    plot=True,
):
    """Calculate performance metrics and plot results.

    Plot metrics of a random forest model, including the ROC curve and confusion matrix.
    And print the accuracy and classification report.

    Parameters:
        x (array-like): Input features for training data.
        y (array-like): Target variable for training data.
        rf_model (RandomForestClassifier, optional): Trained random forest model.
            Defaults to None.
        best_recall (bool, optional): If True, the threshold is set to the minimum
            probability of the positive class. Defaults to False.
        train_threshold (float, optional): Threshold for the positive class. Defaults
        plot (bool, optional): If True, plot the ROC curve and confusion matrix and
            print the classification report. Defaults to True.

    Returns:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        f1_score (float): F1 score.
        threshold (float): Threshold for the positive class.
    """
    # If no model is provided, predict all 1s, else predict using the model
    if rf_model is None:
        y_pred = np.ones(y.shape)
        threshold = 0.5
    else:
        y_pred_pb = rf_model.predict_proba(x)[:, 1]

        if train_threshold is not None:
            threshold = train_threshold
            y_pred = (y_pred_pb >= threshold).astype(float)
        elif best_recall:
            threshold = y_pred_pb[y == 1].min()
            y_pred = (y_pred_pb >= threshold).astype(float)
        else:
            threshold = 0.5
            y_pred = rf_model.predict(x)
        fpr_rf, tpr_rf, _ = roc_curve(y, y_pred_pb)

    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1_score, _ = classification_report(
        y,
        y_pred,
        # labels=np.unique(y_pred),
        zero_division=0,
        output_dict=True,
    )["1.0"].values()

    if plot:
        # Performance Metrics
        print(classification_report(y, y_pred, labels=np.unique(y_pred)))

        print("Accuracy")
        print(accuracy)

        if rf_model is not None:
            plt.figure(1)
            plt.plot([0, 1], [0, 1], "k--")
            plt.plot(fpr_rf, tpr_rf, label="RF")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")
            plt.legend(loc="best")
            plt.show()

        ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot()

    return accuracy, precision, recall, f1_score, threshold
