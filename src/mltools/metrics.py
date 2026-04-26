import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    Compute R-squared (coefficient of determination).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def precision_score(y_true, y_pred):
    """
    Compute precision for binary classification.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-9)


def recall_score(y_true, y_pred):
    """
    Compute recall for binary classification.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn + 1e-9)


def f1_score(y_true, y_pred):
    """
    Compute F1 score for binary classification.
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-9)


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for binary classification.
    Returns [[TN, FP], [FN, TP]]
    """
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])