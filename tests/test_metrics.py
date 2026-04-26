import numpy as np
import pytest
from src.mltools.metrics import (
    accuracy_score,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def test_accuracy_perfect():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    assert accuracy_score(y_true, y_pred) == 1.0


def test_accuracy_zero():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    assert accuracy_score(y_true, y_pred) == 0.0


def test_mse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == 0.0


def test_mse_positive():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mean_squared_error(y_true, y_pred) == 1.0


def test_rmse_positive():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert root_mean_squared_error(y_true, y_pred) == 1.0


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0


def test_r2_range():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.9])
    assert -1 <= r2_score(y_true, y_pred) <= 1.0


def test_precision():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    p = precision_score(y_true, y_pred)
    assert 0 <= p <= 1


def test_recall():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    r = recall_score(y_true, y_pred)
    assert 0 <= r <= 1


def test_f1():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    f1 = f1_score(y_true, y_pred)
    assert 0 <= f1 <= 1


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_confusion_matrix_values():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert cm[1][1] == 2  # TP
    assert cm[0][0] == 2  # TN