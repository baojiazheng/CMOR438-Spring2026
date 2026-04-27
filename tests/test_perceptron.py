import numpy as np
import pytest
from src.mltools.supervised.perceptron import Perceptron


@pytest.fixture
def binary_data():
    np.random.seed(42)
    X0 = np.random.randn(50, 2) - 2
    X1 = np.random.randn(50, 2) + 2
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


def test_fit_predict(binary_data):
    X, y = binary_data
    model = Perceptron(lr=0.01, epochs=100)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_accuracy(binary_data):
    X, y = binary_data
    model = Perceptron(lr=0.01, epochs=100)
    model.fit(X, y)
    assert model.score(X, y) > 0.85


def test_binary_output(binary_data):
    X, y = binary_data
    model = Perceptron(lr=0.01, epochs=100)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_errors_recorded(binary_data):
    X, y = binary_data
    model = Perceptron(lr=0.01, epochs=100)
    model.fit(X, y)
    assert len(model.errors_) == 100