import numpy as np
import pytest
from src.mltools.supervised.mlp import MLP


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
    model = MLP(layer_sizes=[8, 4], lr=0.01, epochs=50)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_accuracy(binary_data):
    X, y = binary_data
    model = MLP(layer_sizes=[16, 8], lr=0.1, epochs=200)
    model.fit(X, y)
    assert model.score(X, y) > 0.75


def test_binary_output(binary_data):
    X, y = binary_data
    model = MLP(layer_sizes=[8, 4], lr=0.01, epochs=50)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_predict_proba_range(binary_data):
    X, y = binary_data
    model = MLP(layer_sizes=[8, 4], lr=0.01, epochs=50)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_loss_decreases(binary_data):
    X, y = binary_data
    model = MLP(layer_sizes=[16, 8], lr=0.1, epochs=200)
    model.fit(X, y)
    assert model.losses[-1] < model.losses[0]