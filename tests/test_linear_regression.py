import numpy as np
import pytest
from src.mltools.supervised.linear_regression import LinearRegression


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
    return X, y


def test_ols_fit_predict(simple_data):
    X, y = simple_data
    model = LinearRegression(method='ols')
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_ols_r2_high(simple_data):
    X, y = simple_data
    model = LinearRegression(method='ols')
    model.fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.95


def test_gd_fit_predict(simple_data):
    X, y = simple_data
    model = LinearRegression(method='gd', lr=0.01, epochs=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_rmse_positive(simple_data):
    X, y = simple_data
    model = LinearRegression(method='ols')
    model.fit(X, y)
    rmse = model.rmse(X, y)
    assert rmse >= 0


def test_single_feature():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    model = LinearRegression(method='ols')
    model.fit(X, y)
    assert model.score(X, y) > 0.99