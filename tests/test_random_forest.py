import numpy as np
import pytest
from src.mltools.supervised.random_forest import RandomForestClassifier, RandomForestRegressor


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X0 = np.random.randn(50, 2) - 2
    X1 = np.random.randn(50, 2) + 2
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1]
    return X, y


def test_classifier_fit_predict(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_classifier_accuracy(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y)
    assert model.score(X, y) > 0.90


def test_classifier_binary_output(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_regressor_fit_predict(regression_data):
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=10, max_depth=3)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_regressor_r2(regression_data):
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=10, max_depth=3)
    model.fit(X, y)
    assert model.score(X, y) > 0.80