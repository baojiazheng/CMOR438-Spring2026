import numpy as np
import pytest
from src.mltools.unsupervised.dbscan import DBSCAN


@pytest.fixture
def cluster_data():
    np.random.seed(42)
    X0 = np.random.randn(50, 2) * 0.3
    X1 = np.random.randn(50, 2) * 0.3 + 5
    X = np.vstack([X0, X1])
    return X


def test_fit_labels_shape(cluster_data):
    X = cluster_data
    model = DBSCAN(eps=1.0, min_samples=5)
    model.fit(X)
    assert model.labels_.shape == (len(X),)


def test_finds_clusters(cluster_data):
    X = cluster_data
    model = DBSCAN(eps=1.0, min_samples=5)
    model.fit(X)
    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    assert n_clusters >= 2


def test_fit_predict(cluster_data):
    X = cluster_data
    model = DBSCAN(eps=1.0, min_samples=5)
    labels = model.fit_predict(X)
    assert labels.shape == (len(X),)


def test_noise_points(cluster_data):
    X = cluster_data
    model = DBSCAN(eps=0.1, min_samples=20)
    model.fit(X)
    assert -1 in model.labels_