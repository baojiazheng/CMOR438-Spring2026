import numpy as np
import pytest
from src.mltools.unsupervised.kmeans import KMeans


@pytest.fixture
def cluster_data():
    np.random.seed(42)
    X0 = np.random.randn(50, 2) - 3
    X1 = np.random.randn(50, 2) + 3
    X2 = np.random.randn(50, 2) * 0.5
    X = np.vstack([X0, X1, X2])
    return X


def test_fit_labels_shape(cluster_data):
    X = cluster_data
    model = KMeans(k=3)
    model.fit(X)
    assert model.labels_.shape == (len(X),)


def test_correct_number_of_clusters(cluster_data):
    X = cluster_data
    model = KMeans(k=3)
    model.fit(X)
    assert len(np.unique(model.labels_)) == 3


def test_centroids_shape(cluster_data):
    X = cluster_data
    model = KMeans(k=3)
    model.fit(X)
    assert model.centroids.shape == (3, X.shape[1])


def test_inertia_positive(cluster_data):
    X = cluster_data
    model = KMeans(k=3)
    model.fit(X)
    assert model.inertia_ >= 0


def test_predict(cluster_data):
    X = cluster_data
    model = KMeans(k=3)
    model.fit(X)
    labels = model.predict(X)
    assert labels.shape == (len(X),)


def test_fit_predict(cluster_data):
    X = cluster_data
    model = KMeans(k=3)
    labels = model.fit_predict(X)
    assert labels.shape == (len(X),)