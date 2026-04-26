import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        return self.y_train[k_indices].mean()

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot