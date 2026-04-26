import numpy as np


class KMeans:
    """
    K-Means Clustering.

    Parameters
    ----------
    k : int, default=3
        Number of clusters.
    max_iters : int, default=100
        Maximum number of iterations.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()

        for _ in range(self.max_iters):
            self.labels_ = self._assign(X)
            new_centroids = np.array([
                X[self.labels_ == k].mean(axis=0)
                if (self.labels_ == k).sum() > 0
                else self.centroids[k]
                for k in range(self.k)
            ])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.inertia_ = sum(
            np.sum((X[self.labels_ == k] - self.centroids[k]) ** 2)
            for k in range(self.k)
        )
        return self

    def _assign(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_