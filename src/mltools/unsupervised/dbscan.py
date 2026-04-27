import numpy as np


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples to be considered neighbors.
    min_samples : int, default=5
        Minimum number of samples in a neighborhood to form a core point.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = len(X)
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue
            neighbors = self._get_neighbors(X, i)
            if len(neighbors) < self.min_samples:
                continue
            self._expand_cluster(X, i, neighbors, cluster_id)
            cluster_id += 1
        return self

    def _get_neighbors(self, X, idx):
        distances = np.sqrt(np.sum((X - X[idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id):
        self.labels_[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = cluster_id
                new_neighbors = self._get_neighbors(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])
            i += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_