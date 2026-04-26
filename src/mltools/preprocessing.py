import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MinMaxScaler:
    """
    Scale features to a given range [0, 1].
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        denom = self.max_ - self.min_
        denom[denom == 0] = 1
        return (X - self.min_) / denom

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    test_size : float, default=0.2
    random_state : int, optional
    """
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    idx = np.random.permutation(n)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]