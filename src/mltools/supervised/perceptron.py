import numpy as np


class Perceptron:
    """
    Single-layer Perceptron for binary classification.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    epochs : int, default=1000
        Number of training iterations.
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict_single(xi)
                update = self.lr * (yi - y_pred)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def predict_single(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)