import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression using gradient descent.

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
        self.losses = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            loss = -np.mean(y * np.log(y_pred + 1e-9) +
                           (1 - y) * np.log(1 - y_pred + 1e-9))
            self.losses.append(loss)
            dw = X.T @ (y_pred - y) / n_samples
            db = np.mean(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)