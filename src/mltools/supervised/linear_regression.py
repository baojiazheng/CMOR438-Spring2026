import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate for gradient descent.
    epochs : int, default=1000
        Number of training iterations.
    method : str, default='ols'
        'ols' for closed-form solution, 'gd' for gradient descent.
    """

    def __init__(self, lr=0.01, epochs=1000, method='ols'):
        self.lr = lr
        self.epochs = epochs
        self.method = method
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.method == 'ols':
            X_b = np.c_[np.ones((n_samples, 1)), X]
            self.weights_full = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = self.weights_full[0]
            self.weights = self.weights_full[1:]
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0
            for _ in range(self.epochs):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y
                dw = X.T @ error / n_samples
                db = error.mean()
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                loss = np.mean(error ** 2)
                self.losses.append(loss)
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def rmse(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))