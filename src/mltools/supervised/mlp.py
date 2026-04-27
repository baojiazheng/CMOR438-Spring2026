import numpy as np


class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Neural Network).

    Parameters
    ----------
    layer_sizes : list
        List of integers specifying the number of neurons in each layer.
        e.g. [64, 32] means two hidden layers with 64 and 32 neurons.
    lr : float, default=0.01
        Learning rate.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Mini-batch size for stochastic gradient descent.
    """

    def __init__(self, layer_sizes, lr=0.01, epochs=100, batch_size=32):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.losses = []

    def _init_weights(self, n_features):
        sizes = [n_features] + self.layer_sizes + [1]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * 0.01
            b = np.zeros((1, sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X):
        self.activations = [X]
        self.z_values = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self.z_values.append(z)
            if i == len(self.weights) - 1:
                a = self._sigmoid(z)
            else:
                a = self._relu(z)
            self.activations.append(a)
        return a

    def _backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        delta = self.activations[-1] - y
        for i in reversed(range(len(self.weights))):
            dW = self.activations[i].T @ delta / m
            db = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_deriv(self.z_values[i-1])
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    def fit(self, X, y):
        self._init_weights(X.shape[1])
        for epoch in range(self.epochs):
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
            for start in range(0, len(X), self.batch_size):
                X_batch = X[start:start+self.batch_size]
                y_batch = y[start:start+self.batch_size]
                self._forward(X_batch)
                self._backward(X_batch, y_batch)
            y_pred = self._forward(X)
            loss = -np.mean(y.reshape(-1,1) * np.log(y_pred + 1e-9) +
                           (1 - y.reshape(-1,1)) * np.log(1 - y_pred + 1e-9))
            self.losses.append(loss)
        return self

    def predict_proba(self, X):
        return self._forward(X).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)