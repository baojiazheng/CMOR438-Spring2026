import numpy as np
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    """
    Random Forest Classifier using bagging of Decision Trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=5
        Maximum depth of each tree.
    max_features : float, default=0.5
        Fraction of features to consider at each split.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=5, max_features=0.5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        self.feature_indices = []
        n_samples, n_features = X.shape
        n_feats = max(1, int(n_features * self.max_features))

        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idx], y[idx]
            # Random feature subset
            feat_idx = np.random.choice(n_features, n_feats, replace=False)
            self.feature_indices.append(feat_idx)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample[:, feat_idx], y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        return np.array([
            np.bincount(predictions[:, i]).argmax()
            for i in range(X.shape[0])
        ])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class RandomForestRegressor:
    """
    Random Forest Regressor using bagging of Decision Trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=5
        Maximum depth of each tree.
    max_features : float, default=0.5
        Fraction of features to consider at each split.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=5, max_features=0.5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        self.feature_indices = []
        n_samples, n_features = X.shape
        n_feats = max(1, int(n_features * self.max_features))

        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idx], y[idx]
            feat_idx = np.random.choice(n_features, n_feats, replace=False)
            self.feature_indices.append(feat_idx)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sample[:, feat_idx], y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        return predictions.mean(axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot