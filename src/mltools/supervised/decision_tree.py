import numpy as np


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using information gain (entropy).

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _best_split(self, X, y):
        best_gain, best_feature, best_threshold = 0, None, None
        base_entropy = self._entropy(y)
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = base_entropy - (
                    len(left) / len(y) * self._entropy(left) +
                    len(right) / len(y) * self._entropy(right)
                )
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.bincount(y).argmax()
        left_mask = X[:, feature] <= threshold
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        }

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        return self._predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class DecisionTreeRegressor:
    """
    Decision Tree Regressor using variance reduction.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        best_var_red, best_feature, best_threshold = 0, None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                var_red = np.var(y) - (
                    len(left) / len(y) * np.var(left) +
                    len(right) / len(y) * np.var(right)
                )
                if var_red > best_var_red:
                    best_var_red, best_feature, best_threshold = var_red, feature, threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y)
        left_mask = X[:, feature] <= threshold
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        }

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        return self._predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot