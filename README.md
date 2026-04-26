# CMOR 438 Data Science & Machine Learning

---

## About

This repository contains my coursework, implementations, and final project for **CMOR 438: Data Science and Machine Learning** at Rice University. The goal is to build a comprehensive, well-documented body of work covering classical and modern machine learning algorithms — implemented from scratch and applied to real datasets.

---

## Repository Structure

```
CMOR438-Spring2026/
│
├── notebooks/                        # Jupyter notebooks organized by topic
│   ├── 01_data_preprocessing/
│   ├── 02_linear_models/
│   ├── 03_knn/
│   ├── 04_decision_trees_ensembles/
│   ├── 05_clustering/
│   ├── 06_dimensionality_reduction/
│   ├── 07_neural_networks/
│   └── 08_svm/
│
├── src/
│   └── mltools/                      # Custom ML package (implemented from scratch)
│       ├── __init__.py
│       ├── supervised/
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   └── knn.py
│       ├── unsupervised/
│       │   ├── kmeans.py
│       │   └── pca.py
│       ├── preprocessing.py
│       └── metrics.py
│
├── examples/
│   └── demo_notebook.ipynb           # End-to-end demo using mltools package
│
├── tests/                            # Unit tests with pytest
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_knn.py
│   ├── test_kmeans.py
│   └── test_metrics.py
│
├── requirements.txt
├── setup.py
└── README.md
```

## Algorithms

### Supervised Learning
- **Linear Regression** — Ordinary Least Squares (OLS) with RMSE and R² evaluation.
- **Ridge / Lasso Regression** — Regularized linear models to handle overfitting.
- **Logistic Regression** — Batch gradient descent with sigmoid activation and accuracy metrics.
- **K-Nearest Neighbors (KNN)** — Distance-based classifier supporting multiple distance metrics.
- **Decision Trees** — Recursive tree-building using information gain (classification) or variance reduction (regression).
- **Random Forests** — Ensemble of decision trees with bagging for improved generalization.
- **Gradient Boosting** — Sequential ensemble method minimizing residual errors.
- **Support Vector Machines (SVM)** — Margin-maximizing classifier with kernel support.
- **Multi-Layer Perceptron (MLP)** — Feedforward neural network with backpropagation and cross-entropy loss.

### Unsupervised Learning
- **K-Means Clustering** — Centroid-based clustering with inertia computation.
- **Hierarchical Clustering** — Agglomerative clustering with dendrogram visualization.
- **DBSCAN** — Density-based clustering for arbitrary-shaped clusters and noise detection.
- **PCA** — Dimensionality reduction and variance analysis.
- **t-SNE** — Non-linear dimensionality reduction for high-dimensional data visualization.

---

## Utilities

- **Preprocessing**
  - `MinMaxScaler`, `StandardScaler` — Feature scaling.
  - `OrdinalEncoder` — Convert categorical features to numerical codes.
  - `train_test_split` — Flexible train/test partitioning.
- **Metrics**
  - `accuracy_score`, `classification_report`, RMSE, R².
- **Visualization**
  - Decision boundaries for classifiers.
  - Loss curves for neural networks.
  - Cluster plots for unsupervised learning.

---

## Example Datasets

### House Prices — Advanced Regression Techniques
- **Source:** [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

### Credit Card Fraud Detection
- **Source:** [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Testing

- Unit tests cover:
  - Correct implementation of ML algorithms.
  - Edge cases (empty inputs, invalid parameters).
  - Preprocessing utilities (scalers, encoders, split functions).
  - Metric computation and evaluation functions.
- Run tests with:

```bash
pytest
```

---

## Installation

```bash
git clone https://github.com/baojiazheng/CMOR438-Spring2026.git
cd CMOR438-Spring2026
pip install -e .
```

---

## Getting Started

```python
from src.mltools import LinearRegression, KNN, train_test_split, MinMaxScaler

# Load data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
print("R²:", model.score(X_test_scaled, y_test))
```
