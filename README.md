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
