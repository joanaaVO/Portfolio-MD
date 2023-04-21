import numpy as np

import sys
sys.path.append('./TPC1')

from dataset import Dataset

class SelectKBest:

    def __init__(self, score_func: callable, k: int):

        if k < 0:
            raise ValueError("k must be non-negative")

        # parameters
        self.score_func = score_func
        self.k = k

        # attributes
        self.scores = None
        self.pvalues = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        score_func_fit = self.score_func.fit(dataset) # Fit score function to dataset
        self.scores = score_func_fit.fvalues # Extract fvalues from fitted score function
        self.pvalues = score_func_fit.pvalues # Extract pvalues from fitted score function
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        X = dataset.X

        # Get the indices of the top K features
        top_k_indices = np.argsort(self.pvalues)[:self.k]

        X = X[:, top_k_indices]
        features = np.array(dataset.features)[top_k_indices]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)