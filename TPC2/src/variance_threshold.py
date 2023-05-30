import numpy as np

import sys
sys.path.append('./TPC1/src')

from dataset import Dataset

class VarianceThreshold:

    def __init__(self, threshold: float = 0.0):
        """
        Initialize VarianceThreshold object.

        Args:
        - threshold: Variance threshold for feature selection (default is 0.0).
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model to the dataset.

        Args:
        - dataset: Dataset object containing the input features and labels.

        Returns:
        - self: VarianceThreshold object.
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting features with variance above the threshold.

        Args:
        - dataset: Dataset object containing the input features and labels.

        Returns:
        - transformed_dataset: Transformed Dataset object.
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)