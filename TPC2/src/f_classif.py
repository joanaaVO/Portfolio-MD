import numpy as np
from scipy.stats import f_oneway

import sys
sys.path.append('./TPC1/src')

from dataset import Dataset

class F_Classif:
    """
    Class computes F-test and p-values for each feature in a dataset.
    """
    
    def __init__(self):
        # attributes
        self.fvalues = None
        self.pvalues = None
    
    def fit(self, dataset: Dataset) -> 'F_Classif':
        """
        Fit the F-test model to the dataset.

        Arguments:
        - dataset: Dataset object containing the input features and labels.

        Returns:
        - self: F_Classif object.
        """
        y = dataset.y
        # Group samples/examples by class
        classes = np.unique(y)
        X_classes = [dataset.X[y == c] for c in classes]
        # Calculate F and p values for each feature
        f_values = []
        p_values = []
        for i in range(dataset.X.shape[1]):
            f, p = f_oneway(*[X[:, i] for X in X_classes])
            f_values.append(f)
            p_values.append(p)
        # Store F and p values
        self.f_values = np.array(f_values)
        self.p_values = np.array(p_values)
        
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting features with p-value under threshold.

        Arguments:
        - dataset: Dataset object containing the input features and labels.

        Returns:
        - transformed_dataset: Dataset object with selected features.
        """
        mask = self.p_values < 0.05
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)  
