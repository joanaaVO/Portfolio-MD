import numpy as np
from scipy.stats import f_oneway

import sys
sys.path.append('./TPC1')

from dataset import Dataset

class F_Classif:
    
    def __init__(self):

        # attributes
        self.fvalues = None
        self.pvalues = None
    
    def fit(self, dataset: Dataset) -> 'F_Classif':
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
        # Select features with p-value under threshold
        mask = self.p_values < 0.05
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)

if __name__ == '__main__':

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                [0, 1, 4, 3],
                                [0, 1, 1, 3]]),
                    y=np.array([0, 1, 0]),
                    features=["f1", "f2", "f3", "f4"],
                    label="y")

    selector = F_Classif()
    selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
