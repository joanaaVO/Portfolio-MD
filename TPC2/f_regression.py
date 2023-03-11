import numpy as np
from scipy.stats import f
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('./TPC1')

from dataset import Dataset

class FRegression:

    def __init__(self, alpha: float = 0.05):

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        # parameters
        self.alpha = alpha

        # attributes
        self.fvalues = None
        self.pvalues = None

    def fit(self, dataset: Dataset) -> 'FRegression':

        X = dataset.X
        y = dataset.y

        # Compute F and p-values
        n_features = X.shape[1]
        self.fvalues = np.zeros(n_features)
        self.pvalues = np.zeros(n_features)
        for i in range(n_features):
            X_i = X[:, i].reshape(-1, 1)
            # Fit linear regression model to predict y from X_i
            model_i = LinearRegression().fit(X_i, y)
            # Compute explained variance (SSE) and residual variance (SSTO - SSE)
            SSE = ((model_i.predict(X_i) - y)**2).sum()
            SSTO = ((y - y.mean())**2).sum()
            # Compute F-statistic and p-value
            df_reg = 1
            df_res = len(y) - 2
            self.fvalues[i] = (SSTO - SSE) / df_reg / (SSE / df_res)
            self.pvalues[i] = 1 - f.cdf(self.fvalues[i], df_reg, df_res)
            print('p-values:', self.pvalues)

        return self

    def transform(self, dataset: Dataset) -> Dataset:

        X = dataset.X

        # Select features with p-value < alpha
        selected = self.pvalues < self.alpha

        # Select columns of X and features
        X_selected = X[:, selected]
        features = np.array(dataset.features)[selected]

        # Create new Dataset object
        return Dataset(X=X_selected, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = FRegression()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
