import numpy as np
from scipy.stats import f
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('./TPC1')

from src.dataset import Dataset

class F_Regression:

    def __init__(self, alpha: float = 0.05):
        """
        Initializes the F_Regression object.

        Args:
        - alpha: significance level for hypothesis testing (default: 0.05)
        """
        
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        # parameters
        self.alpha = alpha

        # attributes
        self.fvalues = None
        self.pvalues = None

    def fit(self, dataset: Dataset) -> 'F_Regression':
        """
        Fit the F-test model to the dataset.

        Args:
        - dataset: Dataset object containing the input features and labels.

        Returns:
        - self: F_Regression object.
        """
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
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the input dataset by selecting features based on p-values.

        Args:
        - dataset: Dataset object containing the input features and labels.

        Returns:
        - Transformed Dataset object with selected features.
        """
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