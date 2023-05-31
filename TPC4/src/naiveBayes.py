import numpy as np
import sys

sys.path.append('./TPC1/src')

from dataset import Dataset
from sklearn.model_selection import cross_val_score, train_test_split

class NaiveBayes:
    def __init__(self, classes = None, mean = None, var = None, priors = None):
        """
        Initializes a new instance of the NaiveBayes class.

        Parameters:
            classes (array-like): The classes in the dataset.
            mean (ndarray): The mean of each feature for each class.
            var (ndarray): The variance of each feature for each class.
            priors (ndarray): The a priori probabilities of each class.

        """
        self.classes = classes
        self.mean = mean
        self.var = var
        self.priors = priors

    def fit(self, X, y):
        """
        Fits the NaiveBayes classifier to the given training data.

        Args:
            X (array-like, shape=(n_samples, n_features)): The training samples.
            y (array-like, shape=(n_samples,)): The target values.

        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        # Initialize the parameters of the model
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        # Calculate mean, variation and probabilities a priori for each class
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0)
            self.priors[i] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predicts the target values for the given test data.

        Args:
            X (array-like, shape=(n_samples, n_features)): The test samples.

        Returns:
            predictions (ndarray, shape=(n_samples,)): The predicted target values.

        """
        # Calculates the probability for each class for each instance
        posteriors = []
        for i, _ in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self._pdf(X, self.mean[i, :], self.var[i, :])), axis=1)
            posterior = prior + likelihood
            posteriors.append(posterior)

        # Selects the class with higher probability
        predictions = np.argmax(posteriors, axis=0)
        return predictions

    def _pdf(self, X, mean, var):
        """
        Calculates the probability density function of a normal distribution.

        Args:
            X (array-like, shape=(n_samples, n_features)): The input data.
            mean (ndarray, shape=(n_features,)): The mean of the distribution.
            var (ndarray, shape=(n_features,)): The variance of the distribution.

        Returns:
            The probability density function of a normal distribution.

        """
        # Probability density function of a normal distribution
        # It assumes that the features are independent
        numerator = np.exp(-((X-mean) ** 2) / (2 * (var + 1e-4)))
        denominator = np.sqrt(2 * np.pi * (var + 1e-4))
        return numerator / denominator

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like, shape=(n_samples, n_features)): The test samples.
            y (array-like, shape=(n_samples,)): The true labels.

        Returns:
            The mean accuracy of the NaiveBayes classifier on the given test data and labels.

        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        """
        Returns the parameters of the NaiveBayes classifier.

        Args:
            deep (boolean): Whether to return deep copies of the parameters.

        Returns:
            A dictionary containing the parameters of the NaiveBayes classifier.
        """
        return {"classes": self.classes, "mean": self.mean, "var": self.var, "priors": self.priors}