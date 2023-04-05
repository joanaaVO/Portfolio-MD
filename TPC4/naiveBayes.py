import numpy as np
import sys

sys.path.append('./TPC1')

from dataset import Dataset
from sklearn.model_selection import cross_val_score, train_test_split

class NaiveBayes:
    def __init__(self, classes = None, mean = None, var = None, priors = None):
        self.classes = classes
        self.mean = mean
        self.var = var
        self.priors = priors

    def fit(self, X, y):
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
        # Probability density function of a normal distribution
        # It assumes that the features are independent
        numerator = np.exp(-((X-mean) ** 2) / (2 * (var + 1e-4)))
        denominator = np.sqrt(2 * np.pi * (var + 1e-4))
        return numerator / denominator

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        return {"classes": self.classes, "mean": self.mean, "var": self.var, "priors": self.priors}

if __name__ == '__main__':
    # Load the dataset using the Dataset class
    data = Dataset.read(file_path="./datasets/iris.csv", label="class")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=2023)

    # Create the NaiveBayes model
    model = NaiveBayes()

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, data.X, data.y, cv=5)
    
    # Print the mean and standard deviation of the cross-validation scores
    print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")


