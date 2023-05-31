import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./datasets')
sys.path.append('./TPC1/src')
sys.path.append('./TPC4/src')
from dataset import Dataset
from naiveBayes import NaiveBayes

class NaiveBayesTest(unittest.TestCase):

    def test_init(self):
        nb = NaiveBayes()
        self.assertIsNone(nb.classes)
        self.assertIsNone(nb.mean)
        self.assertIsNone(nb.var)
        self.assertIsNone(nb.priors)
    
    
    def test_fit(self):
                
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        nb = NaiveBayes()
        nb.fit(X, y)
        self.assertIsNotNone(nb.mean)
        self.assertIsNotNone(nb.var)
        self.assertIsNotNone(nb.priors)
        np.testing.assert_array_equal(nb.classes, np.array([0, 1]))
        self.assertEqual(nb.mean.shape, (2, 2))  
        self.assertEqual(nb.var.shape, (2, 2))   
        self.assertEqual(nb.priors.shape, (2,))
        
        
    def test_predict_returns_predictions(self):
        
        X_test = np.array([[2, 3], [4, 5], [6, 7]])
        expected_predictions = np.array([0, 1, 1])
        nb = NaiveBayes()
        nb.classes = np.array([0, 1])
        nb.mean = np.array([[1, 2], [3, 4]])
        nb.var = np.array([[0.5, 0.5], [0.5, 0.5]])
        nb.priors = np.array([0.5, 0.5])
        predictions = nb.predict(X_test)
        np.testing.assert_array_equal(predictions, expected_predictions)


    def test_score(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        y_test = np.array([0, 1, 0])
        nb = NaiveBayes()
        nb.fit(X_train, y_train)
        self.assertEqual(nb.score(X_test, y_test), 1.0)
        
        
if __name__ == '__main__':
    unittest.main()