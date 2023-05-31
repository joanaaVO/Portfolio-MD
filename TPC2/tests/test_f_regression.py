import unittest
import numpy as np
import sys
sys.path.append('./TPC1/src')
sys.path.append('./TPC2/src')
from dataset import Dataset
from f_regression import F_Regression

class TestCaseF_Regression(unittest.TestCase):
    
    def test_init_raises_ValueError_when_alpha_is_invalid(self):
        self.assertRaises(ValueError, F_Regression, alpha=1.5)
        self.assertRaises(ValueError, F_Regression, alpha=-1)
        
    def test_init_does_not_raise_ValueError_when_alpha_is_valid(self):
        try:
            F_Regression(alpha=0.05)
        except ValueError:
            self.fail("alpha must be between 0 and 1")

    def test_init_when_alpha_is_valid(self):
        f_regression = F_Regression()
        self.assertEqual(f_regression.alpha, 0.05)
        self.assertIsNone(f_regression.fvalues)
        self.assertIsNone(f_regression.pvalues)


    def test_fit_returns_self(self):
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                      [0, 1, 4, 3],
                                      [0, 1, 1, 3]]),
                          y=np.array([0, 1, 0]),
                          features=["f1", "f2", "f3", "f4"],
                          label="y")
        f_regression = F_Regression()
        self.assertEqual(f_regression.fit(dataset), f_regression)
    
    
    def test_transform_returns_dataset(self):   
        dataset = Dataset(X=np.array([[1, 2, 1, 3],
                                      [1, 1, 4, 3],
                                      [1, 1, 1, 3]]),
                          y=np.array([0, 1, 0]),
                          features=["f1", "f2", "f3", "f4"],
                          label="y")
        f_regression = F_Regression()
        f_regression.fit(dataset)
        f_regression_transform = f_regression.transform(dataset)
        np.testing.assert_array_equal(f_regression_transform.X, np.array([[1],[4],[1]]))
        self.assertEqual(f_regression_transform.features, ["f3"])

    
if __name__ == '__main__':
    unittest.main()