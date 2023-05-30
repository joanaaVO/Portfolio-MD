import unittest
import numpy as np
import sys
sys.path.append('./TPC1/src')
sys.path.append('./TPC2/src')
from dataset import Dataset
from f_regression import F_Regression
from variance_threshold import VarianceThreshold


class VarianceThresholdTest(unittest.TestCase):
    
    def test_init_raises_ValueError_when_threshold_is_less_than_zero(self):
        self.assertRaises(ValueError, VarianceThreshold, -1)
    
    
    def test_init_when_threshold_is_valid(self):
        selector = VarianceThreshold(1)
        self.assertEqual(selector.threshold, 1)
        self.assertIsNone(selector.variance)
    
    
    def test_fit_returns_self(self):
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                      [0, 1, 4, 3],
                                      [0, 1, 1, 3]]),
                          y=np.array([0, 1, 0]),
                          features=["f1", "f2", "f3", "f4"],
                          label="y")
        selector = VarianceThreshold()
        self.assertEqual(selector.fit(dataset), selector)
    
    
    def test_transform_returns_dataset(self):
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                      [0, 1, 4, 3],
                                      [0, 1, 1, 3]]),
                          y=np.array([0, 1, 0]),
                          features=["f1", "f2", "f3", "f4"],
                          label="y")
        selector = VarianceThreshold()
        selector.fit(dataset)
        selector_transform = selector.transform(dataset)
        self.assertEqual(selector_transform.features, ["f2", "f3"])
        np.testing.assert_array_equal(selector_transform.X, np.array([[2,0],[1,4],[1,1]]))

        
if __name__ == '__main__':
    unittest.main()