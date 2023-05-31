import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./datasets')
sys.path.append('./TPC1/src')
sys.path.append('./TPC2/src')
from dataset import Dataset
from f_classif import F_Classif
from selectKBest import SelectKBest
from f_regression import F_Regression

class TestCaseSelectKBest(unittest.TestCase):
    
    def test_init_raises_ValueError_when_k_is_negative(self):
        self.assertRaises(ValueError, SelectKBest, None, -1)
    
    
    def test_init_does_not_raises_ValueError_when_k_is_positive(self):
        try:
            SelectKBest(None, 1)
        except ValueError:
            self.fail("SelectKBest() raised ValueError unexpectedly!")
    
    
    def test_fit_with_f_regression_returns_SelectKBest(self):
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                      [0, 1, 4, 3],
                                      [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
        f_regression = F_Regression()
        selectKBest = SelectKBest(score_func=f_regression, k=2)
        select = selectKBest.fit(dataset)
        self.assertTrue(np.allclose(select.scores, [0, 0.333333, 16.333333, 0]))
        self.assertTrue(np.allclose(select.pvalues, [1, 0.66666667, 0.15442096, 1]))
    
    
    def test_transform_returns_dataset(self):
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
        f_regression = F_Regression()
        selector = SelectKBest(score_func=f_regression, k=2)
        selector = selector.fit(dataset)
        dataset = selector.transform(dataset)
        np.testing.assert_array_equal(sorted(dataset.features), sorted(["f2", "f3"]))
        np.testing.assert_allclose(dataset.X, np.array([[0, 2], [4, 1], [1, 1]]))
        
        
        
if __name__ == '__main__':
    unittest.main()
