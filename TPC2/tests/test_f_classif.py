import unittest
import numpy as np
import sys
sys.path.append('./TPC1/src')
sys.path.append('./TPC2/src')
from dataset import Dataset
from f_classif import F_Classif



class TestCaseF_Classif(unittest.TestCase):
    
    def test_init(self):
        f_classif = F_Classif()
        self.assertIsNone(f_classif.fvalues)
        self.assertIsNone(f_classif.pvalues)
    
    
    def test_fit_returns_F_Classif(self):
        X = np.array([[1, 2, 1, 3], [1, 1, 4, 3], [1, 1, 1, 3]])
        y = np.array([0, 1, 0])
        features = ["f1", "f2", "f3", "f4"]
        label = "y"
        dataset = Dataset(X, y, features, label)
        f_classif = F_Classif()
        self.assertIsInstance(f_classif.fit(dataset), F_Classif)
    
    
    def test_transform_returns_Dataset_object(self):
        X = np.array([[1, 2, 1, 3], [1, 1, 4, 3], [1, 1, 1, 3]])
        y = np.array([0, 1, 0])
        features = ["f1", "f2", "f3", "f4"]
        label = "y"
        dataset = Dataset(X, y, features, label)
        f_classif = F_Classif()
        f_classif.fit(dataset)
        transformed_dataset = f_classif.transform(dataset)
        self.assertIsInstance(transformed_dataset, Dataset)


if __name__ == '__main__':
    unittest.main()