import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./datasets')
sys.path.append('./TPC1/src')
from dataset import Dataset


class DatasetTestCase(unittest.TestCase):
    
    def test_init_raises_ValueError_when_X_is_None(self):
        self.assertRaises(ValueError, Dataset, None, np.ndarray([1, 2, 3]))


    def test_init_does_not_raise_ValueError_when_X_is_not_None(self):
        try:
            Dataset(np.ndarray([0, 1]),np.ndarray([1, 2, 3]))
        except ValueError:
            self.fail("Dataset() raised ValueError unexpectedly!")
        
        
    def test_init_create_features_when_features_are_None(self):
        dataset = Dataset(np.ndarray([0, 1]),np.ndarray([1, 2, 3]))
        self.assertEqual(dataset.features, ["0"])
    
    
    def test_init_create_features_when_features_are_not_None(self):
        dataset = Dataset(np.ndarray([0, 1]),np.ndarray([1, 2, 3]), ["a", "b"])
        self.assertEqual(dataset.features, ["a", "b"])
    
    
    def test_init_create_label_when_y_is_not_None_and_label_are_None(self):
        dataset = Dataset(np.ndarray([0, 1]),np.ndarray([1, 2, 3]))
        self.assertEqual(dataset.label, "y")


    def test_init_assign_all_variables_as_expected(self):
        y = np.ndarray([1, 2, 3])
        dataset = Dataset(np.ndarray([0, 1]), y, ["a", "b"], "c")
        np.testing.assert_array_equal(dataset.X, np.ndarray([0, 1]))
        np.testing.assert_array_equal(dataset.y, y)
        self.assertEqual(dataset.features, ["a", "b"])
        self.assertEqual(dataset.label, "c")
        self.assertEqual(dataset.feature_types, ["numerical"])
        self.assertEqual(dataset.categories, {})
        self.assertEqual(dataset.classes_dict, None)
     
        
    def test_read_raises_ValueError_when_read_is_called_with_invalid_file_path(self):
        self.assertRaises(ValueError, Dataset.read, "invalid_file_path")
    
    
    def test_read_create_y_when_label_is_None(self):
        dataset = Dataset.read("./datasets/iris.csv", None)
        self.assertEqual(dataset.y, None)
    
    
    def test_read_create_y_when_label_is_not_None(self):
        dataset = Dataset.read("./datasets/iris.csv", "class")
        y = [0, 1, 2]
        np.testing.assert_array_equal(np.unique(dataset.y), y)
        
    """
    def test_write_create_data_when_y_is_not_None(self):
        y = np.ndarray([1, 2])
        x = np.ndarray([0, 1])
        dataset = Dataset(x, y)
        dataset_write = Dataset.write(dataset, "./datasets/test.csv")
        dataset_read = Dataset.read("./datasets/test.csv", "class")
        np.testing.assert_array_equal(dataset_read.X, np.ndarray([0, 1]))
        self.assertEqual(dataset_read.y, np.ndarray([1, 2]))
    
    def test_write_create_data_when_y_is_None(self):
        x = np.ndarray([0, 1])
        dataset = Dataset(x, None)
        dataset_write = Dataset.write(dataset, "./datasets/test.csv")
        dataset_read = Dataset.read("./datasets/test.csv", None)
        np.testing.assert_array_equal(dataset_read.X, np.ndarray([0, 1]))
        self.assertEqual(dataset_read.y, None)
        """
    
        
if __name__ == '__main__':
    unittest.main()