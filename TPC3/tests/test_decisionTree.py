import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./datasets')
sys.path.append('./TPC1/src')
sys.path.append('./TPC3/src')
from dataset import Dataset
from decisionTree import DecisionTree

class DecisionTree2Test(unittest.TestCase):
    
    def test_init(self):
        dt = DecisionTree()
        self.assertEqual(dt.attribute_selection, 'entropy')
        self.assertIsNone(dt.pre_pruning)
        self.assertIsNone(dt.root)


    def test_entropy(self):
        dt = DecisionTree()
        labels = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        expected_entropy = -np.sum(probabilities * np.log2(probabilities))
        calculated_entropy = dt.calculate_entropy(labels)
        self.assertAlmostEqual(calculated_entropy, expected_entropy)
        
        
    def test_gini_index(self):
        dt = DecisionTree()
        labels = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_instances = float(sum(counts))
        gini = 0.0
        for count in counts:
            proportion = count / n_instances
            gini += proportion * (1.0 - proportion)
        calculated_gini_index = dt.calculate_gini_index(labels)
        self.assertAlmostEqual(calculated_gini_index, gini)
    
    
    def test_gain_ratio(self):
        dt = DecisionTree()
        feature = np.array([1, 2, 3, 1, 2, 3])
        labels = np.array([0, 1, 0, 1, 1, 0])

        expected_gain_ratio = dt.gain_ratio(feature, labels)[0]

        calculated_gain_ratio = dt.gain_ratio(feature, labels)[0]
        self.assertAlmostEqual(calculated_gain_ratio, expected_gain_ratio)
        
        
        
    
if __name__ == '__main__':
    unittest.main()