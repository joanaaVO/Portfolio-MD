import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./datasets')
sys.path.append('./TPC1/src')
sys.path.append('./TPC3/src')
from dataset import Dataset
from decisionTree import DecisionTrees

class DecisionTreeTest(unittest.TestCase):
    
    def test_init(self):
        dt = DecisionTrees()
        self.assertEqual(dt.max_depth, 3)
        self.assertEqual(dt.min_samples_split, 2)
        self.assertEqual(dt.min_samples_leaf, 1)
        self.assertEqual(dt.max_features, 3)
        self.assertEqual(dt.criterion, 'gini')
        self.assertEqual(dt.pre_pruning, 'max_depth')
        self.assertEqual(dt.post_pruning, 'pessimistic')
        self.assertIsNone(dt.tree)
        self.assertEqual(dt.internal_nodes, [])
        self.assertEqual(dt.leaf_nodes, [])
        self.assertIsNone(dt.root)
        self.assertEqual(dt.threshold, 5)
    
    def test_fit_when_pre_prunning_is_equal_to_size(self):
        
    
if __name__ == '__main__':
    unittest.main()