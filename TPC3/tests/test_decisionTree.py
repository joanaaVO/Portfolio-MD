import unittest
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

    
    def test_build_tree_when_pre_pruning_is_size(self):
        dt = DecisionTree(pre_pruning='size')
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        root = dt.build_tree(X, y)
        expected_label = dt.majority_vote(y, default_label=None)
        self.assertIsNone(root.label)
        self.assertEqual(root.left.label, 0)
        self.assertEqual(root.right.label, 1) 


    def test_build_tree_when_pre_pruning_is_depth(self):
        data = load_iris()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dt = DecisionTree(pre_pruning='depth')
        dt.fit(X_train, y_train)
        root = dt.build_tree(X, y)
        self.assertIsNone(root.label)
        self.assertIsNotNone(root.left) 
        self.assertIsNotNone(root.right)
    
    
    def test_build_tree_when_len_of_y_is_1(self):
        dt = DecisionTree(pre_pruning='size')
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 1, 1, 1])
        root = dt.build_tree(X, y)
        self.assertEqual(root.label, 1)
        
        
    def test_build_tree_when_len_of_y_is_0(self):
        dt = DecisionTree(pre_pruning='size')
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 0, 0])
        root = dt.build_tree(X, y)
        self.assertEqual(root.label, 0)


    def test_predict(self):
        dt = DecisionTree(pre_pruning='size')
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([0, 0, 1, 1])
        dt.fit(X_train, y_train)
        X_test = np.array([[1.5], [3.5]])
        expected_predictions = np.array([0, 1])
        predictions = dt.predict(X_test)
        np.testing.assert_array_equal(predictions, expected_predictions)
    
    
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
        
    
    def test_repr(self):
        data = load_iris()
        X = data.data
        y = data.target

        # Divida os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        dt_e = DecisionTree(attribute_selection='entropy', pre_pruning='size')
        dt_e.fit(X_train, y_train)
        print(dt_e)
        
        dt_ed = DecisionTree(attribute_selection='entropy', pre_pruning='depth')
        dt_ed.fit(X_train, y_train)
        print(dt_ed)
        
        
    
        
        
    
if __name__ == '__main__':
    unittest.main()