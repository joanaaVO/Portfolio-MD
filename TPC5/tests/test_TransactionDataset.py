import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./TPC5/src')
from apriori import Apriori
from apriori import TransactionDataset

class TestTransactionDataset(unittest.TestCase):
    
    def test_init(self):
        dataset = TransactionDataset(transactions=[['A', 'B'], ['B', 'C'], ['A', 'C']])
        self.assertEqual(dataset.transactions, [['A', 'B'], ['B', 'C'], ['A', 'C']])
    
    
    def test_build_frequent_items(self):
        dataset = TransactionDataset(transactions=[['A', 'B'], ['B', 'C'], ['A', 'C']])
        dataset.build_frequent_items()
        expected_frequent_items = {'A': 0.6666666666666666, 'B': 0.6666666666666666, 'C': 0.6666666666666666}
        self.assertDictEqual(dataset.frequent_items, expected_frequent_items)
            
        
if __name__ == '__main__':
    unittest.main() 