import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./TPC5/src')
from apriori import Apriori
from apriori import TransactionDataset

class TestApriori(unittest.TestCase):
    
    def test_init(self):
        transaction_dataset = TransactionDataset(transactions=[['A', 'B'], ['B', 'C'], ['A', 'C']])
        min_support = 0.5
        min_confidence = 0.6
        apriori = Apriori(transaction_dataset, min_support, min_confidence)
        self.assertEqual(apriori.transaction_dataset, transaction_dataset)
        self.assertEqual(apriori.min_support, min_support)
        self.assertEqual(apriori.min_confidence, min_confidence)
        self.assertEqual(apriori.itemsets, {})
        self.assertEqual(apriori.rules, {})
    
    
    def test_fit(self):
        transaction_dataset = TransactionDataset([
            ['A', 'B', 'C'],
            ['A', 'C', 'D'],
            ['B', 'C', 'E'],
            ['A', 'B', 'C', 'E']
        ])
        min_support = 0.5
        min_confidence = 0.6
        apriori = Apriori(transaction_dataset, min_support, min_confidence)
        apriori.fit()
        self.assertIsNotNone(apriori.itemsets)
        self.assertIsNotNone(apriori.rules)
        self.assertGreater(len(apriori.itemsets), 0)
        self.assertGreater(len(apriori.rules), 0)
    
    def test_generate_candidates(self):
        apriori = Apriori()
        itemsets = [{1}, {2}, {3}]
        k = 2
        candidates = apriori.generate_candidates(itemsets, k)
        expected_candidates = {(1, 2), (1, 3), (2, 3)}
        self.assertEqual(candidates, expected_candidates)
    
    

if __name__ == '__main__':
    unittest.main()