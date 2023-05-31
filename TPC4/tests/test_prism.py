import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('./datasets')
sys.path.append('./TPC1/src')
sys.path.append('./TPC4/src')
from dataset import Dataset
from prism import Prism

class PrismTest(unittest.TestCase):
    
    def test_init(self):
        pass
    
    def test_fit(self):
        pass
    
    def test_majority_class(self):
        data = [
            {"attribute1": 1, "attribute2": "A", "target": "class1"},
            {"attribute1": 2, "attribute2": "B", "target": "class2"},
            {"attribute1": 3, "attribute2": "A", "target": "class1"},
            {"attribute1": 4, "attribute2": "B", "target": "class1"},
        ]
        target = "target"

        prism = Prism(data=data, target=target)
        majority_class = prism.majority_class(data)

        self.assertEqual(majority_class, "class1")

    
    def test_evaluate_rule_when_covered_data(self):
        data = [{'attribute': 'value', 'class': 'A'}, {'attribute': 'value', 'class': 'B'}]
        prism = Prism(data=data, target="class")
        rule = {"attribute": "value"}
        accuracy = prism.evaluate_rule(rule)
        self.assertEqual(accuracy, 0.5)
    
    def test_predict(self):
        data = [{'attribute': 'value1', 'class': 'A'}, {'attribute': 'value2', 'class': 'B'}]
        prism = Prism(data=data, target="class")
        predicted_class_1 = prism.predict({'attribute': 'value1'})
        predicted_class_2 = prism.predict({'attribute': 'value1', 'class': 'A'})
        predicted_class_3 = prism.predict({'attribute': 'value2', 'class': 'B'})
        self.assertEqual(predicted_class_1, "A")
        self.assertEqual(predicted_class_2, "A")
        self.assertEqual(predicted_class_3, prism.default_class)


    def test_repr(self):
        data = [{'attribute': 'value1', 'class': 'A'},{'attribute': 'value2', 'class': 'B'},{'attribute': 'value3', 'class': 'A'}]
        prism = Prism(data=data, target="class")
        prism.rules = [
            {"attribute": "value1", "class": "A"},
            {"attribute": "value2", "class": "B"},
            {"attribute": "value3", "class": "A"}
        ]
        expected_repr = "Rule 1: attribute=value1, class=A, predict=A\n" \
                   "Rule 2: attribute=value2, class=B, predict=B\n" \
                   "Rule 3: attribute=value3, class=A, predict=A"
        
        self.assertEqual(repr(prism), expected_repr)

        
        
        

    

if __name__ == '__main__':
    unittest.main()