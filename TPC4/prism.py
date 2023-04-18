import sys

sys.path.append('./TPC1')

from dataset import Dataset
from sklearn.model_selection import train_test_split

class Prism:
    def __init__(self, data=None, target=None, attributes=None):
        self.data = data
        self.target = target
        self.attributes = attributes
        self.rules = []
        self.default_class = self.majority_class(self.data)

    def fit(self):
        while True:
            best_rule = None
            best_coverage = 0
            best_accuracy = 0
            
            for attribute in self.attributes:
                for value in set([d[attribute] for d in self.data]):
                    rule = {attribute: value}
                    coverage = sum(self.rule_covers(rule, d) for d in self.data)
                    
                    if coverage > best_coverage:
                        accuracy = self.evaluate_rule(rule)
                        
                        if accuracy > best_accuracy:
                            best_rule = rule
                            best_coverage = coverage
                            best_accuracy = accuracy
            
            if best_rule is None:
                break
                
            self.rules.append(best_rule)
            self.data = [d for d in self.data if not self.rule_covers(best_rule, d)]
            
            if not self.data:
                break

    def evaluate_rule(self, rule):
        covered_data = [d for d in self.data if self.rule_covers(rule, d)]
        if not covered_data:
            return 0.0
        
        correct_predictions = sum(d[self.target] == self.predict(d) for d in covered_data)
        return correct_predictions / len(covered_data)

    def rule_covers(self, rule, data):
        return all(data[attr] == value for attr, value in rule.items())

    def majority_class(self, data):
        classes = [d[self.target] for d in data]
        return max(set(classes), key=classes.count)

    def predict(self, data):
        for rule in self.rules:
            if self.rule_covers(rule, data):
                return rule[self.target]
        return self.default_class

