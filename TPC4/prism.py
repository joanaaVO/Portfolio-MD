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


if __name__ == '__main__':

    # Load the dataset using the Dataset class
    label = 'class'
    dataset = Dataset.read(file_path="./datasets/iris.csv", label=label)

    # Convert data to a dictionary
    data = [{attribute: value for attribute, value in zip(dataset.features, row)}
        for row in dataset.X]

    # Add target column to dictionary
    for i in range(len(data)):
        data[i][label] = dataset.y[i]

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=2023)

    # Create a PRISM object and fit it on the dataset
    prism = Prism(train_data, label, dataset.features)
    prism.fit()

    # Use the classifier to predict the classes of the test data and calculate the accuracy of the predictions
    correct_predictions = sum(test_data[i][label] == prism.predict(test_data[i]) for i in range(len(test_data)))
    accuracy = correct_predictions / len(test_data)
    print(f'Test set accuracy: {accuracy}')

