from sklearn.model_selection import train_test_split

class Prism:
    """
    A class that implements the PRISM algorithm for inducing decision rules from data.

    """

    def __init__(self, data=None, target=None, attributes=None):
        """
        Initialize a Prism instance.

        Parameters:
            data (array-like, shape (n_samples, n_features)): The data to induce rules from.
            target (string): The name of the column containing the target variable.
            attributes (list of strings): The names of the columns containing the attributes.

        """
        self.data = data
        self.target = target
        self.attributes = attributes
        self.rules = []
        self.default_class = self.majority_class(self.data)

    def fit(self):
        """
        Induce the decision rules from the data.

        """
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
        """
        Evaluate the accuracy of a given rule on the data.

        Parameters:
            rule (dictionary): A rule, represented as a dictionary mapping attribute names to values.

        Returns:
            (float): The accuracy of the rule on the data.

        """
        covered_data = [d for d in self.data if self.rule_covers(rule, d)]
        if not covered_data:
            return 0.0
        
        correct_predictions = sum(d[self.target] == self.predict(d) for d in covered_data)
        return correct_predictions / len(covered_data)

    def rule_covers(self, rule, data):
        """
        Check if a given rule covers a given data point.

        Parameters:
            rule (dictionary): A rule, represented as a dictionary mapping attribute names to values.
            data (array-like, shape (n_features,)): The data point to check.

        Returns:
            (bool): Whether the rule covers the data point.

        """
        return all(data[attr] == value for attr, value in rule.items())

    def majority_class(self, data):
        """
        Compute the most frequent class in the data.

        Parameters:
            data (array-like, shape (n_samples, n_features)): The data to compute the most frequent class from.

        Returns:
            (string): The most frequent class in the data.

        """
        classes = [d[self.target] for d in data]
        return max(set(classes), key=classes.count)

    def predict(self, data):
        """
        Predict the class of a given data point using the induced rules.

        Parameters:
            data (array-like, shape (n_features,)): The data point to predict the class for.

        Returns:
            (string): The predicted class for the data point.

        """
        for rule in self.rules:
            if self.rule_covers(rule, data):
                return rule[self.target]
        return self.default_class

    def __repr__(self):
        """
        Return a string representation of the induced rules.

        Returns:
            rule_strings (string): The string representation of the induced rules.

        """
        rule_strings = []
        for i, rule in enumerate(self.rules):
            rule_string = f"Rule {i+1}: "
            for attribute, value in rule.items():
                rule_string += f"{attribute}={value}, "
            rule_string += f"predict={rule[self.target]}"
            rule_strings.append(rule_string)
        return "\n".join(rule_strings)