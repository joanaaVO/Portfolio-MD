import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, attribute=None, threshold=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None


class DecisionTree:
    
    def __init__(self, attribute_selection='entropy', pre_pruning=None):
        """
        Initialize a DecisionTree object with the specified parameters.
        """
        self.attribute_selection = attribute_selection
        self.pre_pruning = pre_pruning
        self.root = None


    def fit(self, X, y):
        """
        Build the decision tree using the provided training data.

        Parameters:
        - X (array): The feature matrix of shape (n_samples, n_features).
        - y (array): The target labels of shape (n_samples,).

        """
        self.root = self.build_tree(X, y)
   
        
    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree using the provided data.

        Parameters:
        - X (array): The feature matrix of shape (n_samples, n_features).
        - y (array): The target labels of shape (n_samples,).
        - depth (int): The current depth of the tree (used for pre-pruning).

        Returns:
        - Node: The root node of the built decision tree.
        """
        # Check stopping criteria for pre-pruning
        if self.pre_pruning == 'size' and len(X) < 2:
            return Node(label=self.majority_vote(y))
        if self.pre_pruning == 'depth' and depth >= 3:
            return Node(label=self.majority_vote(y))

        # Check if all samples have the same label
        if len(np.unique(y)) == 1:
            return Node(label=y[0])

        # Select the best attribute and threshold for splitting
        attribute, threshold = self.select_attribute(X, y)

        # Check if no further split is possible
        if attribute is None or threshold is None:
            return Node(label=self.majority_vote(y))

        node = Node(attribute, threshold)

        # Split the data based on the selected attribute and threshold
        left_mask = X[:, attribute] <= threshold
        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # Recursively build the left and right subtrees
        node.left = self.build_tree(left_X, left_y, depth + 1)
        node.right = self.build_tree(right_X, right_y, depth + 1)

        return node


    def predict(self, X):
        """
        Predict the labels for input samples.

        Parameters:
        - X (array): The feature matrix of shape (n_samples, n_features).

        Returns:
        - array: The predicted labels of shape (n_samples,).
        """
        predictions = []
        for sample in X:
            node = self.root
            while node.attribute is not None:
                if sample[node.attribute] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.label)
        return np.array(predictions)

    
    def select_attribute(self, X, y):
        """
        Select the best attribute and threshold for splitting the data based on the specified attribute selection method.

        Parameters:
        - X (array): The feature matrix of shape (n_samples, n_features).
        - y (array): The target labels of shape (n_samples,).

        Returns:
        - tuple: The selected attribute and threshold for splitting.
        """
        if self.attribute_selection == 'entropy':
            return self.select_attribute_entropy(X, y)
        elif self.attribute_selection == 'gini':
            return self.select_attribute_gini(X, y)
        elif self.attribute_selection == 'gain_ratio':
            return self.select_attribute_gain_ratio(X, y)
        else:
            raise ValueError('Invalid attribute selection method specified')


    def select_attribute_entropy(self, X, y):
        """
        Select the best attribute and threshold for splitting the data based on the entropy criterion.

        Parameters:
        - X (array): The feature matrix of shape (n_samples, n_features).
        - y (array): The target labels of shape (n_samples,).

        Returns:
        - tuple: The selected attribute and threshold for splitting.
        """
        num_features = X.shape[1]
        best_gain = -np.inf
        best_attribute = None
        best_threshold = None

        # Calculate the entropy of the entire dataset
        entropy = self.calculate_entropy(y)

        for attribute in range(num_features):
            # Sort the data based on the attribute values
            sorted_indices = np.argsort(X[:, attribute])
            sorted_X = X[sorted_indices]
            sorted_y = y[sorted_indices]

            # Iterate over possible split points
            for i in range(1, len(sorted_X)):
                if sorted_X[i, attribute] != sorted_X[i - 1, attribute]:
                    threshold = (sorted_X[i, attribute] + sorted_X[i - 1, attribute]) / 2

                    # Split the data into left and right subsets
                    left_mask = sorted_X[:, attribute] <= threshold
                    right_mask = ~left_mask

                    left_y = sorted_y[left_mask]
                    right_y = sorted_y[right_mask]

                    # Calculate the information gain based on the split
                    left_entropy = self.calculate_entropy(left_y)
                    right_entropy = self.calculate_entropy(right_y)

                    information_gain = entropy - (
                        len(left_y) / len(y) * left_entropy +
                        len(right_y) / len(y) * right_entropy
                    )

                    # Update the best split if the information gain is higher
                    if information_gain > best_gain:
                        best_gain = information_gain
                        best_attribute = attribute
                        best_threshold = threshold

        return best_attribute, best_threshold


    def calculate_entropy(self, labels):
        """
        Calculate the entropy of the given labels.

        Parameters:
        - labels (array): The target labels of shape (n_samples,).

        Returns:
        - float: The entropy value.
        """
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy


    def select_attribute_gini(self, X, y):
        """
        Select the best attribute and threshold for splitting the data based on the Gini index criterion.

        Parameters:
        - X (array): The feature matrix of shape (n_samples, n_features).
        - y (array): The target labels of shape (n_samples,).

        Returns:
        - tuple: The selected attribute and threshold for splitting.
        """
        num_features = X.shape[1]
        best_gain = -np.inf
        best_attribute = None
        best_threshold = None

        # Calculate the Gini index of the entire dataset
        gini_index = self.calculate_gini_index(y)

        for attribute in range(num_features):
            # Sort the data based on the attribute values
            sorted_indices = np.argsort(X[:, attribute])
            sorted_X = X[sorted_indices]
            sorted_y = y[sorted_indices]

            # Check if all attribute values are the same
            if np.all(sorted_X[:, attribute] == sorted_X[0, attribute]):
                # Return the attribute with any threshold
                return attribute, sorted_X[0, attribute] + 1

            # Iterate over possible split points
            for i in range(1, len(X)):
                if sorted_X[i, attribute] != sorted_X[i - 1, attribute]:
                    threshold = (sorted_X[i, attribute] + sorted_X[i - 1, attribute]) / 2

                    # Split the data into left and right subsets
                    left_mask = sorted_X[:, attribute] <= threshold
                    right_mask = ~left_mask

                    left_y = sorted_y[left_mask]
                    right_y = sorted_y[right_mask]

                    # Calculate the Gini index based on the split
                    gini_split = (
                        len(left_y) / len(y) * self.calculate_gini_index(left_y) +
                        len(right_y) / len(y) * self.calculate_gini_index(right_y)
                    )

                    # Calculate the Gini gain
                    gini_gain = gini_index - gini_split

                    # Update the best split if the Gini gain is higher
                    if gini_gain > best_gain:
                        best_gain = gini_gain
                        best_attribute = attribute
                        best_threshold = threshold

        
        return best_attribute, best_threshold


    def calculate_gini_index(self, labels):
        """
        Calculate the Gini index of the given labels.

        Parameters:
        - labels (array): The target labels of shape (n_samples,).

        Returns:
        - float: The Gini index value.
        """
        unique, counts = np.unique(labels, return_counts=True)
        n_instances = float(sum(counts))
        gini = 0.0
        for count in counts:
            proportion = count / n_instances
            gini += proportion * (1.0 - proportion)
        return gini
    

    def select_attribute_gain_ratio(self, X, y):
        """
        Select the best attribute and threshold for splitting the data based on the gain ratio criterion.

        Parameters:
        - X (array): The feature matrix of shape (n_samples, n_features).
        - y (array): The target labels of shape (n_samples,).

        Returns:
        - tuple: The selected attribute and threshold for splitting.
        """
        num_features = X.shape[1]
        best_gain_ratio = -np.inf
        best_attribute = None
        best_threshold = None

        # Calculate the entropy of the entire dataset
        entropy = self.calculate_entropy(y)

        for attribute in range(num_features):
            # Sort the data based on the attribute values
            sorted_indices = np.argsort(X[:, attribute])
            sorted_X = X[sorted_indices]
            sorted_y = y[sorted_indices]

            # Iterate over possible split points
            for i in range(1, len(X)):
                if sorted_X[i, attribute] != sorted_X[i - 1, attribute]:
                    threshold = (sorted_X[i, attribute] + sorted_X[i - 1, attribute]) / 2

                    # Split the data into left and right subsets
                    left_mask = sorted_X[:, attribute] <= threshold
                    right_mask = ~left_mask

                    left_y = sorted_y[left_mask]
                    right_y = sorted_y[right_mask]

                    # Calculate the information gain based on the split
                    gain, _ = self.gain_ratio(X[:, attribute], y)

                    # Calculate the gain ratio
                    gain_ratio = gain / entropy if entropy != 0 else 0

                    # Update the best split if the gain ratio is higher
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_attribute = attribute
                        best_threshold = threshold

        return best_attribute, best_threshold


    def gain_ratio(self, feature, labels):
        """
        Calculate the gain ratio of the given labels.

        Parameters:
        - feature (array): The feature vector of shape (n_samples,).
        - labels (array): The target labels of shape (n_samples,).

        Returns:
        - float: The gain ratio value.
        """
        n = len(labels)
        values, counts = np.unique(feature, return_counts=True)
        H = self.calculate_entropy(labels)
        IV = - np.sum((counts / n) * np.log2(counts / n))
        IG = H
        for value, count in zip(values, counts):
            subset_labels = labels[feature == value]
            IG -= (count / n) * self.calculate_entropy(subset_labels)

        gain_ratio = IG / IV if IV != 0 else 0

        return gain_ratio, None



    def majority_vote(self, labels, default_label=0):
        """
        Determine the majority label from the given labels.

        Parameters:
        - labels (array): The target labels of shape (n_samples,).
        - default_label: The default label to return if the unique_labels array is empty (default: 0).

        Returns:
        - _type_: The majority label.
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) == 0:
            return default_label
        majority_label = unique_labels[np.argmax(counts)]
        return majority_label


    def __repr__(self):
        """
        Return a string representation of the decision tree.

        Returns:
        - str: The string representation of the decision tree.
        """
        return self.print_tree(self.root)

    def print_tree(self, node, indent=''):
        """
        Recursively generate a string representation of the decision tree.

        Parameters:
        - node (Node): The current node being printed.
        - indent (str): The indentation string for formatting.

        Returns:
        - str: The string representation of the decision tree.
        """
        if node.label is not None:
            return str(node.label)
        else:
            attribute = f'X[{node.attribute}]'
            left_subtree = self.print_tree(node.left, indent + '  | ')
            right_subtree = self.print_tree(node.right, indent + '  | ')
            return f'{attribute} <= {node.threshold}\n{indent}├─ True: {left_subtree}\n{indent}└─ False: {right_subtree}'
