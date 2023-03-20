import numpy as np 
import sys

sys.path.append('./TPC1')

from dataset import Dataset 
from sklearn.metrics import accuracy_score
   
class Node:

    """Base class for nodes in a decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, data=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.children = []
        self.data = data

    def add_child(self, child):
        self.children.append(child)

class InternalNode(Node):

    """Node representing a split on a feature."""
    def __init__(self, feature, threshold, left, right):
        super().__init__(feature, threshold, left, right)

class LeafNode(Node):

    """Node representing a leaf in the decision tree."""
    def __init__(self, value):
        super().__init__()
        self.value = value

class DecisionTrees:
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, 
                criterion='gini', pre_pruning='size', post_pruning='pessimistic'):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.pre_pruning = pre_pruning
        self.post_pruning = post_pruning
        self.tree = None
        self.internal_nodes = []
        self.leaf_nodes = []
        self.root = None

    def predict(self, X):
        if self.tree is None:
            raise Exception("The tree has not been built yet!")
        
        return [self.traverse(x, self.tree) for x in X]

    def traverse(self, x, node):
        if isinstance(node, LeafNode):
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left)
        else:
            return self.traverse(x, node.right)
    
    # Building tree
    def build_tree(self, X, y, X_val, y_val, depth=0):

        # Call independence pruning
        should_prune = self.independence_pruning(X_val, y_val)

        if should_prune:
            value = self.most_common_class(y)
            leaf = LeafNode(value)
            leaf.data = y
            self.leaf_nodes.append(leaf)
            return leaf
            
        best_attribute, best_threshold = self.choose_attribute(X, y)
        node = InternalNode(best_attribute, best_threshold, None, None)
        node.data = y
        self.internal_nodes.append(node)
        
        X_left, y_left, X_right, y_right = self.split_data(X, y, X_val, y_val, best_attribute, best_threshold)
        
        if len(X_left) == 0 or len(X_right) == 0:
            value = self.most_common_class(y)
            leaf = LeafNode(value)
            leaf.data = y
            self.leaf_nodes.append(leaf)
            return leaf
        
        node.left = self.build_tree(X_left, y_left, X_val, y_val, depth+1)
        node.right = self.build_tree(X_right, y_right, X_val, y_val, depth+1)

        # Call pruning function
        if self.pre_pruning == 'size':
            self.size_pruning()

        elif self.pre_pruning == 'depth':
            self.max_depth_pruning(self.tree, self.max_depth)
        
        # TO-DO: Add the post pruning functions
        #node = self.post_pruning(node, X, y)

        if depth == 0:
            self.root = node
            
        return node

    # Splits data based on feature and threshold values
    def split_data(self, X, y, X_val, y_val, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        
        # Create left and right node based on split
        left_node = LeafNode(self.most_common_class(y[left_idx]))
        right_node = LeafNode(self.most_common_class(y[right_idx]))

        # Check if node meets pre-pruning conditions
        if self.pre_pruning is not None and self.pre_pruning(X, y, left_idx, right_idx):
            return None, None
        
        # Check if node is a leaf
        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0 or self.max_depth == 0 or \
            len(X) < self.min_samples_split or len(y) < self.min_samples_split:
            return left_node, right_node
        
        # Create internal node
        internal_node = InternalNode(feature, threshold, left_node, right_node)
        
        # Append to lists of nodes
        self.internal_nodes.append(internal_node)
        self.leaf_nodes.append(left_node)
        self.leaf_nodes.append(right_node)

        # Recursively split left and right nodes
        left_node, right_node = self.build_tree(X[left_idx], y[left_idx], X_val, y_val)
        internal_node.left = left_node
        
        left_node, right_node = self.build_tree(X[right_idx], y[right_idx], X_val, y_val)
        internal_node.right = right_node
        
        return internal_node, None

    # Returns the most commom class in 'y'
    def most_common_class(self, y):
        class_counts = {}
        for label in y:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        max_count = 0
        max_class = None
        for label, count in class_counts.items():
            if count > max_count:
                max_count = count
                max_class = label
        return max_class

    # Selects the best attribute to split the data based on the maximum information gain or gain ratio criterion
    def choose_attribute(self, X, y):
        best_attribute = None
        best_gain = 0
        for attribute in range(X.shape[1]):
            values = X[:, attribute]
            gain = self.gain_ratio(values, y)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
        return best_attribute

    # Calculates the entropy
    def entropy(self, label):
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / len(label)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    # Calculates the gini index
    def gini_index(self, label):
        counts = np.unique(label, return_counts=True)[1]
        proportions = counts / len(label)
        gini = 1 - np.sum(proportions ** 2)
        return gini

    # Calculates the Gain Ratio
    def gain_ratio(self, feature, labels):
        n = len(labels)
        values, counts = np.unique(feature, return_counts=True)
        if self.criterion == 'entropy':
            H = self.entropy(labels)
        elif self.criterion == 'gini':
            H = self.gini_index(labels)
        else:
            raise ValueError('Invalid measure specified')

        IV = - np.sum((counts / n) * np.log2(counts / n))
        IG = H
        for value, count in zip(values, counts):
            subset_labels = labels[feature == value]
            if self.criterion == 'entropy':
                IG -= (count / n) * self.entropy(subset_labels)
            elif self.criterion == 'gini':
                IG -= (count / n) * self.gini_index(subset_labels)
            else:
                raise ValueError('Invalid measure specified')

        return IG / IV if IV != 0 else 0
    def independence_pruning(self, X_val, y_val):
        if self.tree is None:
            raise Exception('Decision tree not trained.')
            
        # Calculates total accuracy of the indepent validation set
        current_acc = accuracy_score(y_val, self.predict(X_val))
            
        # Goes through every internal node of the tree
        for node in self.internal_nodes:
            # Saves temporarily the right and left children from the current node
            left_child = node.left
            right_child = node.right

            # Removes the children of the current node
            node.left = None
            node.right = None
                
            #Calculates accucary after removing the children from current node
            pruned_acc = accuracy_score(y_val, self.predict(X_val))

            # If accuracy doesn't get better, undoes the children removal
            if pruned_acc <= current_acc:
                node.left = left_child
                node.right = right_child
            else:
                return True
        
        return False

    def size_pruning(self):
        if self.tree is None:
            raise Exception('Decision tree not trained.')
            
        # Goes through every internal node of the tree
        for node in self.internal_nodes:

            # Calculates the number of samples in each child of the current node
            left_n_samples = len(node.left.data)
            right_n_samples = len(node.right.data)

            # If both children have less than min_samples_leaf, prunes the current node
            if left_n_samples < self.min_samples_leaf and right_n_samples < self.min_samples_leaf:
                node.left = None
                node.right = None

        # Updates the list of internal nodes of the tree
        self.update_internal_nodes()

    # Rebuilds the list of internal nodes based on the current state of the tree
    def update_internal_nodes(self, node=None):
        if node is None:
            node = self.tree
        
        if isinstance(node, InternalNode):
            self.internal_nodes.append(node)
            self.update_internal_nodes(node.left)
            self.update_internal_nodes(node.right)

    # Prunes tree nodes beyond a specified maximum depth
    def max_depth_pruning(self, tree, max_depth):
        if self.depth(tree.root) > max_depth:
            tree = None
        elif tree.left is not None:
            tree.left = self.max_depth_pruning(tree.left, max_depth - 1)
        elif tree.right is not None:
            tree.right = self.max_depth_pruning(tree.right, max_depth - 1)
        return tree

    # Returns the total depth of a tree
    def depth(self, node):
        if isinstance(node, LeafNode):
            return 0
        else:
            left_depth = self.depth(node.left)
            right_depth = self.depth(node.right)
            return max(left_depth, right_depth) + 1

    def pessimistic_pruning(self):
        pass

    def reduced_error_pruning(self):
        pass

#if __name__ == '__main__':

#dataset = Dataset.read(file_path="./datasets/iris.csv", label="class")
dt = DecisionTrees()
data = Dataset.read("./datasets/play_tennis.csv", label="play")
a = DecisionTrees.gini_index(data,data.y)
print(a)