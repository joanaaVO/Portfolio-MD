import numpy as np
import math
import sys

sys.path.append('./TPC1/src')
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
    
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=3, 
                criterion='gini', pre_pruning='max_depth', post_pruning='pessimistic', threshold=5):
        """
        Initialize a DecisionTrees object with the specified parameters.
        """
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
        self.threshold = threshold

    def fit(self, X, y, X_val, y_val, depth=0):
        """
        Fit the decision tree model to the training data.

        Args:
            X (array): Training feature dataset.
            y (array): Training target values.
            X_val (array): Validation feature dataset.
            y_val (array): Validation target values.
            depth (int): Current depth of the tree. (default: 0)
        """
        self.tree = self.build_tree(X, y, X_val, y_val, depth)
        if self.pre_pruning == 'size':
            self.size_pruning()
        elif self.pre_pruning == 'max_depth':
            self.tree = self.max_depth_pruning(self.tree, self.max_depth)
            self.update_internal_nodes()

        if self.post_pruning == 'pessimistic':
            self.pessimistic_pruning(X_val, y_val)
            
        elif self.post_pruning == 'optimistic':
            self.optimistic_pruning(X_val, y_val)

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


        # Check stopping criteria
        if depth == self.max_depth or len(X) < self.min_samples_split:
            # Create a leaf node
            leaf = LeafNode(self.most_common_class(y))  
            leaf.data = y
            return leaf if len(y) > 0 else None
        
        # Build the decision tree recursively
        best_attribute, best_threshold = self.choose_attribute(X, y)
        # Check if splitting the data will result in any improvement
        if best_attribute is None:
            # Create a leaf node
            leaf = LeafNode(self.most_common_class(y))
            leaf.data = y
            return leaf if len(y) > 0 else None

        else:
            node = InternalNode(best_attribute, best_threshold, None, None)
            node.data = y
            self.internal_nodes.append(node)

        # Split the data based on the best attribute and threshold
        X_left, y_left, X_right, y_right = self.split_data(X, y, X_val, y_val, best_attribute, best_threshold)

        # Recursively build the left and right subtrees
        if len(X_left) > 0 and len(X_right) > 0:
            node.left = self.build_tree(X_left, y_left, X_val, y_val, depth+1)
            node.right = self.build_tree(X_right, y_right, X_val, y_val, depth+1)
        else:
            # If one of the splits is empty, create a leaf node with the majority class of the parent node
            leaf = LeafNode(self.most_common_class(y))
            leaf.data = y
            self.leaf_nodes.append(leaf)
            if len(X_left) == 0:
                node.left = leaf
            else:
                node.right = leaf


         # Check if the resulting children have too few samples
        if len(X_left) < self.min_samples_split or len(X_right) < self.min_samples_split:
            # Create a leaf node
            leaf = LeafNode(self.most_common_class(y))
            leaf.data = y
            return leaf

        # Check if the internal nodes list is empty before calling independence_pruning
        if len(self.internal_nodes) > 0 and self.pre_pruning == 'independence':
            should_prune = self.independence_pruning(X_val, y_val)
            if should_prune:
                leaf = LeafNode(self.most_common_class(y))
                leaf.data = y
                self.leaf_nodes.append(leaf)
                return leaf

        # Set the root node and return the tree
        if depth == 0:
            self.root = node

        return node

    # Splits data based on feature and threshold values
    def split_data(self, X, y, X_val, y_val, feature, threshold):
        if threshold is None:
            raise ValueError("Invalid threshold value: None")
            
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        
        # Create left and right node based on split
        left_node = LeafNode(self.most_common_class(y[left_idx]))
        right_node = LeafNode(self.most_common_class(y[right_idx]))
            
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
            
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

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
        best_attribute = 0
        best_gain = 0
        best_threshold = 0
        for attribute in range(X.shape[1]):
            values = X[:, attribute]
            gain, threshold = self.gain_ratio(values, y)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                best_threshold = threshold
        return best_attribute, best_threshold

    # Calculates the entropy
    def entropy(self, label):
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / len(label)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    # Calculates the gini index
    def gini_index(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        n_instances = float(sum(counts))
        gini = 0.0
        for count in counts:
            proportion = count / n_instances
            gini += proportion * (1.0 - proportion)
        return gini

    # Calculates the Gain Ratio
    def gain_ratio(self, feature, labels):
        n = len(labels)
        values, counts = np.unique(feature, return_counts=True)
        if len(values) <= self.threshold:
            # Discrete feature
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

            return IG / IV if IV != 0 else 0, None
        else:
            # Continuous feature
            thresholds = np.unique(feature)
            max_gain_ratio = -np.inf
            best_threshold = None
            for threshold in thresholds:
                left_labels = labels[feature <= threshold]
                right_labels = labels[feature > threshold]
                if self.criterion == 'entropy':
                    H = self.entropy(labels)
                    left_H = self.entropy(left_labels)
                    right_H = self.entropy(right_labels)
                elif self.criterion == 'gini':
                    H = self.gini_index(labels)
                    left_H = self.gini_index(left_labels)
                    right_H = self.gini_index(right_labels)
                else:
                    raise ValueError('Invalid measure specified')

                left_ratio = len(left_labels) / n
                right_ratio = len(right_labels) / n
                gain = H - left_ratio * left_H - right_ratio * right_H
                IV = - (left_ratio * np.log2(left_ratio) + right_ratio * np.log2(right_ratio))
                gain_ratio = gain / IV if IV != 0 else 0
                if gain_ratio > max_gain_ratio:
                    max_gain_ratio = gain_ratio
                    best_threshold = threshold

            return max_gain_ratio, best_threshold
            
    def independence_pruning(self, X_val, y_val):
        if self.tree is None:
            raise Exception('Decision tree not trained.')

        # Check if the internal nodes list is empty
        if len(self.internal_nodes) == 0:
            return False
            
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

    #NOTE - VERIFICAR
    def prune_pessimistic(self, X_val, y_val):
        if self.post_pruning != 'pessimistic':
            return
        
        self._pessimistic_prune_helper(self.root, X_val, y_val)
        
    def _pessimistic_prune_helper(self, node, X_val, y_val):
        if node is None:
            return
        
        # Prune left and right subtrees
        self._pessimistic_prune_helper(node.left, X_val, y_val)
        self._pessimistic_prune_helper(node.right, X_val, y_val)
        
        # Check if node is internal
        if isinstance(node, InternalNode):
            # Calculate current error rate
            y_pred = node.predict(X_val)
            error_rate = 1 - accuracy_score(y_val, y_pred)
            
            # Calculate pessimistic error rate
            n = len(y_val)
            z = 1.96
            pessimistic_error = error_rate + z * math.sqrt((error_rate * (1 - error_rate) + z**2 / (4 * n)) / n)
            
            # Check if pessimistic error rate is lower than current error rate
            if pessimistic_error <= error_rate:
                # Convert node to leaf
                value = self.most_common_class(node.data)
                leaf = LeafNode(value)
                leaf.data = node.data
                node.left = None
                node.right = None
                self.leaf_nodes.append(leaf)
                
                # Remove node from internal nodes list
                self.internal_nodes.remove(node)


    def reduced_error_pruning(self):
        # Armazena o estado atual da árvore de decisão antes da poda
        self.backup_tree = np.copy.deepcopy(self.tree)
        
        # Calcula o erro na subárvore completa
        before_pruning_error = self.test(self.data)
        
        # Chama a função de poda
        self._prune(self.tree)
        
        # Calcula o erro após a poda
        after_pruning_error = self.test(self.data)
        
        # Se o erro for maior após a poda, reverta para o backup
        if after_pruning_error >= before_pruning_error:
            self.tree = self.backup_tree
        
    def _prune(self, node):
        # Se o nó atual for uma folha, retorne
        if node.leaf:
            return
        
        # Poda os filhos
        for child in node.children:
            self._prune(child)
        
        # Armazena o estado atual da árvore de decisão antes da poda
        backup_children = np.copy.deepcopy(node.children)
        
        # Remove todos os filhos do nó atual
        node.children = []
        
        # Verifica a acurácia da subárvore após a poda
        accuracy = self.test(self.data)
        
        # Se a acurácia melhorar, mantém a poda
        if accuracy > self.test(self.data, True):
            node.leaf = True
            node.children = None
            return
        else:
            node.children = backup_children

    def repr(node, indent=''):
        if node.left is None and node.right is None:
            return f"{indent}Predict: {node.label}"
        else:
            decision = f"If {node.feature} <= {node.threshold}:"
            left_tree = repr(node.left, indent + '  ')
            right_tree = repr(node.right, indent + '  ')
            return f"{indent}{decision}\n{left_tree}\n{indent}else:\n{right_tree}"