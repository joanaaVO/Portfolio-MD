import numpy as np 
import sys

sys.path.append('../TPC1')

from dataset import Dataset 
   
class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTrees:
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, criterion='gini'):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.tree = None
    
    #Building tree 
    def building_tree(self, X, y, depth=0):
        pass

    #Calculates the entropy
    def entropy(self, label):
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / len(label)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    #Calculates the gini index
    def gini_index(self, label):
        counts = np.unique(label, return_counts=True)[1]
        proportions = counts / len(label)
        gini = 1 - np.sum(proportions ** 2)
        return gini

    #Calculates the Gain Ratio
    def gain_ratio(feature, labels):
        n = len(labels)
        values, counts = np.unique(feature, return_counts=True)
        H = entropy(labels)
        IV = - np.sum((counts / n) * np.log2(counts / n))
        IG = H
        for value, count in zip(values, counts):
            subset_labels = labels[feature == value]
            IG -= (count / n) * entropy(subset_labels)
        return IG / IV if IV != 0 else 0

    def predict():
        pass

#if __name__ == '__main__':

#dataset = Dataset.read(file_path="./datasets/iris.csv", label="class")
dt = DecisionTrees()
data = Dataset.read("../datasets/play_tennis.csv", label="play")
a = DecisionTrees.gini_index(data,data.y)
print(a)