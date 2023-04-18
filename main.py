import sys

sys.path.append('./TPC1')
sys.path.append('./TPC2')
sys.path.append('./TPC3')
sys.path.append('./TPC4')
sys.path.append('./TPC5')
sys.path.append('./datasets')

from TPC1.dataset import Dataset
from TPC2.f_classif import F_Classif
from TPC2.f_regression import F_Regression
from TPC2.selectKBest import SelectKBest
from TPC2.variance_threshold import VarianceThreshold
from TPC3.decisionTree import DecisionTrees
from TPC4.prism import Prism

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

def main(exercise):
    
    if exercise == "1":
        
        file_path = "./datasets/iris.csv"
        label = "class"
        
        dataset = Dataset.read(file_path = file_path, label= label)

        print("Describe:")
        print(dataset.describe())
        print("Median:")
        print(dataset.get_median())
        print("Classes:")
        print(dataset.get_classes())
        print("Max:")
        print(dataset.get_max())
        print("Min:")
        print(dataset.get_min())
        print("NULLS:")
        print(dataset.get_null_values())
        print("Shape")
        print(dataset.get_shape())

    
    if exercise == "2-f_classif":
        
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                [0, 1, 4, 3],
                                [0, 1, 1, 3]]),
                    y=np.array([0, 1, 0]),
                    features=["f1", "f2", "f3", "f4"],
                    label="y")

        selector = F_Classif()
        selector.fit(dataset)
        dataset = selector.transform(dataset)
        print(dataset.features)


    if exercise == "2-f_regression":
        
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

        selector = F_Regression()
        selector = selector.fit(dataset)
        dataset = selector.transform(dataset)
        print(dataset.features)


    if exercise == "2-selectKBest":
        
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

        #f_classif = F_Classif()
        #selector = SelectKBest(score_func=f_classif, k=2)
        f_regression = F_Regression()
        selector = SelectKBest(score_func=f_regression, k=2)
        selector = selector.fit(dataset)
        dataset = selector.transform(dataset)
        print(dataset.features)
        

    if exercise == "2-variance_threshold":
        
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

        selector = VarianceThreshold()
        selector = selector.fit(dataset)
        dataset = selector.transform(dataset)
        print(dataset.features)
        
    
    if exercise == "3":
        
        file_path = "./datasets/iris.csv"
        label = "class"
            
        dataset = Dataset.read(file_path = file_path, label= label)
        
        tree = DecisionTrees(dataset)

        # Print the tree
        print(repr(tree))
        

if __name__ == '__main__':
    
    exercise = "3"

    main (exercise)