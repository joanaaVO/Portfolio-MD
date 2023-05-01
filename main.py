import sys
import numpy as np

sys.path.append('./datasets')
sys.path.append('./TPC1')
sys.path.append('./TPC2')
sys.path.append('./TPC3')
sys.path.append('./TPC4')
sys.path.append('./TPC5')
sys.path.append('./TPC6')

from dataset import Dataset
from f_classif import F_Classif
from f_regression import F_Regression
from selectKBest import SelectKBest
from variance_threshold import VarianceThreshold
from decisionTree import DecisionTrees
from prism import Prism
from naiveBayes import NaiveBayes
from apriori import Apriori
from apriori import TransactionDataset
from mlp import MLP
from datasetMLP import DatasetMLP

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


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
    
    if exercise == "4-prism":
        
        print("Não sei o que colocar")
    
    if exercise == "4-naiveBayes":
        
        # Load the dataset using the Dataset class
        data = Dataset.read(file_path="./datasets/iris.csv", label="class")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = Dataset.train_test_split(data.X, data.y, test_size=0.2, random_state=2023)

        # Create the NaiveBayes model
        model = NaiveBayes()

        # Evaluate the model using cross-validation
        scores = Dataset.cross_val_score(model, data.X, data.y, cv=5)
        
        # Print the mean and standard deviation of the cross-validation scores
        print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
            
        
    if exercise == "5":

        transactions = [    
        ['milk', 'bread', 'eggs'],
        ['bread', 'sugar', 'coffee'],
        ['bread', 'milk', 'sugar', 'coffee'],
        ['bread', 'sugar', 'eggs'],
        ['milk', 'sugar', 'coffee'],
        ['milk', 'bread', 'sugar', 'coffee'],
        ['milk', 'bread', 'sugar', 'eggs'],
        ['milk', 'bread', 'sugar', 'coffee', 'eggs'],
        ['bread', 'sugar', 'eggs', 'coffee'],
        ['milk', 'sugar', 'eggs']
        ]

        min_support = 0.4
        min_confidence = 0.5

        transaction_dataset = TransactionDataset(transactions)

        apriori = Apriori(transaction_dataset, min_support, min_confidence)
        apriori.fit()

        # Print frequent itemsets
        print("Frequent itemsets:")
        for itemset, support in apriori.itemsets.items():
            print(f"{itemset}: {support}")
        
        
        # Print association rules
        print("\nAssociation rules:")
        for rule, confidence in apriori.rules.items():
            premise, conclusion = rule
            print(f"{premise} => {conclusion}: {confidence}")

    if exercise == "6":

        ds= DatasetMLP("./datasets/xnor.csv")
        nn = MLP(ds, 2)
        w1 = np.array([[-30,20,20],[10,-20,-20]])
        w2 = np.array([[-10,20,20]])
        nn.setWeights(w1, w2)
        print( nn.predict(np.array([0,0]) ) )
        print( nn.predict(np.array([0,1]) ) )
        print( nn.predict(np.array([1,0]) ) )
        print( nn.predict(np.array([1,1]) ) )
        print(nn.costFunction())


if __name__ == '__main__':
    
    exercise = "6"

    main (exercise)