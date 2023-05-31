import sys
import os
import numpy as np
import csv
from typing import Tuple, Sequence
sys.path.append('./datasets')


class Dataset:


    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
        """
        Initializes the Dataset object.
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray, optional): The label array. Defaults to None.
            features (Sequence[str], optional): The list of feature names. Defaults to None.
            label (str, optional): The name of the label. Defaults to None.
        """
        if X is None:
            raise ValueError("X cannot be None")

        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)
    
        if y is not None and label is None:
            label = "y"
    
        self.X = X
        self.y = y
        self.features = features
        self.label = label
        self.feature_types = []
        self.categories = {}
        self.classes_dict = None
        
        # Infer the feature types and categories
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.issubdtype(col.dtype, np.number):
                self.feature_types.append('numerical')
            else:
                self.feature_types.append('categorical')
                categories = self.get_categories(X, i)
                self.categories[features[i]] = categories
    

    def read(file_path: str, label: str = None):
        """
        Reads a file and returns a Dataset object.
        
        Args:
            file_path (str): The path to the file.
            label (str, optional): The name of the label. Defaults to None.
        
        Returns:
            Dataset: The dataset object.
        """
        if os.path.isfile(file_path) is False:
            raise ValueError("File does not exist")
        
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype='str') 
        features = np.genfromtxt(file_path, delimiter=',', max_rows=1, dtype='str')
        
        X = np.zeros((data.shape[0], data.shape[1]-1))
        for i in range(X.shape[1]):
            col = data[:, i]
            if np.issubdtype(col.dtype, np.number):
                X[:, i] = col.astype(float)
            else:
                categories = np.unique(col)
                categories = categories[categories != '']

                indices = [np.where(categories == x)[0] for x in col]
                X[:, i] = np.array([ind[0] if ind.size > 0 else -1 for ind in indices])

        if label is not None:
            label_index = np.where(features == label)[0][0]
            labels = data[:, label_index]
            label_dict = {label: i for i, label in enumerate(np.unique(labels))}
            y = np.vectorize(label_dict.get)(labels).astype('int')
        else:
            y = None
            
        return Dataset(X, y, features, label)


    def write(self, file_path: str):
        """
        Writes the dataset to a CSV file.
        
        Args:
            file_path (str): The path to the output file.
        """
        if self.y is None:
            data = self.X
        else:
            data = np.concatenate((self.X, self.y.reshape(-1, 1)), axis=1)

        with open(file_path, 'w', newline='') as file:

            writer = csv.writer(file)
            
            if self.features:
                writer.writerow(self.features)
            
            writer.writerows(data)
        
        print("File written successfully!")


    def get_X(self) -> np.ndarray:
        """
        Returns the feature matrix X.
        """
        return self.X
    

    def get_y(self) -> np.ndarray:
        """
        Returns the feature matrix y.
        """
        return self.y
    
    
    def get_categories(self, data, col_idx) -> np.ndarray:
        """
        Args:
            data (np.ndarray): dataset
            col_idx (int): column index

        Returns:
            np.ndarray: categories of the dataset
        """
        col = data[:, col_idx]
        uniques = np.unique(col)
        categories = uniques[np.vectorize(lambda x: x != '')(uniques)]
        return categories
    

    def get_shape(self) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: shape of the dataset
        """
        return self.X.shape


    def get_classes(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: unique classes in the dataset
        """
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)


    def get_mean(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: mean of each feature
        """
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmean(self.X[:, numerical_features], axis=0)


    def get_variance(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: variance of each feature
        """
        
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanvar(self.X[:, numerical_features], axis=0)


    def get_median(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: median of each feature
        """
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmedian(self.X[:, numerical_features], axis=0)


    def get_min(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: minimum of each feature
        """
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmin(self.X[:, numerical_features], axis=0)


    def get_max(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: maximum of each feature
        """
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmax(self.X[:, numerical_features], axis=0)


    def get_null_values(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: number of null values of each feature
        """
        return np.sum(np.isnan(self.X), axis=0)

            
    def set_X(self, X):
        """
        Args:
            X: feature matrix
        """
        self.X = X
    
    
    def set_y(self, y):
        """
        Args:
            y: label matrix
        """
        self.y = y
    
    
    def set_categories(self, col_idx, categories):
        """
        Args:
            col_idx (int): the index of the column
            categories (np.ndarray): the array of categories
        """
        self.categories[self.features[col_idx]] = categories
    
    
    def set_shape(self, shape):
        """
        Args:
            shape (Tuple[int, int]): the shape of the dataset as a tuple (n_rows, n_columns)
        """
        self.shape = shape


    def set_classes(self, classes):
        """
        Args:
            classes (np.ndarray): the array of classes
        """
        self.classes = classes
    
    
    def set_mean(self, mean):
        """
        Args:
            mean (np.ndarray): the array of means
        """
        self.mean = mean
    
    
    def set_variance(self, variance):
        """
        Args:
            variance (np.ndarray): the array of variances
        """
        self.variance = variance
    
    
    def set_median(self, median):
        """
        Args:
            median (np.ndarray): the array of medians
        """
        self.median = median
    
    
    def set_min(self, min):
        """
        Args:
            min (np.ndarray): the array of minimums
        """
        self.min = min
    
    
    def set_max(self, max):
        """
        Args:
            max (np.ndarray): the array of maximums
        """
        self.max = max
    
    
    def set_null_values(self, null_values):
        """
        Args:
            null_values (np.ndarray): the array of null values
        """
        self.null_values = null_values
    

    def replace_null_values(self):
        """
        Replaces all null values with mean
        """
        col_mean = np.nanmean(self.X, axis=0)
        self.X = np.nan_to_num(self.X, nan=col_mean)
    

    def describe(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: a description of the dataset
        """
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        n_numerical_features = len(numerical_features)
        stats = np.zeros((5, n_numerical_features))
        stats[0] = np.mean(self.X[:, numerical_features], axis=0)
        stats[1] = np.median(self.X[:, numerical_features], axis=0)
        stats[2] = np.min(self.X[:, numerical_features], axis=0)
        stats[3] = np.max(self.X[:, numerical_features], axis=0)
        stats[4] = np.var(self.X[:, numerical_features], axis=0)
        return stats

if __name__ == '__main__':
     file_path = "./datasets/iris.csv"
     label = "class"
     
     dataset = Dataset.read(file_path = file_path, label= label)
     
     print("X:", dataset.get_X())
     print("y:", dataset.get_y())
     print("Categories:", dataset.get_categories(dataset.get_X(), 0))
     print("Shape", dataset.get_shape())
     print("Classes:", dataset.get_classes())
     print("Mean:", dataset.get_mean())
     print("Variance:", dataset.get_variance())
     print("Median:", dataset.get_median())
     print("Min:", dataset.get_min())
     print("Max:", dataset.get_max())
     print("NULLS:", dataset.get_null_values())
     print("Describe:", dataset.describe())
