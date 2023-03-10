from typing import Tuple, Sequence

import numpy as np

class Dataset:

    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
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
    
    #Read a file and returns the Dataset(X, y, features, label)
    def read(file_path: str, label: str = None):
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
                #print(categories)

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

    #Returns the categories of the 
    def get_categories(self, data, col_idx):
        col = data[:, col_idx]
        uniques = np.unique(col)
        categories = np.delete(uniques, uniques == '')
        return categories

    # Returns the shape of the dataset
    def get_shape(self) -> Tuple[int, int]:
        return self.X.shape

    # Returns the unique classes in the dataset
    def get_classes(self) -> np.ndarray:
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)

    # Returns the mean of each feature
    def get_mean(self) -> np.ndarray:
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmean(self.X[:, numerical_features], axis=0)


    # Returns the variance of each feature
    def get_variance(self) -> np.ndarray:
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanvar(self.X[:, numerical_features], axis=0)

    # Returns the median of each feature
    def get_median(self) -> np.ndarray:
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmedian(self.X[:, numerical_features], axis=0)

    # Returns the minimum of each feature
    def get_min(self) -> np.ndarray:
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmin(self.X[:, numerical_features], axis=0)

    # Returns the maximum of each feature
    def get_max(self) -> np.ndarray:
        if len(self.feature_types) == 0:
            raise ValueError("Dataset feature types have not been inferred")
        numerical_features = [i for i, ft in enumerate(self.feature_types) if ft == 'numerical']
        return np.nanmax(self.X[:, numerical_features], axis=0)

    # Returns the number of null values of each feature
    def get_null_values(self) -> np.ndarray:
        return np.sum(np.isnan(self.X), axis=0)

    # Replaces all null values with mean
    def replace_null_values(self):
        col_mean = np.nanmean(self.X, axis=0)
        self.X = np.nan_to_num(self.X, nan=col_mean)
    
    # Returns a description of the dataset
    def describe(self) -> np.ndarray:
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

def test1():
    dataset = Dataset.read(file_path="./datasets/iris.csv", label="class")

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

def test2():
    dataset = Dataset.read("./datasets/titanic.csv", label="Survived")

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

#test1()
test2()
