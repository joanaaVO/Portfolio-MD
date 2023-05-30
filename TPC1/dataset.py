import sys
import numpy as np
from typing import Tuple, Sequence
sys.path.append('./datasets')


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

    #write the dataset in a file
    def write(self, file_path: str):
        if self.y is None:
            data = self.X
        else:
            data = np.concatenate((self.X, self.y.reshape(-1, 1)), axis=1)
        np.savetxt(file_path, data, delimiter=',', fmt='%s')
        print("File written successfully!")

    #Returns the matrix X
    def get_X(self) -> np.ndarray:
        return self.X
    
    #Returns the matrix y
    def get_y(self) -> np.ndarray:
        return self.y
    
    #Returns the categories of the dataset
    def get_categories(self, data, col_idx) -> np.ndarray:
        col = data[:, col_idx]
        uniques = np.unique(col)
        categories = uniques[np.vectorize(lambda x: x != '')(uniques)]
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

            
    def set_X(self, X):
        self.X = X
    
    def set_y(self, y):
        self.y = y
    
    def set_categories(self, col_idx, categories):
        self.categories[self.features[col_idx]] = categories
    
    def set_shape(self, shape):
        self.shape = shape

    def set_classes(self, classes):
        self.classes = classes
    
    def set_mean(self, mean):
        self.mean = mean
    
    def set_variance(self, variance):
        self.variance = variance
    
    def set_median(self, median):
        self.median = median
    
    def set_min(self, min):
        self.min = min
    
    def set_max(self, max):
        self.max = max
    
    def set_null_values(self, null_values):
        self.null_values = null_values
    

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
     
     dataset.write("./datasets/iris_test.csv")