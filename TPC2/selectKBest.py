import numpy as np

from TPC1.dataset import Dataset
from typing import Callable

class SelectKBest:

    def __init__(self, score_func: Callable, k: int):

        if k <= 0:
            raise ValueError("k must be positive")

        # parameters
        self.score_func = score_func
        self.k = k

        # attributes
        self.f = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        pass

    def transform(self, dataset: Dataset) -> Dataset:
        pass

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)

