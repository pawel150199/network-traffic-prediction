from typing import Any
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from helpers.import_data import import_data

class FeatureSelection(object):
    def __init__(self, random_state: int = None) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Find the most valuable features in all dataset

        Args:
            X (np.array): Dataset to fit
            y (np.array): Labels of X dataset
            random_state (str, optional): Random State. Defaults to None.

        Returns:
            np.array: Returns transformed data with selected features
        """    
        
        tree = DecisionTreeRegressor(random_state=self.random_state)
        tree.fit(X, y)

        self._importances = tree.feature_importances_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform numpy ndarray and select features

        Args:
            X (np.ndarray): Dataset to transformation

        Returns:
            np.ndarray: Transformed X ndarray
        """        
        _indices = np.array([idx for idx, importance in enumerate(self._importances) if importance > 0])
        X_t = X[:, _indices]
        return X_t
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform X ndarray

        Args:
            X (np.ndarray): Dataset
            y (np.ndarray): Labels of X dataset

        Returns:
            np.ndarray: Transformed X ndarray
        """        
        tree = DecisionTreeRegressor(random_state=self.random_state)
        tree.fit(X, y)

        self._importances = tree.feature_importances_
        _indices = np.array([idx for idx, importance in enumerate(self.importances) if importance > 0])

        X_t = X[:, _indices]

        return X_t
