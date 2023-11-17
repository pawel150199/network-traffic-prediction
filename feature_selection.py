import numpy as np
from sklearn.tree import DecisionTreeRegressor

def feature_selection(X: np.array, y: np.array, random_state: str = None) -> np.array:
    """
    Find the most valuable features in all dataset

    Args:
        X (np.array): Data
        y (np.array): Labels
        random_state (str, optional): Random State. Defaults to None.

    Returns:
        np.array: Returns transformed data with selected features
    """    
    
    tree = DecisionTreeRegressor(random_state=random_state)
    tree.fit(X, y)

    importances = tree.feature_importances_
    indices= np.array([idx for idx, importance in enumerate(importances) if importance > 0])
    X_t = X[:, indices]

    return X_t