import numpy as np

def accuracy(y_pred: np.array, y_real: np.array) -> int:
    """Accuracy for regression

    Args:
        y_pred (np.array): predicted values
        y_real (np.array): real values

    Returns:
        int: accuracy score for regression
    """    
    acc = 0
    for i in range(len(y_pred)):
        partial_acc = 100 * (1 - abs(y_pred[i] - y_real[i]) / y_real[i])
        #print(f"Partial Accuracy: {partial_acc}")
        acc += partial_acc
    final_acc = acc / len(y_pred)
    return final_acc