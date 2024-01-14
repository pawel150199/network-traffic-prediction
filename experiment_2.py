from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import warnings

from helpers.import_data import import_data
from helpers.evaluation import Evaluator
from LSTMModel import LSTM


warnings.filterwarnings("ignore")

RANDOM_STATE=1410

def main(name: str, dataset: str):

    # Import data
    X, y = import_data(dataset)

    # Clasificators
    clfs = {
        "CART" : DecisionTreeRegressor(random_state=RANDOM_STATE),
        "KNN" : KNeighborsRegressor(),
        "SVR" : SVR(kernel="poly"),
        "RF" : RandomForestRegressor(random_state=RANDOM_STATE),
        "LSTM" : LSTM().build_model()
    }

    # Metrics
    metrics = {
        "MSE" : mean_squared_error,
        "MAE" : mean_absolute_error
    }

    ev = Evaluator(storage_dir="results", X=X, y=y, random_state=RANDOM_STATE, metrics=metrics)
    ev.evaluate(clfs, result_name=f"scores_{name}")
    ev.process_ranks(result_name=f"ranks_{name}")

if __name__ == "__main__":
    main(name="simple_models", dataset="Euro28")


